"""Range 下载边界使用 requests 替身；不联网、不读取真实数据、不依赖 thriftpy2。"""

import hashlib
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "collect_p4_test", Path(__file__).resolve().parents[1] / "scripts/collect_p4.py"
)
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


class FakeRaw:
    def __init__(self, body: bytes, forbidden: bool = False) -> None:
        self.body = body
        self.forbidden = forbidden
        self.read_sizes = []

    def read(self, size: int) -> bytes:
        if self.forbidden:
            raise AssertionError("拒绝状态码或 Content-Range 前不能读取响应 body")
        self.read_sizes.append(size)
        return self.body[:size]


class FakeResponse:
    def __init__(
        self,
        status: int = 206,
        content_range: str | None = "bytes 10-13/100",
        body: bytes = b"data",
        forbid_body: bool = False,
    ) -> None:
        self.status_code = status
        self.headers = {} if content_range is None else {"Content-Range": content_range}
        self.raw = FakeRaw(body, forbidden=forbid_body)
        self.closed = False

    @property
    def content(self) -> bytes:
        raise AssertionError("不能通过 response.content 无上限读取 Range body")

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise collector.requests.HTTPError(f"HTTP {self.status_code}")

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args) -> None:
        self.closed = True


def install_responses(monkeypatch, responses: list[FakeResponse]) -> list[dict]:
    calls = []

    def get(url, **kwargs):
        calls.append(dict(url=url, **kwargs))
        if len(calls) > len(responses):
            raise AssertionError("超过预算后不应再发起请求")
        return responses[len(calls) - 1]

    monkeypatch.setattr(collector.requests, "get", get)
    return calls


@pytest.mark.parametrize(
    ("status", "content_range"),
    [
        (200, "bytes 10-13/100"),
        (206, None),
        (206, "bytes 0-3/100"),
        (206, "bytes 10-14/100"),
        (206, "bytes 10-13/101"),
        (206, "bytes 10-13/*"),
    ],
)
def test_rejects_wrong_status_or_range_before_reading_body(
    monkeypatch, status: int, content_range: str | None
) -> None:
    response = FakeResponse(status=status, content_range=content_range, forbid_body=True)
    calls = install_responses(monkeypatch, [response])
    source = collector.RangeSource("https://official.example/file.parquet", size=100, budget=4)

    with pytest.raises(RuntimeError, match="精确 Range"):
        source.read(10, 4)
    assert len(calls) == 1 and calls[0]["stream"] is True
    assert response.closed and response.raw.read_sizes == []
    assert source.transferred == 0 and source.ranges == []


def test_rejects_http_failure_without_reading_body(monkeypatch) -> None:
    response = FakeResponse(status=403, forbid_body=True)
    install_responses(monkeypatch, [response])
    source = collector.RangeSource("https://official.example/file.parquet", size=100, budget=4)
    with pytest.raises(collector.requests.HTTPError):
        source.read(10, 4)
    assert response.closed and response.raw.read_sizes == []
    assert source.transferred == 0 and source.ranges == []


@pytest.mark.parametrize("budget", [0, 3])
def test_insufficient_budget_rejects_before_http_request(monkeypatch, budget: int) -> None:
    calls = install_responses(monkeypatch, [])
    source = collector.RangeSource("https://official.example/file.parquet", size=100, budget=budget)
    with pytest.raises(RuntimeError, match="传输上限"):
        source.read(10, 4)
    assert calls == [] and source.transferred == 0 and source.ranges == []


def test_success_records_exact_ranges_hashes_and_stops_at_budget(monkeypatch) -> None:
    responses = [
        FakeResponse(content_range="bytes 0-5/100", body=b"abcdef"),
        FakeResponse(content_range="bytes 20-23/100", body=b"WXYZ"),
    ]
    calls = install_responses(monkeypatch, responses)
    source = collector.RangeSource("https://official.example/file.parquet", size=100, budget=10)
    assert source.read(0, 6) == b"abcdef"
    assert source.read(20, 4) == b"WXYZ"
    assert source.transferred == 10
    assert source.ranges == [
        dict(start=0, length=6, sha256=hashlib.sha256(b"abcdef").hexdigest()),
        dict(start=20, length=4, sha256=hashlib.sha256(b"WXYZ").hexdigest()),
    ]
    assert [call["headers"] for call in calls] == [
        {"Range": "bytes=0-5"},
        {"Range": "bytes=20-23"},
    ]
    assert all(call["stream"] is True for call in calls)
    assert all(response.closed for response in responses)
    assert [response.raw.read_sizes for response in responses] == [[7], [5]]
    with pytest.raises(RuntimeError, match="传输上限"):
        source.read(24, 1)
    assert len(calls) == 2 and source.transferred == 10


def test_cumulative_budget_rejects_before_request_that_would_exceed_it(monkeypatch) -> None:
    response = FakeResponse(content_range="bytes 0-5/100", body=b"abcdef")
    calls = install_responses(monkeypatch, [response])
    source = collector.RangeSource("https://official.example/file.parquet", size=100, budget=10)
    source.read(0, 6)
    with pytest.raises(RuntimeError, match="传输上限"):
        source.read(6, 5)
    assert len(calls) == 1 and source.transferred == 6
    assert len(source.ranges) == 1


@pytest.mark.parametrize("body", [b"abc", b"abcde"])
def test_exact_range_headers_cannot_hide_short_or_overlong_body(monkeypatch, body: bytes) -> None:
    response = FakeResponse(body=body)
    install_responses(monkeypatch, [response])
    source = collector.RangeSource("https://official.example/file.parquet", size=100, budget=4)
    with pytest.raises(RuntimeError, match="实际字节数"):
        source.read(10, 4)
    assert response.closed and response.raw.read_sizes == [5]
    assert source.transferred == 0 and source.ranges == []
