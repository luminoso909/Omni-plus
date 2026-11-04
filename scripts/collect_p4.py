"""显式收集官方 P4 小样本：只读取 Parquet 字典和前缀页，不下载完整文件。"""

import argparse
import copy
import hashlib
import json
from io import BytesIO
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests

from omni.data.prepare import media_hashes

ROOT = Path(__file__).resolve().parents[1]
REPO = "jingyaogong/minimind-o_dataset"
REVISION = "d6588e12ac2ac8ced65eb58a7d7b3eef4aa220de"
FORMAT_URL = "https://raw.githubusercontent.com/apache/parquet-format/apache-parquet-format-2.11.0/"
FORMAT_SHA256 = "bf8e93151ebe8eebbb8f3c829d97f0dd537dc43f411d03df15837ffaa2a749cf"
FILES = {"t2a": "sft_t2a_mini.parquet", "a2a": "sft_a2a_mini.parquet", "i2t": "sft_i2t.parquet"}


class RangeSource:
    def __init__(self, url: str, size: int, budget: int) -> None:
        self.url, self.size, self.budget = url, size, budget
        self.transferred = 0
        self.ranges = []

    def read(self, start: int, length: int) -> bytes:
        if length < 1 or start < 0 or start + length > self.size:
            raise ValueError("请求范围超出源文件边界。")
        if self.transferred + length > self.budget:
            raise RuntimeError("达到本次小样本传输上限；停止，禁止回退到完整下载。")

        # 在读取 body 之前检查 206 与 Content-Range，拒绝服务端忽略 Range 的整文件响应
        end = start + length - 1
        with requests.get(
            self.url, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=(15, 60)
        ) as response:
            response.raise_for_status()
            if response.status_code != 206 or response.headers.get("Content-Range") != (
                f"bytes {start}-{end}/{self.size}"
            ):
                raise RuntimeError("官方服务未保持精确 Range；未读取响应 body，禁止完整下载。")
            value = response.raw.read(length + 1)
        if len(value) != length:
            raise RuntimeError("Range 实际字节数与请求不一致。")

        self.transferred += length
        self.ranges.append(
            dict(start=start, length=length, sha256=hashlib.sha256(value).hexdigest())
        )
        return value


def read_thrift(value: bytes, struct):
    from thriftpy2.protocol.compact import TCompactProtocol
    from thriftpy2.transport import TMemoryBuffer

    stream = TMemoryBuffer(value)
    result = struct()
    result.read(TCompactProtocol(stream))
    return result, len(value) - len(stream.read(len(value)))


def write_thrift(value) -> bytes:
    from thriftpy2.protocol.compact import TCompactProtocol
    from thriftpy2.transport import TMemoryBuffer

    stream = TMemoryBuffer()
    value.write(TCompactProtocol(stream))
    return stream.getvalue()


def prefix_table(source: RangeSource, parquet_format, rows: int, pages: int = 1) -> pa.Table:
    """取首个 rowgroup 的各列前缀页，仅迭代前 rows 条，再写成有效的小 Parquet。"""
    tail = source.read(source.size - 8, 8)
    if tail[-4:] != b"PAR1":
        raise ValueError("官方文件不是 Parquet。")
    footer_size = int.from_bytes(tail[:4], "little")
    footer = source.read(source.size - footer_size - 8, footer_size)
    original, _ = read_thrift(footer, parquet_format.FileMetaData)
    metadata = copy.deepcopy(original)
    metadata.row_groups = metadata.row_groups[:1]
    group = metadata.row_groups[0]
    metadata.num_rows = group.num_rows
    content = BytesIO(b"PAR1")
    content.seek(4)

    # 原始单组音频文件可达 GB；只保留字典页和少量数据页，避免读取整个列块
    for column in group.columns:
        column_meta = column.meta_data
        start = column_meta.dictionary_page_offset
        start = column_meta.data_page_offset if start is None else start
        position, data_pages, parts = start, 0, []
        column_end = start + column_meta.total_compressed_size
        while position < column_end and data_pages < pages:
            header_bytes = source.read(position, min(16384, column_end - position))
            header, header_size = read_thrift(header_bytes, parquet_format.PageHeader)
            length = header_size + header.compressed_page_size
            page = header_bytes[:length]
            if len(page) < length:
                page += source.read(position + len(page), length - len(page))
            parts.append(page)
            position += length
            data_pages += int(header.type in (0, 3))

        chunk = b"".join(parts)
        offset = content.tell() - start
        column.file_offset = content.tell()
        column_meta.data_page_offset += offset
        if column_meta.dictionary_page_offset is not None:
            column_meta.dictionary_page_offset += offset
        column_meta.total_compressed_size = len(chunk)
        content.write(chunk)
    group.total_compressed_size = content.tell() - 4
    encoded = write_thrift(metadata)
    content.write(encoded + len(encoded).to_bytes(4, "little") + b"PAR1")

    # 这个内存视图只用于首批读取，绝不作为完整源文件保存；持久化的是重新编码的小表
    reader = pq.ParquetFile(pa.BufferReader(content.getvalue()))
    first = next(reader.iter_batches(batch_size=rows))
    if first.num_rows != rows:
        raise ValueError("前缀页未覆盖所需样本，不重复或补零凑数。")
    return pa.Table.from_batches([first])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-per-task", type=int, default=32)
    parser.add_argument(
        "--scan-rows", type=int, default=128, help="前缀候选行数，去除共享媒体后保留少量行"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("cache/p4/sources"))
    parser.add_argument("--report", type=Path, default=Path("experiments/p4/subset-assets.json"))
    parser.add_argument("--max-transfer-mb", type=int, default=64)
    args = parser.parse_args()
    output_dir, report_path = ROOT / args.output_dir, ROOT / args.report
    if not 25 <= args.rows_per_task <= 64:
        parser.error("每个任务仅允许 25–64 条，避免意外收集大数据。")
    if not args.rows_per_task <= args.scan_rows <= 256:
        parser.error("scan-rows 必须覆盖保留行数且不超过 256。")
    if report_path.exists() or any(output_dir.glob("*.parquet")):
        parser.error("目标已有资产或报告；使用新路径保留旧记录。")

    import thriftpy2

    # 仅下载 Apache 官方格式定义及许可；模型/数据代码不来自这里
    format_dir = output_dir.parent / "format"
    format_dir.mkdir(parents=True, exist_ok=True)
    for name, remote in (
        ("parquet.thrift", "src/main/thrift/parquet.thrift"),
        ("LICENSE", "LICENSE"),
        ("NOTICE", "NOTICE"),
    ):
        path = format_dir / name
        if not path.exists():
            response = requests.get(FORMAT_URL + remote, timeout=30)
            response.raise_for_status()
            path.write_bytes(response.content)
    if hashlib.sha256((format_dir / "parquet.thrift").read_bytes()).hexdigest() != FORMAT_SHA256:
        raise ValueError("Apache Parquet 格式定义 SHA 不符。")
    parquet_format = thriftpy2.load(
        str(format_dir / "parquet.thrift"), module_name="p4_parquet_thrift"
    )
    base = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/"
    metadata_response = requests.get(
        f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}", timeout=30
    )
    metadata_response.raise_for_status()
    source_files = {item["path"]: item for item in metadata_response.json()}
    card = requests.get(base + "README.md", timeout=30)
    card.raise_for_status()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "README.upstream.md").write_bytes(card.content)

    assets = []
    total_transferred = 0
    seen_media = set()
    for task, filename in FILES.items():
        info = source_files[filename]
        source = RangeSource(
            base + filename, info["size"], args.max_transfer_mb * 1024**2 - total_transferred
        )
        table = prefix_table(source, parquet_format, args.scan_rows)
        records = []
        for index, record in enumerate(table.to_pylist()):
            media = media_hashes(record)
            if media & seen_media:
                continue
            seen_media.update(media)
            record.update(_source_row=index, _source_file=filename, _source_revision=REVISION)
            records.append(record)
            if len(records) == args.rows_per_task:
                break
        if len(records) != args.rows_per_task:
            raise ValueError(f"{task} 前缀中独立媒体不足，不关闭去重凑数。")
        target = output_dir / filename
        pq.write_table(pa.Table.from_pylist(records), target)
        total_transferred += source.transferred
        assets.append(
            dict(
                task=task,
                source_file=filename,
                source_size=info["size"],
                official_full_file_sha256=info["lfs"]["oid"],
                full_file_downloaded=False,
                full_file_hash_verified=False,
                original_rows=[record["_source_row"] for record in records],
                retained_rows=len(records),
                local_path=str(target.relative_to(ROOT)),
                local_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                transferred_bytes=source.transferred,
                ranges=source.ranges,
            )
        )
        print(
            f"{task}: retained {len(records)} rows, Range transfer {source.transferred} bytes",
            flush=True,
        )

    report = dict(
        repo_id=REPO,
        revision=REVISION,
        declared_licenses=["apache-2.0", "gpl-3.0"],
        license_note="官方卡未逐文件区分两项许可；子集仅本地诊断，不公开再分发。",
        extraction="Dictionary pages and initial data pages; first batch decoded by PyArrow.",
        retained_candidate_rows=sum(item["retained_rows"] for item in assets),
        transferred_bytes=total_transferred,
        prefix_pages_may_contain_additional_encoded_rows=True,
        full_dataset_downloaded=False,
        format=dict(url=FORMAT_URL, sha256=FORMAT_SHA256, thriftpy2_version=thriftpy2.__version__),
        files=assets,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
