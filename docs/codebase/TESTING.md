## 一、测试工具与命令

`pyproject.toml` 声明 pytest `>=8,<10`，当前 `.venv` 为 9.2.0。断言使用 pytest 和 `torch.testing.assert_close`，替身通过 `monkeypatch` 注入。

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m pytest -q
python -m pytest -q tests/test_thinker.py tests/test_omni.py tests/test_omni_generation.py
python -m pytest -q tests/test_dataset.py tests/test_codec.py tests/test_streaming.py
python -m pytest -q tests/test_p4_data.py tests/test_sft.py tests/test_sft_cli.py
ruff check omni tests scripts
ruff format --check omni tests scripts
```

## 二、布局和运行环境

- pytest 从 `tests/` 发现用例，现有文件统一为 `test_*.py`，`addopts = "-ra"` 汇总跳过原因。fixture 定义在对应测试模块内。
- Ubuntu CI 先安装 CPU-only PyTorch，再安装 `.[dev,media,plots]`。测试阶段启用 HF 离线变量，MPS 基础算术按可用性跳过。
- `soundfile`/SciPy 波形 IO 与 matplotlib 训练 CLI 测试通过 `importorskip` 处理可选依赖。装入对应 extras 后可覆盖这些路径。

## 三、验证范围

| 级别 | 现有覆盖 | 证据 |
| --- | --- | --- |
| 单元 | Dense/MoE、HF 权重回环、Cache/左 padding、模态索引、码本延迟、loss 与梯度 | `test_thinker.py`、`test_omni_generation.py`、`test_dataset.py`、`test_codec.py`、`test_sft.py` |
| 本地集成 | 临时 Parquet 和 HF tokenizer、极小 Qwen/Mimi、HTTP Range 替身、训练 CLI 保存/恢复 | `test_p1_qwen.py`、`test_dataset.py`、`test_streaming.py`、`test_collect_p4.py`、`test_sft_cli.py` |
| 真实资产 | MiniMind 严格权重加载、Qwen3-0.6B 文本、SenseVoice/SigLIP2 混合前向、Mimi 波形与 P5 流式基准 | `scripts/check_minimind.py`、`check_qwen3.py`、`check_p2.py`、`check_p3.py`、`check_p5.py` |
| 历史训练 | 64/8 条真实子集、小主干 SFT、优化器及 RNG 恢复 | `docs/stages/P4.md`、`experiments/p4/` |

复核真实资产时使用已有本地路径和新的输出位置。`check_p4.py` 的 smoke 路径会执行优化器更新，R0 的只读 checkpoint 核查与该训练入口分别记录。

## 四、隔离方式

- 测试模型和样本以极小配置构造，临时文件使用 `tmp_path`。Mimi 流式测试固定 seed，覆盖尾块冲刷、取消和独立状态。
- SenseVoice/SigLIP2 测试替换构造时的模块入口。Transformers 的 `LazyModule` 可能在其他导入后替换，测试在 `sys.modules` 中隔离实际使用的入口，并由 pytest 恢复。
- Range 测试替换 `requests.get`，检查错误状态、范围和预算是否在读取 body 前被拒绝。普通测试不依赖真实网络调用或大型权重。

## 五、覆盖率与实测记录

- pytest 配置未声明覆盖率阈值。当前交付信号是各阶段的通过数、专项断言、数值误差与报告。[TODO] 当前代码覆盖率尚无审计数据。
- P5 历史记录为真实主机 151 项通过，包含 MPS 基础运算。P4 历史训练 checkpoint 与 P5 CPU 复核均有原始报告。
- R0 的普通回归与真实资产复核以本阶段结果为准。v2 Qwen3-1.7B Backend、联合 SFT、Adaptive Bridge 和 GRPO 的验收安排见 v2 主规划。 [R0 阶段报告](../stages/R0.md)

## 六、证据

- [pytest 配置](../../pyproject.toml)、[CI 回归](../../.github/workflows/ci.yml)、[设备测试](../../tests/test_environment.py)
- [流式测试](../../tests/test_streaming.py)、[下载边界](../../tests/test_collect_p4.py)、[训练恢复测试](../../tests/test_sft_cli.py)
- [P4 历史](../stages/P4.md)、[P5 历史](../stages/P5.md)
