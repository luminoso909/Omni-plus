## 一、运行时

| 项目 | R0 工作区所见 | 依据 |
| --- | --- | --- |
| 语言与解释器 | Python 3.12，项目范围 `>=3.12,<3.13`。当前 `.venv` 为 3.12.14、macOS arm64 | `pyproject.toml`、`.venv/bin/python` |
| 安装与锁定 | `pip` 可编辑安装，`uv.lock` 记录解析结果和 extras | `README.md`、`uv.lock` |
| 构建与分发 | `setuptools.build_meta`，要求 `setuptools>=77`。分发名 `Omni-plus`，导入名 `omni`，包发现范围 `omni`、`omni.*` | `pyproject.toml` |

## 二、运行依赖

表中版本为当前 `.venv` 已安装版本，声明范围以 `pyproject.toml` 为准。

| 依赖 | 当前版本 | 用途与依据 |
| --- | --- | --- |
| PyTorch | 2.15.0 | `nn.Module`、张量、优化器及 AMP，见 `omni/models/`、`omni/training/sft.py` |
| Transformers / Hugging Face Hub | 4.57.6 / 0.36.2 | HF 模型与 tokenizer，本地 Mimi/SigLIP 加载及显式 Qwen 资产获取，见 `scripts/check_qwen3.py`、`scripts/check_p3.py` |
| PyArrow / NumPy | 26.0.0 / 2.5.4 | Parquet 与数据、随机状态转换，见 `omni/data/prepare.py`、`omni/training/sft.py` |
| jaxtyping / PyYAML | 0.3.11 / 6.0.3 | 张量形状标注、训练配置读取，见 `omni/models/model_omni.py`、`scripts/train_sft.py` |
| safetensors / datasets / einops | 0.8.0 / 4.8.5 / 0.8.2 | 基础依赖声明。现有 P4 的 Parquet 读取由 PyArrow 执行，见 `pyproject.toml`、`omni/data/omni_dataset.py` |
| `media` | Pillow 13.0.0、soundfile 0.14.0、SciPy 1.18.1 | 图像读取、波形 IO 和有理数重采样，见 `omni/models/encoders/model_encoders.py`、`omni/inference/codec.py` |
| `encoders` | FunASR 1.4.17、torchaudio 2.11.1 | 本地 SenseVoice 适配依赖，见 `pyproject.toml`、`omni/models/encoders/model_encoders.py` |
| `plots` / `data` | matplotlib 3.11.2、requests 2.34.2、thriftpy2 0.5.2 | Loss 曲线及显式 Parquet 前缀收集，见 `scripts/train_sft.py`、`scripts/collect_p4.py` |

## 三、开发工具

| 工具 | 用途 | 依据 |
| --- | --- | --- |
| pytest 9.2.0 | 普通回归测试，发现范围为 `tests/` | `pyproject.toml` |
| Ruff 0.17.3 | `omni tests scripts` 的格式与静态检查 | `pyproject.toml`、`.github/workflows/ci.yml` |
| GitHub Actions | Ubuntu CPU 回归，先从 PyTorch CPU 索引安装 torch，再安装项目 extras | `.github/workflows/ci.yml` |

`uv.lock` 含按平台选择的 PyTorch 依赖，Linux 条目中包含 CUDA 相关包。GitHub Actions 的 CPU wheel 安装路线独立于这份通用锁。

## 四、常用命令

```bash
python -m pip install -e '.[dev,media,plots]'
python -m pip wheel . --no-deps --no-build-isolation --wheel-dir outputs/package-review
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m pytest -q
ruff check omni tests scripts
ruff format --check omni tests scripts
python -m pip check
uv lock --check --offline
```

## 五、环境与配置

- 模型字段由 `OmniPlusConfig` 定义。`configs/model/thinker_tiny.json` 提供小配置，`configs/training/p4_*.yaml` 供数据准备和训练脚本读取，`configs/assets.yaml` 记录本地资产路径。
- `.env.example` 给出可选离线变量 `HF_HUB_OFFLINE=1` 和 `HF_DATASETS_OFFLINE=1`，由调用者在 shell 设置。代码入口通过显式本地路径加载资产。
- 已有历史验证覆盖 macOS CPU/MPS。CI 为 Ubuntu CPU。各阶段的设备、精度和实际结果以对应报告为准。

## 六、证据

- [依赖和构建声明](../../pyproject.toml)、[依赖锁](../../uv.lock)
- [CI 配置](../../.github/workflows/ci.yml)、[环境变量示例](../../.env.example)
- [模型配置](../../omni/models/config.py)、[资产路径表](../../configs/assets.yaml)
- [P5 历史环境与回归](../stages/P5.md)
