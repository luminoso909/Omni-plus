## 一、目录地图

| 路径 | 职责 | 代表文件 |
| --- | --- | --- |
| `omni/` | 模型、数据、训练和推理 Python 包 | `omni/models/model_omni.py` |
| `configs/` | 模型参数、历史资产位置、P4 数据与训练参数 | `configs/assets.yaml`、`configs/training/p4_overfit.yaml` |
| `scripts/` | 资产验收、数据准备、训练及演示入口 | `scripts/check_minimind.py`、`scripts/train_sft.py`、`scripts/demo_p5.py` |
| `tests/` | 本地 fixture、回归与小规模集成测试 | `tests/test_thinker.py`、`tests/test_streaming.py` |
| `docs/` | 规划、阶段报告、专题文档及本次代码库审计 | `docs/stages/P5.md`、`docs/reports/P2-P4.md` |
| `experiments/` | 结构化摘要、数据来源与损失曲线 | `experiments/p4/data.json`、`experiments/p5/cpu-fp32-20261105.json` |
| `weights/`、`tokenizers/` | 本地模型权重、词表和模板，Git 跟踪对应占位文件 | `.gitignore`、`configs/assets.yaml` |
| `cache/`、`.cache/`、`outputs/` | 本地子集、冻结特征、checkpoint、波形与临时复核输出 | `.gitignore`、`configs/training/p4_data.yaml` |
| `.github/` | CPU CI | `.github/workflows/ci.yml` |

## 二、运行入口

- Python 接口从 `omni.models` 导入 `OmniPlusForCausalLM` 或 `OmniPlusOmni`。`omni.inference` 导出波形 IO、Mimi 和流式事件接口。
- 使用 `python scripts/demo_p5.py ...` 运行 MiniMind 文本/波形演示。`check_minimind.py`、`check_qwen3.py`、`check_p2.py`、`check_p3.py` 与 `check_p5.py` 提供各自的验收入口。
- `collect_p4.py` 获取有限官方样本，`prepare_p4.py` 划分并可选缓存特征，`train_sft.py --config ...` 启动小样本训练。`check_p4.py` 的训练烟测会执行优化器更新。
- 脚本采用 `argparse`，通常通过脚本文件位置解析项目根目录。`pyproject.toml` 定义包发现和依赖，CLI 直接通过脚本文件启动。

## 三、模块边界

| 模块 | 当前职责 | 调用边界 |
| --- | --- | --- |
| `omni/models/config.py` | `OmniPlusConfig` 保存 Thinker、Talker、模态与 Bridge 配置 | 构造期校验尺寸和配置项 |
| `omni/models/thinker/model_thinker.py` | MiniMind 文本主干、HF 语言模型、嵌入前向与 Cache | `forward_embeddings()` 提供可选 Bridge 层输出及位置 |
| `omni/models/model_omni.py` | 特征注入、固定取层、联合前向、文本/音频码本生成 | 接收按样本顺序排列的已编码特征，返回联合 logits 或生成事件 |
| `omni/models/encoders/`、`talker/` | 冻结预训练编码器、投影、Talker 主干与八码本 Head/Embedding | 编码器由调用方从本地资产创建。Talker 接收 Bridge 与历史码本 |
| `omni/data/` | 样本读取、特征缓存、占位符、标签、padding、码本延迟变换 | Dataset 接收外部媒体回调，输出已对齐的训练张量 |
| `omni/training/` | 损失、训练步骤、冻结模式、评估、checkpoint | CLI 负责配置、采样进度与实验记录 |
| `omni/inference/` | `MimiCodec`、波形 IO、`MimiStreamingDecoder`、逐块 PCM | 消费模型生成的码本事件，每次流式调用拥有独立状态 |
| `omni/evaluation/` | 当前包含包占位 | 历史评测实际由 `scripts/check_*.py` 执行 |

## 四、命名与组织

- 源码按职责划分，文件和模块使用 `snake_case`，如 `model_thinker.py`、`audio_alignment.py`。测试集中在 `tests/test_*.py`。
- 包内相邻模型组件可用相对导入，跨包和脚本使用 `omni.*`。模型、数据和推理包通过 `__init__.py` 显式暴露公共入口。
- R0 读取工作区源码与阶段记录。现有 P5 文件和未跟踪的 `test.ipynb` 在开始审计时已存在，Git 范围见 R0 阶段报告。

## 五、证据

- [包发现与测试路径](../../pyproject.toml)、[公开模型入口](../../omni/models/__init__.py)、[推理入口](../../omni/inference/__init__.py)
- [联合模型](../../omni/models/model_omni.py)、[数据准备](../../omni/data/prepare.py)、[训练入口](../../scripts/train_sft.py)
- [历史资产路径](../../configs/assets.yaml)、[本地输出忽略规则](../../.gitignore)
