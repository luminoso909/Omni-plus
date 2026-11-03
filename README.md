## 一、Omni-plus

基于 MiniMind-O 架构思路的全模态模型自主复现与研究项目。保留直观的 PyTorch 核心，优先保证可读性、正确性和权重兼容性。

当前已有 Thinker、Projector、固定 Bridge、Talker、Dataset/Collator 和离散音频码本生成代码；已完成 P0 工程收尾与 P1 文本基线验收，后续阶段由用户另行启动。

### 1 当前范围

- MiniMind：复用本地 Dense/MoE 权重和配套 tokenizer，验收自研 Thinker。
- Qwen3-0.6B：收集官方未量化 HF 资产，使用官方 Transformers 实现验证文本推理；尚未接入自研 Thinker 或 Talker。
- 音频/视觉编码器、Mimi 波形链路、SFT、GRPO、Adaptive Bridge 和 Qwen3.5-0.8B 均按 [项目待办](docs/项目待办.md) 后续推进。

验收证据分别见 [P0 记录](docs/stages/P0.md) 和 [P1 记录](docs/stages/P1.md)。小模型单元测试与真实权重验收分开记录，不把随机权重测试当成真实模型效果。

### 2 环境与普通测试

使用 Python 3.12；当前目录为 `Omni++`，分发名为 `Omni-plus`，导入名为 `omni`。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m pytest -q
ruff check omni tests scripts
ruff format --check omni tests scripts
python -m pip check
```

已存在 `.venv` 时只需激活。普通 pytest 不下载或加载大型模型；MPS 基础测试在可用主机执行，不可用时明确跳过。CI 使用 Ubuntu 和 CPU-only PyTorch，检查命令与上面一致。未推送时不声称远程 CI 已运行。

### 3 两套文本资产与真实验收

路径说明见 `configs/assets.yaml`，所有相对路径从项目根目录解析。二进制权重、tokenizer 目录和缓存均不提交 Git。

```text
weights/minimind/           本地 llm_768.pth、llm_768_moe.pth
tokenizers/minimind/       MiniMind tokenizer.json、tokenizer_config.json
weights/qwen3-0.6b/         官方完整 HF 目录（包含权重、配置与 tokenizer）
experiments/p1/             可提交的来源、校验值和验收摘要
```

使用已有本地资产运行验收：

```bash
python scripts/check_minimind.py --variant all
python scripts/check_qwen3.py --model-dir weights/qwen3-0.6b --output experiments/p1/qwen3.json
```

脚本不训练模型或 tokenizer。MiniMind 直接用 `load_state_dict()` 严格加载；Qwen 使用 `AutoModelForCausalLM` 和 `AutoTokenizer`，验收时设置 `local_files_only=True`。资产缺失时明确报错，不在验收脚本里自动联网下载。来源 revision、文件 SHA-256、实际设备/精度与生成结果保存在 `experiments/p1/`。

### 4 项目目录与数据约定

```text
omni/models/config.py               共享模型配置
omni/models/thinker/model_thinker.py Thinker 与 CausalLM
omni/models/encoders/                模态特征投影
omni/models/talker/                  音频码本与 Talker
omni/models/model_omni.py            OmniPlusOmni 联合模型
omni/data/omni_dataset.py            Dataset 与 Collator
scripts/                            显式验收入口
configs/                            模型、资产及后续运行配置
experiments/                        小规模可复现摘要
```

主模型统一为 `OmniPlusOmni`。Dataset 的 `labels/audio_labels` 已 shift 一次，与 Omni logits 对应计算 CE 时不再 shift；原始媒体通过外部编码回调接入。正式训练和真实多模态闭环仍需分别验收。

### 5 致谢与许可

感谢 Jingyao Gong 的 [MiniMind-O](https://github.com/jingyaogong/minimind-o)，上游代码采用 [Apache-2.0](https://github.com/jingyaogong/minimind-o/blob/master/LICENSE)。本地参考仓库保持只读；代码独立整理，参考模型仅用于对照，不代表原作者背书。

Qwen3 资产来自 [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)，保留其官方许可证与出处。本项目自身许可证尚未指定；外部模型、tokenizer 和数据的许可分别记录，不将代码许可证直接套用于所有资产。
