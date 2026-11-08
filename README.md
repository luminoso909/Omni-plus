## 一、OmniPlus v2.1

OmniPlus 以官方 **Qwen3-1.7B 非 Base** 为 Thinker，复用冻结的 SenseVoice、SigLIP2 与 Mimi，训练 Projector、Fixed Bridge 和轻量 Talker。正式多模态 SFT 形成 B1 后，再从同一基线研究 Adaptive Bridge 与 Multimodal GRPO。 [项目主规划](docs/PROJECT_PLAN.md)

| 状态 | 内容 |
| --- | --- |
| ✅ 已验收（旧 MiniMind 或现有工程） | P0–P5、独立 Qwen3-0.6B 文本基线、R0 仓库审计、R1 Qwen3-1.7B 文本 Backend |
| ✅ 已验收（Qwen Omni 架构） | R2 真实资产离线 Smoke：音文对齐、梯度、Cache 与 PCM 分块通过；尚未训练新 Talker |
| ✅ 已验收（本地训练工程） | R3-L：四任务各 8 条诊断样本、版本化缓存、tiny 参数更新及新进程完整恢复 |
| 🧪 待实测 | 本地 R4-L–R6-L。云端 G3–G7 的资源测试、B1 正式训练、研究实验与 Benchmark |

本地按 `R2 → R3-L → R4-L → R5-L → R6-L` 逐项验收接口、数学正确性与极小样本闭环。之后按 `G3 → G4 → G5/G6 → G7` 记录云端资源、训练和实验结果。正式 B1 成立后再决定 Bridge 消融、GRPO Reward 与单卡或多卡方案。 [本地与云端手册](docs/OMNIPLUS_LOCAL_FIRST_CLOUD_PLAYBOOK.md) · [架构与数据决策](docs/OMNIPLUS_DECISIONS_AND_DATA.md) · [双轨待办](docs/项目待办.md)

### 1 历史验收成果（Historical validated work）

- MiniMind：复用本地 Dense/MoE 权重和配套 tokenizer，验收自研 Thinker。
- Qwen3-0.6B：收集官方未量化 HF 资产，使用官方 Transformers 实现验证文本推理；尚未接入自研 Thinker 或 Talker。
- SenseVoice/SigLIP2、Mimi：复用本地预训练模型，已验收混合输入与现成 SFT 权重的 10 秒语音输出。
- P4：官方 64/8 条训练/验证样本、小主干训练、损失/梯度和恢复验收通过；不代表完成全尺寸正式训练。
- P5 将逐帧音频码本解码为连续 PCM，并测量 MiniMind 的 Cache 开关、首段音频延迟与分块间隔。 [P5 报告](docs/stages/P5.md)
- R0 审计记录历史指纹与真实回归，并列出 R1 的待实施文件。 [R0 报告](docs/stages/R0.md) · [项目待办](docs/项目待办.md)

验收证据见 [P0](docs/stages/P0.md)、[P1](docs/stages/P1.md) 及 [P2–P4 汇总](docs/reports/P2-P4.md)。汇总报告第二章逐项列出相比原项目的补充与修改，详细实测与限制分别记录。

### 2 环境与普通测试

使用 Python 3.12；当前目录为 `Omni++`，分发名为 `Omni-plus`，导入名为 `omni`。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,media,plots]"
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m pytest -q
ruff check omni tests scripts
ruff format --check omni tests scripts
python -m pip check
```

已存在 `.venv` 时只需激活。普通 pytest 不下载或加载大型模型；MPS 基础测试在可用主机执行，不可用时明确跳过。CI 使用 Ubuntu 和 CPU-only PyTorch，检查命令与上面一致。未推送时不声称远程 CI 已运行。

`pip install -e '.[dev]'` 仍支持最小文本环境；运行全部测试加装 `media,plots`。真实编码器验收另需 `encoders`，官方小样本收集另需 `data`。FunASR 带有较多间接依赖，单独放在可选组，不强加给基础文本环境。

### 3 文本资产与真实验收

历史路径见 `configs/assets.yaml`，R1 固定版本见 `configs/assets_v2.yaml`。所有相对路径从项目根目录解析。二进制权重、tokenizer 目录和缓存均不提交 Git。

```text
weights/minimind/           本地 llm_768.pth、llm_768_moe.pth
tokenizers/minimind/       MiniMind tokenizer.json、tokenizer_config.json
weights/qwen3-0.6b/         官方完整 HF 目录（包含权重、配置与 tokenizer）
weights/qwen3-1.7b/         官方非 Base 权重、配置与 tokenizer
experiments/p1/             可提交的来源、校验值和验收摘要
experiments/r1/             1.7B Backend 离线验收摘要
```

使用已有本地资产运行验收：

```bash
mkdir -p outputs
run_dir=$(mktemp -d "$PWD/outputs/recheck-p1.XXXXXX")
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python scripts/check_minimind.py --variant all --output-dir "$run_dir/minimind"
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python scripts/check_qwen3.py --model-dir weights/qwen3-0.6b --output "$run_dir/qwen3.json"
```

脚本不训练模型或 tokenizer。MiniMind 直接用 `load_state_dict()` 严格加载；Qwen 使用 `AutoModelForCausalLM` 和 `AutoTokenizer`，验收时设置 `local_files_only=True`。资产缺失时明确报错，不在验收脚本里自动联网下载。来源 revision、文件 SHA-256、实际设备/精度与生成结果保存在 `experiments/p1/`。

R1 使用官方 `Qwen3ForCausalLM` 验证独立 Backend。已有固定版本资产时，用一个新的结果目录运行。

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python scripts/check_qwen_backend.py \
  --device cpu --output-dir experiments/r1/recheck-01
```

1.7B 已通过 CPU FP32 原生对照、左 padding、KV prefill/decode、Cache 重置及中英文贪心烟测。下载命令、选层契约和实测误差见阶段报告。 [R1 报告](docs/stages/R1.md)

### 4 多模态验收与小样本训练

```bash
python -m pip install -e '.[dev,media,encoders,plots,data]'
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python scripts/check_p2.py --device cpu
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python scripts/check_p3.py --output-dir outputs/p3/review-01 --report experiments/p3/review-01.json
```

真实编码器、Mimi 与 SFT 权重只读引用 `../minimind-o`。上述验收使用新报告路径，重复执行不得覆盖旧结果。小样本准备/缓存、训练与恢复命令见 [P4 报告](docs/stages/P4.md)，各操作显式启动。完整数据集尚未下载。

R2 复用 R1 的 Qwen3-1.7B 和本地 P4 小样本。已有离线资产时，替换 `--reference-root` 为旧模型仓库实际位置，每次指定新的结果目录：

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 python scripts/check_r2.py \
  --reference-root ../minimind-o --output-dir outputs/r2/recheck-01
```

R2 已核对三条真实 T2A 时间轴、冻结模块与输入梯度、混排、双路 Cache 及 PCM 尾块；原生 1.7B Omni 尚未训练。 [R2 报告](docs/stages/R2.md) · [真实 Smoke 摘要](experiments/r2/20261108-cpu-fp32/manifest.json)

R3-L 的统一入口已验证，配置中的资产路径须对应本机实际位置：

```bash
python scripts/run_experiment.py --help
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python scripts/run_experiment.py inspect --config configs/v2/r3_local.yaml
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python scripts/run_experiment.py prepare --config configs/v2/r3_local.yaml
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python scripts/run_experiment.py smoke --config configs/v2/r3_local.yaml --dry-run
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python scripts/run_experiment.py smoke --config configs/v2/r3_local.yaml --steps 2 --run-dir outputs/r3-local/recheck-part
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python scripts/run_experiment.py smoke --config configs/v2/r3_local.yaml --steps 4 --resume outputs/r3-local/recheck-part/step-000002.pt --run-dir outputs/r3-local/recheck-resumed
```

每次 smoke 使用新目录。`prepare` 只在完整版本契约相同时复用缓存；未知版本的旧回答声码被隔离，真实语音经锁定 Mimi 重编码。当前 A2A 为复述诊断，正式问答数据仍需目标音频溯源。`inspect` 使用 meta 参数统计；tiny 更新与原生 1.7B CUDA 训练分别验收。`train/resume/evaluate/profile` 独立子命令留待后续实现。 [R3-L 报告](docs/stages/R3-local.md) · [G3 云端说明](docs/G3_CLOUD_RUNBOOK.md)

### 5 流式推理

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python scripts/demo_p5.py '你好，请用一句话介绍春天。' --device cpu --chunk-frames 4
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python scripts/check_p5.py --device cpu --chunk-frames 4
```

Demo 与基准使用已有的本地 MiniMind SFT 和 Mimi。波形留在 `outputs/p5/`，基准摘要按 run 保存到 `experiments/p5/`。运行参数、计时口径及适配接口见阶段报告。 [P5 报告](docs/stages/P5.md)

### 6 项目目录与数据约定

```text
omni/models/config.py               共享模型配置
omni/models/thinker/model_thinker.py Thinker 与 CausalLM
omni/models/thinker/qwen_backend.py   官方 Qwen Thinker Adapter
omni/models/encoders/                冻结编码器与模态投影
omni/models/talker/                  音频码本与 Talker
omni/models/model_omni.py            OmniPlusOmni 联合模型
omni/models/model_omni_v2.py         Qwen 路径的 OmniPlusOmni 联合模型
omni/data/omni_dataset.py            Dataset 与 Collator
omni/data/qwen_omni.py                Qwen 原生模板与联合监督
omni/data/prepare.py                 独立数据划分与冻结特征缓存
omni/inference/codec.py              Mimi 与波形 IO
omni/inference/streaming.py          每话轮的 Mimi 流式状态与 PCM 事件
omni/training/                      loss、训练步骤和完整状态恢复
scripts/                            显式验收入口
configs/                            模型、资产及后续运行配置
experiments/                        小规模可复现摘要
```

MiniMind 与 Qwen 的 `OmniPlusOmni` 分属两个模型入口。Dataset 的 `labels/audio_labels` 已 shift 一次，外部 CE 按同位置 logits 对齐。原始媒体通过外部编码接入；Qwen 使用独立的 tokenizer、Cache 和联合标签构造。 [文件级迁移表](docs/reports/R0-migration-map.md)

### 7 致谢与许可

感谢 Jingyao Gong 的 [MiniMind-O](https://github.com/jingyaogong/minimind-o)，上游代码采用 [Apache-2.0](https://github.com/jingyaogong/minimind-o/blob/master/LICENSE)。本地参考仓库保持只读；代码独立整理，参考模型仅用于对照，不代表原作者背书。

Qwen3 资产保留官方许可证与出处。外部模型、tokenizer 和数据的许可分别记录。本项目自身许可证尚未指定。 [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) · [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)

本地 Mimi 转换资产采用 CC-BY-4.0。官方数据卡声明 Apache-2.0 与 GPL-3.0，未逐文件区分；本轮子集仅本地使用，不公开再分发。
