# R3-L｜本地数据工程与训练准备

2026-11-08 验收通过。本阶段完成数据版本、有限样本缓存、参数冻结、统一入口与真实 tiny 更新/跨进程恢复。**正式训练、1.7B CUDA 训练与资源 Profile、B1、LoRA、Bridge 消融和 GRPO 均未执行。** 本地真实样本训练使用随机初始化的小型兼容 Qwen/Talker；没有用本地结果代替全尺寸模型训练验收。R4-L 等待授权。

## 一、前置审查与历史保护

起点 `bd65261`。完整阅读规范、README、主规划、v2.1 两份手册、待办和 R2 报告，并审查 Dataset/Collator、SFT/Loss/Checkpoint、配置和脚本。R2 manifest 的 SHA-256 为 `05d2b568cd90ebad56a3e1e7fbd613f83bc38ec885f594fc40bc89b76c0416ff`，与 R2 报告一致；核心源码、配置和报告与该提交一致。前置 R2/Backend 18 项测试通过，独立 R2/SFT 检查 19 项通过，没有阻断 R3-L 的 R2 问题。

开工时已有 P5 模型/推理/测试改动、未跟踪 P5 报告与实验、个人 Notebook、旧规划更名。143 个既有文件的 SHA 或不存在状态及初始补丁保存在 `outputs/r3-local/baseline-20261108/`。本阶段没有改模型核心、原 SFT/Loss、P0–P5/R0–R2 报告、旧实验、权重或原仓库。阶段提交只包含本轮白名单文件。

## 二、文件级实现与复用

| 文件 | 用途与依据 |
| --- | --- |
| [r3_prepare.py](../../omni/data/r3_prepare.py) | 复用 `content_hash`、R2 `QwenOmniFormatter`、Encoder、Mimi 与 delay/restore。新增媒体连通分组、保留历史分配的三路划分、原始语音重编码、版本契约和原子缓存发布。 |
| [r3_local.py](../../omni/training/r3_local.py) | 配置校验、tiny/原生模型构造、互不重复的参数分组、AdamW/线性 Scheduler、有限步数运行。直接调用 P4 `train_step/save_checkpoint/load_checkpoint`。 |
| [run_experiment.py](../../scripts/run_experiment.py) | 薄 CLI：`inspect/prepare/smoke`、`--dry-run`、`smoke --resume`。未知命令、配置拼写错误、缺资产、超预算、未实现 LoRA 明确失败。 |
| [r3_local.yaml](../../configs/v2/r3_local.yaml) | CPU FP32、seed 42、四任务各 8 条、默认 T2A 4 步、batch 1、累积 2。 |
| [g3_h100.yaml](../../configs/v2/g3_h100.yaml) | 独立候选：原生 Qwen3-1.7B、4×768 Talker、CUDA BF16、10 步上限；仅本地 meta/dry-run 通过。 |
| [数据测试](../../tests/test_r3_data.py)、[训练测试](../../tests/test_r3_training.py) | 14 项测试覆盖传递媒体组、历史划分冲突、预算、缓存损坏/版本错配、冻结、错误配置及新进程完整状态恢复。 |
| [G3 操作说明](../G3_CLOUD_RUNBOOK.md) | 代码固定、环境安装、资产搬运/校验、准备与短运行入口、恢复和结果回传；标明尚未执行的 CUDA 步骤。 |
| README、AGENTS、主规划、v2.1 两文档、待办 | 同步本地已验收状态；原 R3 云端验收条目和所有正式训练任务保持未完成。 |

## 三、真实数据来源、划分与兼容性

[数据 Manifest](../../experiments/r3-local/20261108-cpu-fp32/data-manifest.json)记录全部源文件、资产、处理器、样本行号和媒体摘要。数据来源为已下载的 `jingyaogong/minimind-o_dataset`，revision `d6588e12ac2ac8ced65eb58a7d7b3eef4aa220de`；源 Parquet 文件 SHA 与 P4 清单逐项一致。没有新下载。数据卡声明 Apache-2.0 与 GPL-3.0，未逐文件区分；保留原许可说明，原始数据仅本地诊断使用，不公开再分发。

**旧回答声码缺少逐样本编码器版本证明，因此全部隔离。** 范围合法、8 路声码或可解码不足以证明编码版本相同。本轮选择 8 段已有真实用户语音和对应转写，分别进行 16 kHz SenseVoice 与 24 kHz Mimi 处理；监督声码由锁定的本地 Mimi 重新生成。每段语音形成三个显式诊断视图，加上 8 张原始 I2T 图像：

| 任务 | 实际含义 | train / validation / test | 目标 |
| --- | --- | --- | --- |
| T2A | 根据真实转写朗读 | 6 / 1 / 1 | 转写文本 + 新编码声码 |
| I2T | 原始图像问答 | 6 / 1 / 1 | 原始文本回答 |
| A2T | 对真实用户语音转写 | 6 / 1 / 1 | 对应转写文本 |
| A2A | 明确要求复述输入语音并转写 | 6 / 1 / 1 | 转写文本 + 新编码声码 |

合计 **32 条任务视图、16 份独立媒体**；train/validation/test 为 **24/4/4** 条，对应 **12/2/2** 个媒体组，交叉为 0。同一语音的三种视图共享分配。媒体内容摘要不按输入/输出角色分开命名；多媒体关联形成传递分组。保留 P4 已用媒体的 train/validation 归属，test 只选此前未用的媒体组，seed 42 决定组内抽样。该 test 是本地工程留出，不是正式 Benchmark。

采用官方 Qwen Tokenizer、非 thinking Chat Template，重建原始文本的 token IDs。复用 R2 的占位注入、Teacher Forcing、8 路 Delay、STOP/PAD；`labels/audio_labels` 只 shift 一次，原 P4 CE 与 STOP 权重 10 不变。真实混合 batch 为文本 `[4,149]`、声码 `[4,8,149]`；各任务的监督范围与补齐位置通过校验。所有 split 的最大序列长度分别为 238/251/276。

缓存位于 `.cache/r3-local/tiny-v1/`，每份 Tensor 文件均有 SHA-256。契约包含 Qwen 固定 revision、三个本地模型内容摘要和 local-sha revision、模型/processor 配置、包版本、相关源码、原数据 SHA、种子、预算、16k/24k 采样率和声码策略。加载必须传入预期契约；契约或文件校验值不同即拒绝复用，使用新目录重新准备。发布使用临时目录完成后原子改名。

真实 MiniMind tokenizer 被 Qwen formatter 拒绝；声学 ID 2048 被拒绝。详见 [数据检查](../../experiments/r3-local/20261108-cpu-fp32/data-checks.json)。

## 四、参数分组与运行边界

`inspect` 使用 meta 模型统计真实结构的参数，不加载训练权重、不进行 forward。下表每个参数只计入一组；媒体 Encoder/Mimi 位于离线准备侧并保持冻结。Qwen 使用 R1 的 revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`。

| 分组 | Tiny 参数 | 原生 1.7B 配置参数 | R4-A |
| --- | ---: | ---: | --- |
| Qwen Thinker | 4,880,576 | 1,720,574,976 | 冻结 |
| Vision Projector | 27,200 | 5,772,800 | 冻结 |
| Audio Projector | 18,496 | 5,248,000 | 冻结 |
| Fixed Bridge | 1,056 | 1,573,632 | 更新 |
| Talker（不含下列三组） | 39,362 | 30,680,834 | 更新 |
| Audio Embedding | 204,800 | 7,520,256 | 更新 |
| Audio Heads | 204,800 | 7,520,256 | 更新 |
| Speaker Projection | 6,144 | 147,456 | 冻结 |
| **可训练合计** | **450,018** | **47,294,978** | |

Tiny 为 2 层、32 维 Qwen，Talker 保留 4 层但缩至 32 维；仍使用真实 Qwen 151936 词表容量、512/768 输入特征、8×2048 声码和 2112 输出词表。模型随机初始化，不代表使用预训练 1.7B 做过训练。独立 `projectors` 配方仅更新两个 Projector，真实 I2T/A2T/A2A 各验证两次更新，梯度通过冻结 Qwen。

AdamW LR 0.0003、weight decay 0.01、grad clip 1、线性日程、warmup 1。配置限制最多 10 步，CUDA 不可用时实际运行报错；本地运行使用 CPU FP32。LoRA enabled=true 明确拒绝，留待 R4-L。

## 五、实际更新与跨进程恢复

环境：macOS arm64、Python 3.12.14、PyTorch 2.15.0、Transformers 4.57.6、CPU FP32、4 线程。结果来自 [验收摘要](../../experiments/r3-local/20261108-cpu-fp32/acceptance.json)，不是模型质量结论。

| 运行 | 实际更新 | 最后一步总 Loss | 非零梯度证据 |
| --- | --- | ---: | --- |
| T2A full | 0→4 | 20.867629 | Bridge/Talker/Embedding/Heads |
| T2A part | 0→2 | 21.015567 | 同上 |
| T2A resumed（新进程） | 2→4 | 20.867629 | 同上 |
| I2T Projector | 0→2 | 11.946516 | Vision Projector 最大绝对梯度 0.0294005 |
| A2T Projector | 0→2 | 12.007341 | Audio Projector 最大绝对梯度 0.0277566 |
| A2A 复述 Projector | 0→2 | 21.014114 | Audio Projector 最大绝对梯度 0.108065 |

所有运行的冻结参数前后摘要相同，可训练参数摘要发生变化，Loss 与梯度有限。各步 batch 不同，表中 Loss 不用于证明收敛或泛化。

终点 checkpoint 保存 model、optimizer、scaler、全部 RNG、配置、冻结名单和 progress；progress 包含 Scheduler、step、epoch、cursor、数据/源码/运行环境身份。真实连续 4 步与 **独立进程 2+2 步**的整个 checkpoint 嵌套对象逐元素完全一致，包含参数、AdamW 动量、Scheduler、RNG 和游标；单测还使用非零 dropout 跨 epoch 检查。仅在目标步数完成时保存，避免每步重复保存全尺寸冻结 Qwen。读取旧 checkpoint、写新 run，绝不覆盖；恢复时保守拒绝配置路径、源码或运行版本变化，尚不支持无校验的跨设备/跨配置迁移。

## 六、实际命令与回归结果

从项目根目录执行；已存在的 run 会报错，复验时换新目录。完整实际参数记录在 acceptance.json 的 commands 字段，运行日志保存在 `outputs/r3-local/`。

```bash
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
.venv/bin/python scripts/run_experiment.py --help
.venv/bin/python scripts/run_experiment.py inspect --config configs/v2/r3_local.yaml
.venv/bin/python scripts/run_experiment.py prepare --config configs/v2/r3_local.yaml
.venv/bin/python scripts/run_experiment.py smoke --config configs/v2/r3_local.yaml --dry-run
.venv/bin/python scripts/run_experiment.py smoke --config configs/v2/r3_local.yaml --run-dir outputs/r3-local/full --steps 4
.venv/bin/python scripts/run_experiment.py smoke --config configs/v2/r3_local.yaml --run-dir outputs/r3-local/part --steps 2
.venv/bin/python scripts/run_experiment.py smoke --config configs/v2/r3_local.yaml --run-dir outputs/r3-local/resumed --resume outputs/r3-local/part/step-000002.pt --steps 4
```

Projector 验收复制本地配置，仅将 `training.recipe` 改为 `projectors`、`training.tasks` 分别改为 `[i2t]`、`[a2t]`、`[a2a]`，保存为 `outputs/r3-local/{task}.yaml`，分别用 `smoke --steps 2` 运行。完整生效配置随每个 run manifest 保存。G3 本地 dry-run 使用 `outputs/r3-local/g3-local-paths.yaml`，只把候选配置的 reference_root 改为本机外置盘路径；没有调用 CUDA forward。

```bash
.venv/bin/ruff check omni tests scripts
.venv/bin/ruff format --check omni tests scripts
.venv/bin/python -m pip check
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MPLCONFIGDIR=/private/tmp/omni-r3-mpl .venv/bin/python -m pytest -q
```

| 检查 | 实际结果 |
| --- | --- |
| 当前工作区完整 pytest | **183 passed，19.48 秒**，包含 MPS 基础测试和原 P0–P5/R2 回归 |
| Ruff / 格式 / 依赖 | 全部通过，59 个 Python 文件格式通过，pip check 无冲突 |
| 独立干净源码 | `bd65261` 归档仅叠加新 R3 文件，排除旧 P5 未提交改动；R3/Backend/R2/SFT **41 passed，9.44 秒** |
| prepare 新缓存 / 再次复用 | 均通过，后者 `cache_hit=true`，未重编码 |
| 旧缓存拒绝 | prepare-02 遇到修改前源码生成的缓存，明确报版本不匹配；保留旧目录，prepare-03 在新目录成功 |
| 负例 | 错误 tokenizer、声码越界、处理器版本变化、文件损坏、超预算、错拼字段、缺权重、LoRA 未实现、冻结策略改变均被拒绝 |
| 远程 CI / CUDA | 未执行 |

## 七、R4-L 入口与未解决依赖

1. 以 `scripts/run_experiment.py`、`omni/training/r3_local.py` 和当前缓存为入口，扩展正式可恢复的 train/resume/evaluate 接口、任务配方与验证调度；继续复用 P4 数学定义。
2. 在极小样本上完善 R4-A/C/D：T2A 训练诊断、输入模态条件有效性、混合采样与按任务度量，补齐可开关 LoRA 和 adapter 严格保存恢复。当前只有有限步数 smoke，不宣称完整联合 SFT 已实现。
3. 原始问答式 A2A 需要来源可信的目标波形或精确 Mimi 编码证明；当前复述诊断只用于工程。扩大数据前另审许可、跨来源媒体去重及正式留出集。
4. G3 操作说明和候选配置已就绪。后续 1×H100 80GB 的 1–10 步、恢复、CUDA 显存/吞吐与可选 100 步 Profile 仍待独立授权和实测，预算不能由本地 tiny 运行外推。

当前停止于 R3-L。G3/G4/G5/G6/G7、正式 B1 与质量评测的状态保持未执行。

## 八、历史报告指纹

以下 SHA-256 与开工基线相同，六份 P 报告也与 R0 记录一致。

| 报告 | SHA-256 |
| --- | --- |
| P0 | `01ac1cb1d4bad5c2d938fc5ad60f06d68ec49e11060e01442f38e0d31fcd993f` |
| P1 | `ecb9052d9162d77d632210c2d5bf18c7ff5b52fcda5fe8a70a6be6df51757473` |
| P2 | `b1345c087866ad07026c33f0506d82d5a633e8c3ffb33e1eeb8dbe393cbe04b3` |
| P3 | `14ed2a69dd9340f4373ce35b02f750d7fd372cad5dfeedb9777de46aebcf811a` |
| P4 | `37b80cf84f4c512ab916cc43a03ae2e761e98a8fa6aabdd3c865ebb01df9040f` |
| P5 | `302dd9b5ba88afc4ed21079af94777b2960052c2b9aa3ce6992c6dc4196537ed` |
| R0 | `5b75b59d6add730b323534ed4c4d0a098091fe28b0bf9375e9694a2522538bcf` |
| R1 | `66460276b21b5ee7cea4e2ab2b525ae3212e7cdaffc201dea821096aac449982` |
| R2 | `2a9332dcafaa5ef6de123b43207b4be69ff4b5c077e27c97da5949e390437cf9` |
