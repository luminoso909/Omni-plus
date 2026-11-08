## 一、结论与本次边界

2026-11-08，基于本地提交 `7c6f292` 完成 **R4-L：Local Multimodal SFT Implementation**。A/B/C/D 的训练代码、有限数据入口、数学与 Tiny 更新/恢复已验收；全仓库 **222 passed**。本阶段未 Push、未连接云 GPU、未下载完整数据、未执行正式 SFT、Adaptive Bridge 或 GRPO。**G3/G4、音质验收与 B1 均未完成。**

| 子阶段 | 本地代码与实测 | 仍待 G4 完成 |
| --- | --- | --- |
| R4-A | 原训练器 T2A，冻结 Qwen；真实 6 条训练样本、8 次 Tiny 更新，同集合总 Loss 下降 4.52% | 原生 1.7B + 4×768 Talker 的充分过拟合与可理解语音 |
| R4-B | 长度分桶、验证/早停/最佳状态、每码本 CE/STOP、预测 Mimi WAV、本地 ASR；兼容初始化接口 | 长短句/双语正式训练、独立验证、音质、人评与自然结束 |
| R4-C | Vision/Audio Projector 分别更新；冻结 Qwen 保留输入梯度；真实图像与 fixture 音频调换 | 真实语音同形状对照与独立质量评测、两路正式对齐 |
| R4-D | Qwen LoRA、多任务采样/Loss/分组 LR、阶段组件合并、独立 Adapter 和完整恢复 | 可信 QA A2A、正式联合 SFT、冻结 B1 与首轮 Benchmark |

真实预训练 Qwen3-1.7B 本轮只做短前向 LoRA 兼容验证；全部训练更新使用随机初始化 Tiny Qwen/Talker。不能把 Tiny 更新写成 1.7B 的训练成果。当前真实 A2A 仍是复述诊断，不是问答式语音能力。

证据：[验收与实际命令](../../experiments/r4-local/20261108-cpu-fp32/acceptance.json)、[参数统计](../../experiments/r4-local/20261108-cpu-fp32/parameters.json)、[数据清单](../../experiments/r4-local/20261108-cpu-fp32/data-manifest.json)、[真实 Qwen LoRA](../../experiments/r4-local/20261108-cpu-fp32/qwen-lora.json)、[回归与历史保护](../../experiments/r4-local/20261108-cpu-fp32/verification.json)。

## 二、前置审计与文件级修改

已阅读用户附件、当前 AGENTS/README/主规划/v2.1 手册/数据决策/待办、R3-local、G3 Runbook 与现有数据、模型、SFT、Loss、Checkpoint。R3-L 起点提交与缓存清单匹配；既有 R2/R3 数据语义、版本契约和测试可以继续使用，没有需要重写模型核心的阻断。

开工前保存 157 个既有文件的 SHA 或不存在状态、Git 状态及补丁到 `outputs/r4-local/baseline-20261108/`。排除本阶段授权的 9 个既有编辑文件后，其余 **148 项逐一保持一致**，包括所有历史报告、既有实验和用户原先的脏文件。未修改旧 `minimind-o`、权重、历史 checkpoint、R3 缓存或 P0–P5/R0–R3-local 报告。

| 文件 | 本次变化与复用关系 |
| --- | --- |
| `omni/training/r3_local.py` | 在原配置、模型构造和运行循环上增加 R4 schema 2、继承配置、分组 LR、任务游标、初始化、验证/早停、独立评估和 Profile；schema 1 仍通过原 R3 测试 |
| `omni/training/sft.py` | 原 `train_step` 增加可选任务名称/权重，默认路径不变；原 CE、STOP 权重、梯度累积及 checkpoint 格式沿用 |
| `scripts/run_experiment.py` | 扩展原薄 CLI，提供 inspect/prepare/smoke/train/resume/evaluate/demo/profile 与 dry-run |
| `omni/training/adapters.py` | 简单 Linear LoRA、零 B 初始化、严格独立 Adapter 导出/恢复；没有新依赖或动态注册系统 |
| `omni/evaluation/sft.py` | 按有效监督统计各码本 CE、STOP；回答位置模态调换；仅用 prompt 生成预测 WAV，真实 ASR/CER/WER |
| `omni/data/sft_sources.py` | 经审计的规范化 JSONL 入口，复用 R3 分组/预算/缓存和 R2 Formatter、Encoder、Mimi；不复制数据集或训练框架 |
| `configs/v2/{r4a,r4b,r4c_vision,r4c_audio,r4d}_{local,h100}.yaml` | 10 份可解析配置；本地 Tiny 与原生 CUDA 10-step Pilot 分开，云任务未执行 |
| `tests/test_r4_{training,adapters,evaluation,sources}.py` | 新增 39 项测试，覆盖数学梯度、采样与恢复、来源拒绝、LoRA、评测和失败路径 |
| `docs/cloud/G4_SFT_RUNBOOK.md`、本报告 | 实际入口、数据规范、初始化/恢复、正式训练闸门和已知限制 |
| README、AGENTS、PROJECT_PLAN、待办、两份 v2.1 手册 | 仅同步本地已验收事实；原正式训练与研究要求保留未完成 |
| `experiments/r4-local/20261108-cpu-fp32/*.json` | 5 份小规模证据；原始文本、波形、缓存、Adapter 和 checkpoint 留在忽略目录 |

模型核心文件、原 Dataset/Collator、Delay、Codec 与 Streaming 均未改写。没有引入第二套训练框架。初始未提交的 P5 源码、测试、报告、规划更名及个人 Notebook 不纳入 R4 提交。

## 三、训练数学与参数边界

### 1 数据与 Loss

沿用 Qwen 原生 tokenizer/chat template、R2 文本/音频时序、8 路 Delay、STOP/PAD、Teacher Forcing。外部标签已经 shift 一次，CE 按同位置 logits 计算。STOP 权重仍是 10；每任务损失仍是文本 CE 加 8 个码本各自有效 token 均值的平均值，模型原辅助损失不改。

新增联合权重只作用于外层任务目标。设一次梯度累积组出现任务集合为 $S$，配置全部任务为 $T$，每任务权重为 $\lambda_t$，则：
$$L_{\mathrm{step}}=\sum_{t\in S}\frac{\lambda_t}{\sum_{u\in T}\lambda_u}L_t.$$
同一任务多个 micro-batch 共用该任务整组有效 token 分母；缺席任务贡献为 0。分母不随本步任务子集变化，避免单任务累积时把权重抵消。测试直接比较显式目标和各参数梯度；单任务窗口中系数从 $1/2$ 改为 $3/4$ 时，Loss/梯度均变为 1.5 倍。

`sample_weights` 决定平滑加权轮转，`task_weights` 决定上述目标，`task_batch_sizes` 决定各任务 batch；不把数据源抽样比例混作 Loss 权重。任务内 Seed/epoch 洗牌后按长度分桶，游标全部保存。配置的任务预算/权重/LR 映射整体替换，防止继承时残留其它任务；其余字典按层合并。

### 2 冻结、LoRA 与模块统计

| 配方 | Tiny 可训练参数 | 原生 1.7B 配置可训练参数（meta） | 冻结边界 |
| --- | ---: | ---: | --- |
| A/B | 450,018 | 47,294,978 | Qwen、两个 Projector、Speaker；更新 Bridge/Talker/Embedding/Heads |
| C Vision | 27,200 | 5,772,800 | 仅 Vision Projector 可训练 |
| C Audio | 18,496 | 5,248,000 | 仅 Audio Projector 可训练 |
| D | 496,610 | 59,118,594 | Qwen 主权重/Speaker 冻结；更新两路 Projector、A/B 各组和 LoRA |

原生 Qwen 主权重 1,720,574,976 个参数保持冻结。LoRA 默认 rank=4、alpha=8、dropout=0，原生 28 层各注入 q_proj/v_proj，共 56 个 Linear、802,816 个新增参数；Tiny 为 896。模块路径来自真实 Qwen 结构。零 B 初始化在真实官方权重的 19-token CPU FP32 前向中得到 **logits 最大误差 0**，未运行 1.7B 反向或 optimizer。

Tiny 为 2×32 Qwen、4×32 Talker，但保留真实 Qwen 151936 词表容量、视觉 `[64,768]`、语音 `[L,512]`、8×2048 Mimi codes、2112 输出容量。正式配置仍是 2048 Thinker、Fixed Bridge 2048→768、4×768 Talker，没有修改其结构。

冻结通过 `requires_grad` 完成；Projector 的梯度穿过 Qwen，未用整段 `no_grad()` 切断。LoRA dropout 限定 0，避免 Qwen 的既有 eval 生命周期把非零 dropout 静默关闭。`gradient_checkpointing: true` 明确报未验收错误；本轮不擅自改 Qwen/Talker 的训练生命周期。

### 3 初始化与恢复

默认 Talker 随机初始化。可显式指定完全兼容的 Talker 状态；键、shape、dtype 和非有限值不匹配会拒绝，不自动转换旧 MiniMind 权重。`components_checkpoint` 先加载 B 的全部多模态组件，`vision_checkpoint/audio_checkpoint` 再分别覆盖 C 的对应 Projector，互不覆盖；真实 B+C 检查点组合后又完成 1 次联合更新。

完整恢复复用原 `save_checkpoint/load_checkpoint`，包括 model/LoRA、AdamW、Scheduler、Scaler、RNG、任务采样/分桶游标、最佳验证值与早停进度。记录解析配置、源码/数据/初始化文件 SHA 和运行时；新 run 不覆盖旧目录。旧 R3 checkpoint 要在原签收代码版本恢复，不能删除源码身份校验强行跨版本 Resume。

独立 `adapter.pt` 包含 LoRA 与所有多模态参数（包括冻结模块），不含 Qwen 主权重和优化器。Tiny 重建还绑定基座 seed。对真实 Tiny 联合运行，重新构造基座后加载 Adapter，**整个模型状态与完整 checkpoint 完全相等**。缺键、多键、shape/dtype/NaN、错误身份均有拒绝测试。

## 四、真实数据与本地运行结果

### 1 来源与可信目标

训练复用 R3 已验收的 32 个任务视图、16 份真实媒体，四任务各 8 条，train/validation/test=24/4/4；每任务训练 6、验证 1。未用 test 选择模型。来源 revision、许可和数据 SHA 延续 R3：`jingyaogong/minimind-o_dataset`，revision `d6588e12ac2ac8ced65eb58a7d7b3eef4aa220de`。官方数据卡混合声明 Apache-2.0/GPL-3.0，未逐文件区分；本轮保持私下诊断，不公开再分发原始数据。

另取同一来源的 4 份原始 question audio/转写检验通用 JSONL prepare：原文件实际为 **12 kHz MP3**，原字节不变，按 2 train/1 validation/1 test 保留既有媒体分配。目标重采样至 24 kHz，经锁定 Mimi 重新编码；新缓存独立进程复读、cache_hit=true，四条 Tensor 与 R3 对应 T2A **逐元素完全相等**。语音输入分支继续独立使用 16 kHz SenseVoice。

规范化 JSONL 支持 T2A/I2T/A2T/A2A/TEXT；MiniMind-O/LJSpeech/AISHELL-3 等是显式来源标签，不代表已写好这些数据集压缩包解析器或已下载它们。原始目标录音必须有来源声明、相同的答案转写和媒体 SHA；真实 QA A2A 要求不同的输入/回答录音。程序能校验字节、声明和分组，不能替代人工核实转写真实性或许可。

云端 D 配置对当前 echo 缓存执行 dry-run **按预期退出 1**：`正式联合训练需要可信 QA 目标语音`。未放宽检查或把复述标成问答成功。

### 2 Tiny 更新和全状态对照

环境：Mac arm64，Python 3.12.14、PyTorch 2.15.0、Transformers 4.57.6；CPU FP32，4 线程，seed 42。媒体侧 FunASR 1.4.17。运行参数、逐步 loss、梯度、源码摘要和路径见 acceptance.json。

| 实际运行 | 步数 | 固定训练探针 Loss 前→后 | 梯度证据 |
| --- | ---: | --- | --- |
| A T2A | 8 | 21.247936 → 20.364468 | Bridge/Talker/Embedding/Heads 非零 |
| B T2A | 4 | 21.247936 → 20.546881 | 同 A；每 2 步验证、保存最佳状态 |
| C Vision | 2 | 11.927815 → 11.916353 | Vision 最大绝对梯度 0.0236581 |
| C Audio | 2 | 12.007847 → 11.999826 | Audio 最大绝对梯度 0.0391219 |
| D Joint | 4 | 16.606010 → 16.457052 | 7 个目标组均非零，LoRA 最大绝对梯度 0.102359 |

表中固定探针每任务 1 条，仅用于持续诊断。另在 **同一完整 6 条 T2A 训练集合** 上独立比较初始/第 8 步模型：总 Loss `21.467454 → 20.497402`，下降 **4.52%**；Audio Loss `9.486065 → 8.516013`。这是有限步数的优化证据，不是充分记忆、泛化或可懂语音证明。

所有运行的冻结参数摘要完全相同，可训练参数摘要变化。D 连续 4 步与新进程 2+2 步的完整 checkpoint **459 个 Tensor 叶节点及所有非 Tensor 状态完全一致，rtol=atol=0**，包括 optimizer、scheduler、scaler、RNG 和 sampler。早停机制在独立 fixture 中以显式大 min_delta 验证：第 2 步停止，第 1 步 best 保留；真实 B 的 4 步未触发早停，不冒充自然收敛。

### 3 评测与失败边界

- 各任务验证返回按有效 token 聚合的 Text/Audio Loss、八码本 CE、STOP precision/recall；采样/辅助 loss 与原定义对照测试通过。
- 真实 I2T 训练诊断交换两个不同图像特征，监督回答位置 logits 最大差 `0.174112`。这证明输入依赖，不能证明语义理解或提升。
- 当前独立 validation 每任务只有 1 条，不能作模态交换。6 条真实 A2T 特征长度不同，没有同形状配对，实际音频调换标为 **not_measured**；音频 fixture 调换已通过，不裁切真实媒体冒充受控对照。
- B 的预测生成只取 prompt；没有输入参考目标 codes。24 步生成 16 帧，经真实 Mimi 得到 **30,720 samples、24 kHz、1.28 秒** WAV。命中预算，文本未 EOS；流式 API 未暴露逐路音频 STOP，明确记录不可观测，不伪称自然结束。
- 本地 SenseVoice 对预测 WAV 返回空转写，实际 CER=WER=1.0；没有填写虚构转写或质量分数。WAV 产出仅证明生成/解码/ASR 入口可用。**音质未通过，人工试听未执行。** 原波形在 `outputs/r4-local/b-demo/audio/prediction.wav`，不提交 Git。
- CPU `profile` 实际完成 1 次 Tiny 更新并保存状态；循环约 0.0349 秒，口径含循环中的验证/存盘，非 GPU 数据或纯 forward 吞吐。原生 A/B/C/D meta inspect 均通过，A/B/C dry-run 通过；D 因 QA 数据不足按预期拒绝。

## 五、实际命令、测试与审查修复

完整命令和退出码见 acceptance.json；以下从项目根目录执行，重复运行必须换新结果目录。`--steps` 是累计目标步数，不能超过 YAML 总预算。

```bash
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
.venv/bin/python scripts/run_experiment.py --help
.venv/bin/python scripts/run_experiment.py train --config configs/v2/r4a_local.yaml --dry-run
.venv/bin/python scripts/run_experiment.py train --config configs/v2/r4a_local.yaml --run-dir outputs/r4-local/a
.venv/bin/python scripts/run_experiment.py train --config configs/v2/r4c_vision_local.yaml --steps 2 --run-dir outputs/r4-local/r4c_vision
.venv/bin/python scripts/run_experiment.py train --config configs/v2/r4c_audio_local.yaml --steps 2 --run-dir outputs/r4-local/r4c_audio
.venv/bin/python scripts/run_experiment.py train --config configs/v2/r4d_local.yaml --run-dir outputs/r4-local/d-full
.venv/bin/python scripts/run_experiment.py train --config configs/v2/r4d_local.yaml --steps 2 --run-dir outputs/r4-local/d-part
.venv/bin/python scripts/run_experiment.py resume --config configs/v2/r4d_local.yaml --steps 4 --resume outputs/r4-local/d-part/step-000002.pt --run-dir outputs/r4-local/d-resumed
.venv/bin/python scripts/run_experiment.py evaluate --config configs/v2/r4d_local.yaml --resume outputs/r4-local/d-full/step-000004.pt --run-dir outputs/r4-local/d-eval
.venv/bin/python scripts/run_experiment.py profile --config configs/v2/r4a_local.yaml --steps 1 --run-dir outputs/r4-local/a-profile
```

B 的实际 ASR 配置为 `outputs/r4-local/r4b-asr.yaml`，继承 `../../configs/v2/r4b_local.yaml`，仅覆盖 `evaluation.max_new_tokens: 24` 和 `asr_dir: /Volumes/TOSHIBA/项目/minimind-o/model/SenseVoiceSmall`，训练前固定；配置身份校验不允许训练后随意换评测配置。

```bash
.venv/bin/python scripts/run_experiment.py train --config outputs/r4-local/r4b-asr.yaml --steps 4 --run-dir outputs/r4-local/b
.venv/bin/python scripts/run_experiment.py demo --config outputs/r4-local/r4b-asr.yaml --resume outputs/r4-local/b/step-000004.pt --run-dir outputs/r4-local/b-demo
.venv/bin/python scripts/run_experiment.py prepare --config outputs/r4-local/audited-t2a.yaml
.venv/bin/python scripts/run_experiment.py train --config outputs/r4-local/r4d-components.yaml --steps 1 --run-dir outputs/r4-local/d-components
```

通用 prepare 共运行两次（首次编码、独立进程复读）；原生五份 `*_h100.yaml` 各跑 inspect/dry-run。数据摘要保留相应配置、SHA 与真实调用，训练全部使用已有 R3 缓存。云端路径覆盖与 JSONL 规范见 [G4 Runbook](../cloud/G4_SFT_RUNBOOK.md)。

| 最终实际检查 | 结果 |
| --- | --- |
| `.venv/bin/python -m pytest -q`（主机离线、OMP=4、MPLCONFIGDIR=/private/tmp/omni-r4-mpl） | **222 passed in 30.99s**，包含主机 MPS 基础检查 |
| 干净 `git archive 7c6f292` + 20 个 R4 源码/配置/测试白名单，独立 import | **214 passed, 1 skipped in 32.92s**；沙箱无 MPS，跳过明确记录；未依赖用户原有脏 P5 文件 |
| `.venv/bin/ruff check omni tests scripts` | 通过 |
| `.venv/bin/ruff format --check omni tests scripts` | 66 文件无需修改；干净快照为 63 文件 |
| `.venv/bin/python -m pip check`、`git diff --check` | 通过；pip 仅提示沙箱缓存目录不可写，自动禁用缓存 |
| `pytest tests/test_r3_training.py tests/test_r4_training.py tests/test_sft.py tests/test_sft_cli.py -q` | 定向 **33 passed** |
| Adapter/Eval + 原 SFT；Source + 原 Data 定向测试 | 分别 **29 passed**、**16 passed** |
| 远程 CI / CUDA / 正式训练 | **未执行** |

过程中保留的失败与修复：首次定向命令误含不存在的 `tests/test_training.py`，pytest exit 4；随后 `tests/test_r3_training.py tests/test_r4_training.py` 为 4 failed/17 passed（fixture 缺 spk_emb 及未完整关闭测试 dropout），扩展至上述四文件后为 1 failed/31 passed（Talker 独立 config.dropout）。补齐 fixture、显式关闭该数学对照中的全部 dropout 并增加配置继承测试后，最终 33 passed；未放宽数值容差。数据手工核验最初误把 STOP 当成 2048，按实际配置 2050 更正断言后通过，数据和实现没有变化。

实现审查同时修复了：单任务窗口抵消 Loss 权重、关闭 LoRA 的 joint 配方误拒绝、QA split 清单展开、CUDA 模态调换设备、各 C 组件组合初始化、初始化文件 SHA 绑定、继承任务预算残留。修复均发生在真实训练冻结源码之前；独立干净回归与所有运行记录的源码 SHA 一致。

## 六、历史保护与交付

以下 P0–P5 的 SHA-256 与开工前完全相同；R0–R3-local 与其余保护文件的完整摘要见 verification.json。

| 文件 | SHA-256 |
| --- | --- |
| P0.md | `01ac1cb1d4bad5c2d938fc5ad60f06d68ec49e11060e01442f38e0d31fcd993f` |
| P1.md | `ecb9052d9162d77d632210c2d5bf18c7ff5b52fcda5fe8a70a6be6df51757473` |
| P2.md | `b1345c087866ad07026c33f0506d82d5a633e8c3ffb33e1eeb8dbe393cbe04b3` |
| P3.md | `14ed2a69dd9340f4373ce35b02f750d7fd372cad5dfeedb9777de46aebcf811a` |
| P4.md | `37b80cf84f4c512ab916cc43a03ae2e761e98a8fa6aabdd3c865ebb01df9040f` |
| P5.md | `302dd9b5ba88afc4ed21079af94777b2960052c2b9aa3ce6992c6dc4196537ed` |

本地日志、checkpoint、预测音频与原始清单位于 `outputs/r4-local/`；新原始媒体缓存位于 `.cache/r4-local/audited-t2a-v1/`。提交只包含本轮源码、配置、测试、文档和 5 份小规模 JSON。G4 云端说明覆盖克隆/固定版本、安装、资产/数据准备、Profile、各阶段训练、独立组件衔接、完整恢复、日志与结果回传。没有为了通过 dry-run 放开 echo QA 边界或自动执行云任务。

## 七、下一步 R5-L 与未解决依赖

R5-L 等待用户授权。入口为 `omni/models/model_omni_v2.py` 中已有 Bridge 注入接口，沿用 `r3_local.py`、schema 2 YAML、版本契约及严格 Adapter/Checkpoint；先做可替换 Bridge 的维度、数学梯度、初始化和 tiny 恢复对照，再为候选配置生成可审查的矩阵。**本次没有实现候选 Adaptive Bridge。**

| 后续工作 | 前置条件与限制 |
| --- | --- |
| R5-L 通用 Bridge 接口/候选 Tiny 正确性 | 用户单独授权；固定输入、hidden 层契约、数据与冻结范围；不能宣称收益 |
| G3 真实 1.7B CUDA BF16、显存/性能/恢复 | 使用实际 GPU；当前本地 Profile 和 meta 参数不替代它 |
| G4-B T2A 收敛与音质 | 合法扩容数据、原生训练、ASR + 人工试听；目前空转写/预算截断不是质量通过 |
| G4-C/D 真实语音对照与 QA A2A | 同形状独立媒体、足够验证量、可信回答波形/转写；补齐后重新 prepare |
| 正式 Bridge 消融、GRPO 与 Benchmark | 先由 G4-D 得到冻结 B1，用户再决定方法、Reward、更新范围和资源 |

阶段到此停止，不自动进入 R5-L。
