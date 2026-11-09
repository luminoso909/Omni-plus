## 一、审计结论

审计日期：2026-11-09。代码基线：`6c497ca0ac72d60bbc8d557fa2ca27ce6c1f5312`，同时检查当前工作区。范围为开发完成度、运行入口、证据与文档；未训练、未下载、未连接云 GPU，未修改模型或训练代码。

**R2–R6-L 的主体模型、SFT、Bridge 和 GRPO 本地工程已完成相应验收；整个项目尚未全部开发、训练和交付完成。** R5 两个 P1 已关闭，R6-L 的 PASS 限于数学、Tiny 更新和恢复。正式数据供应、统一只读评测、全任务 Benchmark、性能仪表与最终 Demo/发布仍有工作。

**代码上可以进入 G3 的环境部署与有限资源验收准备；当前不能拿默认配置直接开始云端正式训练。** 需先搬运固定代码与相容资产、准备云端配置和小批量数据，然后完成获批的 1–10 步 CUDA/恢复测试。G3 通过并批准预算后才能进入 G4。G4D 必须立即完成 B1 首轮 Benchmark，不能等到 G7 才第一次评测。

| 问题 | 结论 |
| --- | --- |
| 本地主要算法/训练代码是否完成 | 是，R2–R6-L 对应范围有源码、测试和运行记录；不能扩展为完整 Benchmark/正式数据工程都已完成。 |
| 当前是否发现推翻 R5/R6 本地 PASS 的证据 | 本次定向检查未发现。未重新执行全仓或真实大型资产验收，不冒充新一轮完整数学审查。 |
| 是否可以直接正式训练 | 否。资产、正式数据、CUDA 数值/资源/恢复、预算与 B1 评测必须按闸门完成。 |
| G3–G7 文档是否存在 | 本轮补齐 G7 后均存在；G3 有用户未提交的路径迁移，旧规划示例与 G3 部分早期说明须按第六章纠正。文档存在不等于所有评测入口已实现或云端已通过。 |

## 二、证据与历史状态

阅读当前 `AGENTS.md`、README、主规划、两份 v2.1 手册、待办，以及 R0、R1、R2、R3-local、R4-local、R5-local、R6-local 和 R5/R6 最新审查记录；对照源码、配置和历史实验摘要。以下历史测试数只描述当时验收，本次测试单列第七章。

| 阶段 | 实际代码/证据 | 完成边界 |
| --- | --- | --- |
| R0/R1 | `docs/stages/R0.md`、`R1.md`；`omni/models/thinker/qwen_backend.py`、`scripts/check_qwen_backend.py`、`experiments/r1/` | 审计与独立官方 1.7B Backend 已验收，真实 CPU FP32 parity/Cache/Tokenizer；不是 Omni 训练。 |
| R2 | `model_omni_v2.py`、`qwen_omni.py`、`scripts/check_r2.py`、`experiments/r2/20261108-cpu-fp32/manifest.json` | 真实多模态路由、T2A 时间轴、梯度、Cache、PCM 接线已验收；新 Talker 质量未训练验收。 |
| R3-L | `r3_prepare.py`、`r3_local.py`、`tests/test_r3_*`、阶段报告 | 四任务各 8 个诊断视图、16 个原始媒体、24/4/4 分组；Tiny 更新和跨进程恢复。A2A 为 echo。 |
| R4-L | `sft.py`、`losses.py`、`adapters.py`、`tests/test_r4_*`、阶段报告 | A/B/C/D 本地更新、LoRA、验证、完整恢复通过；历史 222 项测试。随机音频曾解码/ASR，但未通过音质或 B1 验收。 |
| R5-L | `bridge.py`、`bridge_splits.py`、`run_source.py`、`bridge_ablation.py`；最终修复提交 `c07f1df` | 四种 Bridge 本地通过；两项 P1 最终关闭，P2 成本汇总通过。旧复审 `6cd4988` 的未关闭结论不能替代后续修复结果。 |
| R6-L | `grpo.py`、`grpo_rollout.py`、`grpo_rewards.py`、`r6_local.py`；`experiments/r6-local/20261109-cpu-fp32/` | Text/Audio/Joint 实际 Tiny 更新，17 条隔离 CLI，跨进程 519 Tensor/1775 非 Tensor 叶精确一致；工作区 399 passed/1 skipped、干净归档 392/1。为同轮自审，非外部独立审查。 |
| G3–G7 | 各 Runbook、配置模板 | **全部未执行正式云端验收**。没有 B1，也没有可声称的 Bridge/GRPO 质量收益。 |

R5 修复在当前源码中的闭环：

- `omni/data/sft_sources.py:115` 附近在选样预算之外保存完整 `identity_groups`，包含跨任务及未选中媒体别名；`omni/data/bridge_splits.py:61` 起核对组摘要、成员及递归祖先 split，缺身份拒绝正式比较。
- `omni/training/run_source.py:20` 起检查实际导入的 `__file__`、`__spec__.origin` 与包搜索路径；两个 CLI 和运行身份 API 均调用。不能靠当前目录或 Git SHA 冒充实际源码来源。
- R6 通过 `policy_source → read_bridge_base` 复用隔离闸门；`tests/test_r6_integrity.py` 覆盖换样本 ID、丢别名、缺身份、跨 checkout 与合法正例。本次全部通过。
- 该保护只覆盖已登记的身份关系；不具备对任意未登记转码、裁剪或语义近重复的自动识别能力。此限制不等于原 P1 重新打开。

## 三、模块与流程核对

### 1 模型与训练

| 能力 | 实际入口 | 现状与限制 |
| --- | --- | --- |
| Qwen Omni | `omni/models/model_omni_v2.py:OmniPlusOmni.forward/stream_generate`；`thinker/qwen_backend.py` | 官方 Qwen 原生 Tokenizer/Cache；显式注入媒体特征。单样本流式、左 padding，缺媒体报错。没有批量在线服务。 |
| Projector/Bridge | `encoders/model_encoders.py:Projector`；`bridge.py` | 768/512→2048；Fixed/Last/Weighted/Gated 统一 2048→768，投影后 RMSNorm/Mask。取零基原始 Block，末层在最终 Norm 前；Talker 内部该投影为 Identity。 |
| Talker/Mimi | `talker/model_talker.py`；`data/audio_alignment.py`；`inference/codec.py` | 4×768、8 路头，真实码本 2048、含特殊 ID 的头词表 2112；Delay/Teacher Forcing/STOP/PAD 共用实现。 |
| PCM Streaming | `inference/streaming.py:MimiStreamingDecoder/stream_waveform` | 有 PCM 事件、尾块/取消与状态管理；Qwen 对接经 R2 验收。现有 Qwen `demo` 把生成事件收集完再整段解码，不是终端实时 PCM 展示。 |
| T2A A/B | `r3_local.py:smoke` → `sft.train_step`；`configs/v2/r4{a,b}_{local,h100}.yaml` | 正确 Loss/累积、验证/早停、保存恢复；正式全宽收敛及可懂语音未验。 |
| I2T/A2T C | `configure_parameters` 的 `vision/audio`；`r4c_*` 配置 | 仅对应 Projector 更新，保留穿过冻结 Qwen 的梯度；调换特征诊断不等于 VQA/ASR 任务质量。 |
| A2A/联合 D | `sft_sources.py`；`r4d_*`；`adapters.py` | QA/echo 区分、任务采样/权重、LoRA、独立组件初始化/恢复已有；正式 QA 目标数据缺失。 |
| GRPO | `grpo*.py`、`r6_local.py`；六份 `r6_*` 配置 | 同一训练器支持三模式；真实数学/参数更新。Old 固定每轮，Reference 固定初始 SFT；有效头 logprob 求和，精确条件 KL，序列/物理步两种归一化。 |
| GRPO 边界 | `validate_grpo`、`prepare_run`、`reward_score` | 只支持完整合法支持集的正温度采样，拒绝截断 Top-k/Top-p；Audio 模式冻结文本策略。Reward API 可扩展，CLI 内置指标有限，正式 Reward 尚未确定。 |

### 2 数据工程

已有原生小子集下载器 `scripts/collect_p4.py`，以及通用审计 JSONL → 重新编码 → 缓存的 `sft_sources.py`。后者支持 `text/t2a/i2t/a2t/a2a`，原始媒体 SHA、来源、revision、许可声明和固定 split；SenseVoice 16kHz/Mimi 24kHz 分路；未知 Mimi codes 不混用。缓存绑定 Tokenizer、模型文件、Processor/依赖与实现摘要，失配拒绝，发布缓存使用临时目录再重命名。

**尚不是所有候选正式数据集的一键生产流水线。** 未提供 LJSpeech/AISHELL/LibriSpeech/COCO/LLaVA/GSM8K 等所有原始格式的统一有界下载、转换与语义审计；合法性和 transcript 真实性依赖清单编制者。`read_sources()` 把 JSONL 整体读入；`prepare_sources()` 累积全部选中 Tensor，再按 split 保存；`load_prepared()` 整个 split 加载。有限 Pilot 可用，大规模成本尚未证明。

当前默认 `reference_root=/Volumes/TOSHIBA/项目/minimind-o` 不存在，云模板所需 `weights/media-reference` 也不存在；本地 Qwen3-1.7B 与旧 Tiny 缓存存在。不能仅凭缓存存在跳过原资产校验，也不能用随机 fixture 替代真实 Mimi/ASR 重新验收。完整数据集未下载；现有 mini 子集许可与可公开范围仍须逐源核实。

### 3 恢复、指纹与指标

`sft.py:save_checkpoint/load_checkpoint` 保存模型、Optimizer、Scaler、RNG 和进度；R4/R6 将 Scheduler、采样游标、配置/源码/数据身份放入进度。RL 另含 Reference、冻结摘要、Policy 版本和完整 Rollout 边界。完整 Resume 使用原配置、资产和源码，新结果目录；独立 Adapter 不包含 Optimizer。

R5 矩阵可保存共同 B1、数据、层选择、参数量、终点、硬件与 ASR 身份，`collect` 区分训练循环成本和推理成本，保留失败行。R6 保存 Reward/KL/Loss 与逐轨迹记录。**不存在一个自动贯穿 B1、G5、G6、G7 全部模型与指标的统一汇总工具。** 原始 Manifest/JSON/CSV 可作为后续输入，但不能把路径和字段相似视为已经统一。

严格恢复与最终只读评测需分开：`r3_local.py:1005` 和 `r6_local.py:311` 均比较完整身份；其 `settings` 包含评测 split、长度、设备等。当前修改 `evaluation.benchmark` 为 test 会改变身份并被现有加载门槛拒绝。提前锁定 test 的原配置可以评估该 test；但还不能自由地对同一已训练权重执行独立评测协议/系统设置扫描。不能通过删身份校验或重签历史 Checkpoint 解决。

## 四、未完成任务分类

优先级 P1 为影响正式流程/可信比较的缺口，P2 为可靠性或交付改进；本次未发现新的 P0 算法阻断。A 项不要求重写模型。未获授权时只登记，不实施。

### A：正式训练前的本地工程缺口

| ID / 优先级 | 当前状态与实际路径 | 具体缺口、最小建议 | 完成阶段 |
| --- | --- | --- | --- |
| A1 / P1 | `r3_local.py:evaluate_run`、`r6_local.py:prepare_run`、`adapters.py` 已有严格加载 | 增加统一只读推理加载和独立 Eval 配置：锁定模型/Adapter/资产与训练来源，另记录评测源码、固定数据和生成协议，并检查全祖先隔离。不得放松 Resume。否则训练后新测 test/AMP/长度会受完整 settings 限制。 | 正式 G4/B1 配置冻结前补齐；有限 G3 可先行。 |
| A2 / P1 | `collect_p4.py` 仅旧小子集；`sft_sources.py:read_sources` 接受已审计 JSONL | 为**实际获批数据源**补可复现的清单编制/转换步骤或小脚本，保留原始 ID、官方 split、未选别名、许可和 QA 目标溯源；核验跨源转码/裁剪与文本近重复。无需先实现所有候选下载器。 | 每个数据源进入 G4/G6 正式训练之前。 |
| A3 / P2 | `scripts/run_experiment.py:103–135` SFT dry-run；R6 已有共享预检 | SFT dry-run 检查 train/validation 和文件存在，却不执行完整 Checkpoint 身份、实际评测 split 或初始化权重兼容检查。不能把它当作正式恢复/评测通过；建议复用真实预检，补错 Checkpoint/test 缺失正反例。 | G4 长训前；G3 仍须实际新进程恢复，不以 dry-run 代替。 |
| A4 / P1（规模条件） | `sft_sources.py:prepare_sources`、`r3_prepare.py:load_prepared`、`r3_local.py:next_batches` 使用内存列表 | 大规模选样/特征缓存尚无分片、懒加载及可恢复预处理。先测获批规模峰值与磁盘；超内存时再做分片读取和断点准备。保持 ID/顺序/RNG/contract 不变，并补恢复对照。 | 超出已测可载入 Pilot 的 G4/G6 扩容前；不阻塞小 G3，不强制重写现有训练器。 |

### B：代码已有，云端尚待验收

| ID / 优先级 | 当前状态与路径 | 必须实测/实施内容 | 阶段 |
| --- | --- | --- | --- |
| B1 / P1 | `configs/assets_v2.yaml`、`configs/v2/*_h100.yaml` 有模板 | 固定提交、CUDA wheel/依赖、相容媒体模型、可信有限数据和云路径；重新准备云缓存。模板的 Mac 路径、B1 占位符不能直接运行。 | G3；B1 来源在 G5/G6 前补齐。 |
| B2 / P1 | 全尺寸 meta 统计与 CPU 接口已有 | 真实 1.7B+4×768 的 FP32/BF16 前反向、冻结、Cache 重放误差、新进程恢复、显存/吞吐/存盘峰值；据此预算，不直接放宽断言。 | G3，后续各配方变化再 Profile。 |
| B3 / P1 | R4 A/B 可训练，Tiny 音频未达质量 | 原生模型短过拟合、T2A 收敛、双语长短句、CER/WER+失败率+人工试听；决定随机 Talker 是否继续。 | G4-A/B。 |
| B4 / P1 | 两路 Projector/LoRA 代码已有 | 真实 I2T/A2T 条件贡献、可信 QA A2A、联合冲突/文本退化检查；建立 B1 并立即首轮 Benchmark。 | G4-C/D。 |
| B5 / P1 | 四 Bridge 与公平矩阵已实现 | 同一 B1、Split/种子/预算的实际重训练、质量和成本对比；候选/超参数待用户决定，允许负结果。 | G5。 |
| B6 / P1 | 三模式 GRPO 与严格恢复已有 | 可靠 Reward、真实 Mimi/ASR、零方差/Reward Hacking/词表对齐行解码、CUDA 资源与独立质量对照；Audio/Joint 以可靠语音奖励为前提。 | G6。 |
| B7 / P2 | 原始 JSON/CSV/Checkpoint 已保存 | 云端断连、磁盘限额、备份/回传摘要、长运行恢复；记录真实 CUDA 软件锁及硬件，不能把 Mac 的 `pip check` 当替代。 | G3–G6 持续执行。 |

### C：Benchmark 与最终交付缺口

| ID / 优先级 | 当前状态与路径 | 缺口与实施建议 | 最迟阶段 |
| --- | --- | --- | --- |
| C1 / P1 | `evaluation/sft.py` 有 teacher-forced Loss/STOP/特征调换；`bridge_benchmark.py` 仅生成 T2A；`grpo-evaluate` 汇总 Reward | 新增 prompt-only Text/VQA/A2T/QA A2A 预测、统一分任务 Accuracy/CER/WER 和失败覆盖率；冻结评测协议、模板和 test，接 A1。不能用 CE、特征敏感性或 Reward 冒充能力。 | **G4D 的首次 B1 Benchmark 前**，G5/G6 复用，G7 总复核。 |
| C2 / P1 | T2A 仅模型流总时长/CUDA peak；`scripts/check_p5.py` 有历史 MiniMind TTFT/TTFA/RTF 口径，但当前未跟踪 | 为 Qwen/GRPO 同一推理路径增加逐事件时间戳、真实首 PCM、RTF、P95 gap、吞吐、OOM/失败率；同硬件 warmup/重复测量，区分模型与完整解码。 | 基础指标 G4D；系统矩阵 G7。 |
| C3 / P1 | R5 `collect` 与 `loss.svg` 可用；没有跨方法报告器 | B1/Bridge/GRPO 同协议结果表、重复种子/不确定性、训练/推理成本、负结果和失败例；生成统一曲线和原始导出，不能直接混合不同 Reward 标度。 | G5/G6 逐步积累，G7 完成。 |
| C4 / P2 | PCM API 与 T2A 离线 WAV 有；完整交互 Demo/发布包无 | Text/图像/语音输入的 Qwen Demo、许可允许的短音频样例、模型卡、技术报告、架构图、安装/资产获取/推理复现与发布摘要。确定项目自身许可证，不自动再分发第三方资源。 | G7。 |
| C5 / P2 | 本轮补齐 G7 Runbook；G3/P5 等仍有既有未提交文件 | 由用户决定历史文件的单独提交/迁移；发布 checkout 必须能独立包含承诺交付的文档/脚本，不依赖当前工作区额外文件。 | 云端代码交付前确认所选提交；公开完整成果前解决历史发布缺项。 |

### D：可选优化

| ID / 优先级 | 现状与路径 | 建议/阶段 |
| --- | --- | --- |
| D1 / P2 | `sft.py` 有训练 autocast；没有 Triton kernel/统一推理 AMP 扫描 | 先测瓶颈，再做 AMP/Cache 的正确性与性能矩阵；Triton 仅有收益假设时实现，G7 可报告未执行。 |
| D2 / P2 | `r3_local.py`、`r6_local.py` 单设备；gradient checkpointing 明确拒绝 | 单卡不足再考虑已验证的激活重算、offload 或双卡分工；DDP/FSDP/ZeRO 不是核心完成前提，不能直接 torchrun 冒充支持。 |
| D3 / P2 | 当前视觉 64 Tokens、固定 4×768 Talker | Qwen3.5、动态视觉 Token、CAM++、更复杂 Bridge、更多 Backbone/服务 UI 留后续，不影响本轮主线。 |

## 五、现有评测指标的真实范围

| 指标 | 已有计算 | 尚缺/边界 |
| --- | --- | --- |
| Loss/码本 CE/STOP | `evaluate_sft` | Teacher forcing；STOP precision/recall 不代表自由生成停止率。 |
| CER/WER | `transcription_errors` + 预测 WAV 的本地 ASR | T2A 有逐样本与 micro 汇总；中文 WER 目前按空白分词，正式需语言协议。失败覆盖率必须同报，空输出不能丢掉。 |
| Accuracy/语义/音质 | `text_match` 仅 RL Token 严格匹配；自定义 Reward 注入 API | 尚无统一 Text/VQA/QA Benchmark、UTMOS 或人工试听结果录入/汇总器。可懂度与自然度分开。 |
| 流延迟/显存 | `benchmark_audio` 的 stream 总时长、CUDA allocated/reserved | warmup=0；排除 Mimi/ASR/文件 IO，不能当 TTFA/端到端 RTF。CPU peak 为空，不能填 0。 |
| 训练成本 | SFT/GRPO 循环秒数、步数、CUDA peak；R5 collect 分列 | 包含验证/存盘或 Rollout/Reward；不是纯 kernel 吞吐。恢复 run 只计本段，不能重复累加旧步数。 |

## 六、G3–G7 Runbook 与工作流衔接

| 文档 | 当前状态 | 使用规则 |
| --- | --- | --- |
| G3 | HEAD 路径 `docs/G3_CLOUD_RUNBOOK.md`；用户工作区已移动到 `docs/cloud/G3_CLOUD_RUNBOOK.md`，字节相同但未提交 | 本次保护迁移，不纳入提交。旧文中“train/resume/evaluate/profile 未实现”已过时；它们在 R4 实现。 |
| G4 | `docs/cloud/G4_SFT_RUNBOOK.md` | 实际 A/B/C/D CLI、数据格式、组件初始化/恢复；首轮 Benchmark 的完整性依赖 A1/C1/C2，不能只执行现有 `evaluate` 就宣布 B1。 |
| G5 | `docs/cloud/G5_BRIDGE_RUNBOOK.md` | 实际 plan/train/demo/collect、完整媒体组/祖先与源码身份；`collect` 只覆盖原 Bridge 矩阵。 |
| G6 | `docs/cloud/G6_GRPO_RUNBOOK.md` | 实际三模式/基线/恢复/Profile、正式 B1 与 Reward 门槛；通用质量对照仍依赖 C1。 |
| G7 | [G7_FINAL_RUNBOOK.md](../cloud/G7_FINAL_RUNBOOK.md) | 本轮新增：已有命令与待实现入口分列；统一质量、效率、公平比较和成果交付。不是 G7 实验通过。 |

当前薄 CLI 的真实参数为 `--config/--run-dir/--resume/--steps/--dry-run`，GRPO 另有 `--baseline`。总规划中 `configs/v2/model.yaml`、`eval_b1.yaml`、`--checkpoint/--run-id/--device/--max-samples`、`export` 等是明确标为目标的旧示例，**不能复制当作现在的命令**。

G3 启动前的具体清单：

1. 由用户授权云资源和数据预算；固定完整提交并在该 checkout 建环境，别复用另一目录的 editable install。干净提交具备 R6 工程，不必擅自提交当前历史改动。
2. 搬运/恢复与 R2 资产锁一致的 SenseVoice/SigLIP/Mimi、Qwen 与合法原始小样本，核验 SHA。新路径显式写入派生 YAML，保持旧资产和缓存不变。
3. 优先参照 G4 手册，用 `r4a_h100.yaml` 派生 **schema_version: 2** 的有限 T2A 配置和新缓存，便于周期保存/恢复。若沿用 `g3_h100.yaml`，它是 schema 1、最多 10 步；仅改 `--steps 100` 不会生效。
4. 以 `python -m pytest` 执行普通回归；`inspect/prepare/--dry-run` 后实际做获批 1–10 步及新进程恢复。schema 2 配置预算明确提高且获批后，才可 `profile --steps 100`。
5. 固定 BF16/FP32、序列长度、batch/累积、训练范围，记录真实峰值和成本；通过 G3 后再批准 G4，先关闭适用 A 项并准备 B1 首轮评测。

`profile` 和 `grpo-profile` 都会真正更新参数，不是本次审计可执行的只读命令。未来恢复必须使用原 resolved 配置，资产/环境改变时不得删除身份检查强行恢复。

## 七、本轮轻量验证与保护

| 实际命令/检查 | 结果 |
| --- | --- |
| `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python -m pytest -q tests/test_r4_evaluation.py tests/test_r5_data_identity.py tests/test_r5_source_origin.py tests/test_r6_math.py tests/test_r6_integrity.py` | **51 passed in 15.49s**；离线 Tiny/临时目录，无正式训练或大型资产。 |
| `.venv/bin/python scripts/run_experiment.py --help`；`bridge_ablation.py --help`、`plan --help`、`collect --help` | 四条 exit 0，参数与本手册对应。 |
| 逐份调用 `read_settings/read_grpo_settings` 读取 20 份运行 YAML（排除两个候选列表） | 17 份通过配置/本地 Qwen 文件存在检查；3 份 G6 H100 模板因 `outputs/g6/b1/step-XXXXXX.pt` 缺失明确拒绝，属预期前置条件。没有把“解析通过”记为资产或 CUDA 通过。 |
| 实际 `run_identity` 比较原 R4A 与增加 test benchmark 的配置 | 身份不相等，印证 A1；未加载 Checkpoint 或运行评测。 |
| `.venv/bin/ruff check omni tests scripts` | All checks passed。 |
| `.venv/bin/ruff format --check omni tests scripts` | 88 files already formatted。 |
| `.venv/bin/python -m pip check` | No broken requirements found；沙箱 pip 缓存权限提示不影响结果。 |
| 文档命令/链接/变更范围、`git diff --check` | 10 个 Bash 代码块 `bash -n`、1 个 YAML 片段解析、新文档本地链接与 Diff 空白检查通过；云端段未执行。 |

**没有重跑全仓、完整 Tiny 训练、真实 1.7B/Mimi/ASR 或云端验证。** R6 的 399/392 项全仓记录来自原阶段，相关源码本轮不改。当前媒体资产缺失按现状报告，不通过网络或 fixture 隐藏依赖。

开工保护基线位于 `/private/tmp/omni-development-audit-dcddbif1`：235 个已有源码/文档等文件的 SHA/缺失状态，697 个既有实验/资产文件的大小和 mtime。核验结果：除允许更新的 README/待办外，其余 233 个文件状态不变，697 个资产/实验文件大小和 mtime 不变，仅新增指定两份文档。P0–P5、R0–R6、审查原文、用户未提交代码/Notebook/历史结果不动。以下 P0–P5 SHA 同时匹配开工基线与 R6 的历史保护记录：

| 文件 | SHA-256 |
| --- | --- |
| P0.md | `01ac1cb1d4bad5c2d938fc5ad60f06d68ec49e11060e01442f38e0d31fcd993f` |
| P1.md | `ecb9052d9162d77d632210c2d5bf18c7ff5b52fcda5fe8a70a6be6df51757473` |
| P2.md | `b1345c087866ad07026c33f0506d82d5a633e8c3ffb33e1eeb8dbe393cbe04b3` |
| P3.md | `14ed2a69dd9340f4373ce35b02f750d7fd372cad5dfeedb9777de46aebcf811a` |
| P4.md | `37b80cf84f4c512ab916cc43a03ae2e761e98a8fa6aabdd3c865ebb01df9040f` |
| P5.md | `302dd9b5ba88afc4ed21079af94777b2960052c2b9aa3ce6992c6dc4196537ed` |

本轮文件清单：新增本报告和 `docs/cloud/G7_FINAL_RUNBOOK.md`；README/待办只增加审计结论、文档链接和缺口状态。仅本地文档提交，不 Push，不代替用户决定或启动 G3–G7。
