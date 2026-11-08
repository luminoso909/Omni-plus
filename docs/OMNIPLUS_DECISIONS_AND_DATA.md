# OmniPlus｜项目认知、参数、数据、训练与实验总表

> **用途**：本人快速理解项目、交给 Codex 对照实现的事实与决策清单。与 `OmniPlus v2.1｜本地代码先行与云 GPU 实验执行手册` 配套使用。  
> **依据**：已上传的 `PROJECT_PLAN.md`（v2.0）、P0–P5 阶段报告及此前讨论；**带“建议/估算”的数据比例、参数与训练时间未经本项目正式验证**。  
> **状态更新（2026-11-08）**：R4-L 本地 A/B/C/D 代码与 Tiny 验收已完成。[R4-L 报告](stages/R4-local.md)记录真实样本、参数统计、LoRA 与恢复结果；正式训练与 B1 未执行。当前 A2A 为 echo 诊断，Tiny WAV 不代表音质通过。下文带估算的数值继续作为规划参考。

## 1. 这个项目究竟是什么？

**一句话**：以**官方非 Base 的 Qwen3-1.7B**为预训练 Thinker，冻结外部视觉、语音与神经音频 Codec，自主实现/训练多模态 Projectors、Fixed Bridge、4×768 Talker/八码本声学预测头；先得到可信的 **B1：Qwen Omni SFT Baseline**，再分别研究 **Adaptive Bridge** 与 **Multimodal GRPO**，最后完成系统与质量 Benchmark。

**不是**：重新从零预训练 1.7B 语言模型；同时正式训练 MiniMind 全模型与 Qwen 做不公平架构对比；单纯让 Qwen LoRA 在几千条多模态数据上「自动学会说话」；为了简历强行加 DDP/MoE/视觉 Resampler。

| 维度 | 已确定 | 未最终确定/何时决定 |
|---|---|---|
| 主干 | Qwen3-1.7B 非 Base，0.6B 可用于轻量测试 | LoRA rank、target modules：R4-D |
| 冻结模块 | SigLIP2、SenseVoice、Mimi；CAM++ 可选 | 资产精确 revision：R2/R3 manifest |
| Talker | MiniMind-O 参考结构：4 Transformer layers、hidden 768、8 Codebooks | 从零随机初始化是否达到质量要求：G4A/G4B |
| Bridge | 首版固定层 2048→768，可替换接口 | 哪些 Qwen 层/融合函数：G5 前设计 |
| 第一版任务 | T2A、I2T、A2T/A2A，联合 SFT | 数据比例与训练预算：G3/G4 Profile |
| RL | 在 B1 起点上准备 Text/Audio/Joint GRPO | Reward、策略更新范围、单/多 GPU：R6 实验前 |
| GPU | 单 NVIDIA GPU 为主；Mac 本地开发 | GRPO 是否多卡：G6 Profile |
| 优先级 | B1 多模态模型 → R5/R6 研究 → R7 系统评测 | 可选视觉效率、MoE、分布式：不阻塞主线 |

## 2. 总体架构：训练与冻结的关系

```text
                              OmniPlus（第一版）
TEXT ── Qwen3 Tokenizer ──────────────────────────────────────────┐
IMAGE ─ SigLIP2 Vision   [FROZEN, pretrained] ─ VisionProj [TRAIN]┤
AUDIO ─ SenseVoiceSmall  [FROZEN, pretrained] ─ AudioProj  [TRAIN]┤
                                                                  ▼
                                                  Qwen3-1.7B Thinker
                                           [PRETRAINED; R4 先 FROZEN]
                                           [R4-D 按需 Qwen LoRA TRAIN]
                                                    │          │
                                               Text Head    Hidden States
                                                    │          │
                                              TEXT OUTPUT   FixedBridge [TRAIN]
                                                                2048→768
                                                                  │
                                   History Mimi Codes ────────────┤
                                  Speaker Conditioning（可选）───────┤
                                                                  ▼
                                                  Talker 4×768 [TRAIN]
                                                                  │
                                                    8 Audio Heads [TRAIN]
                                                                  │
                                                      8 Mimi Codebooks
                                                                  │
                                           Mimi Decoder [FROZEN, pretrained]
                                                                  ▼
                                                        Streaming WAV/PCM

训练监督：真实目标 WAV → Mimi Encoder [FROZEN] → 8 路 Codes → Audio CE；
         真实回答文本 ───────────────────────────────────────→ Text CE。
```

**重要边界**：Mimi 的 Encoder/Decoder 是同一 Codec 的两个方向，参数统计不能重复相加；MiniMind 原有 `Talker.embed_proj` 与新 FixedBridge 应有明确唯一的 2048→768 投影所有权，避免重复映射。Qwen Token 并不自然与 12.5Hz 音频帧一对一对应；必须重新构造时序/Mask。

## 3. 模型参数量：把“总参数”和“更新参数”分开

| 模块 | 权重来源 | 规模 | 是否更新 | 注释 |
|---|---|---:|---|---|
| **Qwen3-1.7B 主干** | `Qwen/Qwen3-1.7B` | **约 1.72B** | R4-A/B 冻结；R4-D LoRA 可选 | 冻结不代表无需计算激活 |
| SigLIP2 Vision | `google/siglip2-base-patch32-256` 或 P2 兼容本地裁剪版 | 约 95M* | ❄️ 冻结 | 真实目录/加载类需核验 |
| SenseVoiceSmall | `FunAudioLLM/SenseVoiceSmall` 或 P2 已验证本地权重 | 约 234M* | ❄️ 冻结 | 训练仅用其特征 |
| Mimi Codec | `kyutai/mimi` 或 P3 相容版本 | 约 96M* | ❄️ 冻结 | 一次计入整个 Codec |
| Vision Projector | 新初始化 | 约 5–6M* | 🔥 | 768→2048，取决于 MLP 架构 |
| Audio Projector | 新初始化 | 约 5–6M* | 🔥 | 512→2048 |
| Fixed Bridge | 新初始化 | 约 **1.57M**（Linear 例） | 🔥 | 2048→768；MLP 会增加 |
| Talker + Audio Heads / Embedding | 参考 MiniMind-O，首选随机初始化 | 约 **47M*** | 🔥 | 是否包含旧投影/层归一化须审计避免重复 |
| Qwen LoRA（若使用） | 新初始化 | **约数百万到一千多万**，随 target/rank | 🔥（R4-D 等） | LoRA 是参数化方法，不是独立 SFT 阶段 |

*为规划口径而非用户仓库 `sum(p.numel())` 真实实测；显存与可训练参数须由 R2/R3 程序统计并按模块输出。粗略整体运行参数量约 **2.2B**；R4 初期新训练参数约 **5千万～6千万**，加 LoRA 后更新参数有所增加。不要把**总参数 2.2B**写成“从零训练 2.2B”。

## 4. 各阶段更新哪些模块？

符号：**T = 更新**，**F = 冻结（可参与前向与必要梯度传播）**，`—` = 当前任务不使用。

| 模块 | R4-A/B T2A | R4-C Vision | R4-C Audio | R4-D Joint SFT | R5 研究 | R6 研究 |
|---|---|---|---|---|---|---|
| Qwen 主权重 | F | F | F | F | F | F（默认） |
| Qwen LoRA | — | 可选 | 可选 | T（按需） | 固定实验控制 | 何者更新待决 |
| SigLIP2 | — | F | — | F | F | F |
| SenseVoice | — | — | F | F | F | F |
| Mimi Encoder/Decoder | F | — | — | F | F | F |
| Vision Projector | — | T | — | T | 控制不变 | 待决 |
| Audio Projector | — | — | T | T | 控制不变 | 待决 |
| Fixed / Adaptive Bridge | T | — | — | T | **主要研究变量** | 待决 |
| Talker / Embedding / 8 Heads | T | — | — | T | 依公平消融方案 | 待决 |
| Speaker Projector（可选） | 暂缓 | — | — | 按需 | 控制不变 | 待决 |

**关键**：学习 Projector 时，Qwen 参数可以 F，但依然必须保留从 Qwen loss 到 Projector 输入的梯度通路；训练 Talker 时因 Qwen 输入文本固定可按资源情况缓存 Qwen hidden，但缓存必须绑定模型/层/模板/文本版本。

## 5. 数据集：地址、是否有 Caption/Transcript、用途与优先级

### 5.1 语音与多模态交互

| 数据集 | 官方入口 | 大致规模（规划参考） | 标签 | 推荐工作 |
|---|---|---:|---|---|
| MiniMind-O T2A mini | <https://huggingface.co/datasets/jingyaogong/minimind-o_dataset> | 约 470h | 文本 ↔ Mimi 类声码 | T2A pilot；确认码本版本 |
| MiniMind-O T2A full | 同上 | 约 1636h | 文本 ↔ 声码 | 正式 Talker 数据主来源 |
| LJSpeech | <https://keithito.com/LJ-Speech-Dataset/> | 约 24h | 语音+Transcript | 小样本音频测试/T2A 起步 |
| AISHELL-3 | <https://www.openslr.org/93> | 约 85h | 中文语音+Transcript、说话人 | 中文 T2A 扩展 |
| LibriTTS-R | <https://www.openslr.org/141> | 约 585h | 英语 TTS 转录/说话人 | 训练多说话人声学能力 |
| LibriSpeech | <https://www.openslr.org/12> | 约 960h（可用 100h 子集） | 语音+Transcript | Audio Projector 语音识别/输入对齐 |
| MiniMind-O A2A mini | MiniMind-O 同仓库 | 输入约 75h/输出约 57h | 输入语音+答复文本/音频 | 语音交互 pilot |
| MiniMind-O A2A full | 同上 | 输入约 1712h/输出约 423h | 输入语音+答复文本/音频 | 联合 SFT 扩容 |
| VoiceAssistant-400K | <https://huggingface.co/datasets/gpt-omni/VoiceAssistant-400K> | 约 47 万记录 | 指令/语音/回答、SNAC 码 | 备选；需要转换至同一 Mimi 体系 |

注：**语音场景用 Transcript/Answer Text，不叫 Image Caption**。MiniMind mini/full 有重叠时需严格排重；以上时长是前序讨论的近似量，不代表已下载。音频采样率：SenseVoice 输入分支 16k、Mimi 分支通常 24k，各自独立处理与版本确认。

### 5.2 视觉与文本

| 数据集 | 官方入口 | 大致规模 | 是否带 Caption / 目标 | 建议用途 |
|---|---|---:|---|---|
| COCO 2017 | <https://cocodataset.org> | 约 118,287 张训练图 | **有 Caption**（每图通常 5 条），另有标注 | Vision Projector 初始训练 |
| LLaVA-Instruct-150K | <https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K> | 约 15 万级视觉指令样本 | 有视觉问答/详细描述；不等于纯 Caption | 视觉指令 SFT；需获取配套原图 |
| MiniMind-V Dataset | <https://huggingface.co/datasets/jingyaogong/minimind-v_dataset> | 较大（含描述和 SFT） | Caption + VQA/混合任务 | 视觉数据不足时扩展 |
| TextVQA | <https://huggingface.co/datasets/facebook/textvqa> | 约 3.46 万训练问题 | **有问题/答案**，不是通用 Caption | 后期验证文字细节能力 |
| UltraChat-200K | <https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k> | 约 20 万量级对话 | 文本指令/回答 | 联合 SFT 保持文本能力（可选） |
| GSM8K | <https://huggingface.co/datasets/openai/gsm8k> | 约 7.5k 训练题 | 有可核查数学答案 | Text GRPO 的候选，不预先确定 |

**数据可行性判断**：这些数据足够做系统训练与研究的首轮实验，但**不能只凭数据量保证 Talker 随机初始化学出高音质**。Talker 最难；Vision Projector 相对容易启动。Qwen 已预训练，不需要额外用 FineWeb/RedPajama 进行完整 LM 预训练。

## 6. 数据使用比例：此前讨论的推荐起点，不是强制配方

这里所有百分比都是**官方 train / 经审计的可用训练部分的抽样比例**，不是 train:val:test 之比，也不表示必须一次下载这些全部文件。

| 训练对象 / 阶段 | 本地 tiny / 初始 | 云端首轮正式配方建议 | 注意事项 |
|---|---|---|---|
| **Talker + Bridge（R4-A/B）** | 8–32 短句；可从 T2A mini **1% 以内**取 pilot | MiniMind-O T2A full **30%–50%**（先子集，按收敛扩展） | 不以长训练代替质量监控；mini/full 去重 |
| **Vision Projector（R4-C）** | COCO **1% 以内** | COCO **50%** + LLaVA-Instruct **50%** | 先以**不同图片数量**统计，COCO/LLaVA 交叉去重 |
| **Audio Projector（R4-C）** | LibriSpeech clean-100 **1% 以内** | LibriSpeech clean-100 **50%** + A2A mini **30%** | LibriSpeech 主要是转录，不等于开放语音问答 |
| **Joint SFT（R4-D）** | 3 主任务各 4–8 条 | T2A full **20%** + A2A mini **100%** + A2A full **10%–20%** + COCO **30%** + LLaVA **50%** + 可选 UltraChat **5%** | 这些是不同来源的独立抽样**上限建议**；不得等同最终任务 mixing weights |
| **Text GRPO（R6）** | 2–4 prompt × 2–4 rollout | GSM8K train **10%–20% Pilot** | 任务和奖励 R6 再定 |
| **Audio/Joint GRPO（R6）** | 合成 8-Codebook 片段做数学单测 | 从 T2A/A2A 挑选 **0.5%–2% 的合格 prompts** | 要可验证奖励；不直接把所有监督音频当 RL rollout |

**特别提醒**：`数据源内抽样百分比` 与 `联合训练中的任务采样权重` 是两回事。后者要到 G3/G4 根据不同任务有效 loss token 数量、模态比例和语言退化评估决定，不能简单把上述百分比相加或用作 loss 权重。

### 6.1 Train / Val / Test 划分规则

R3-L 实测：四任务各 8 条视图，共 16 份独立媒体，24/4/4 的本地工程划分；语音三视图共享同一媒体组。旧回答 codes 因无精确编码版本证明被隔离，新声码由真实用户音频经锁定 Mimi 重编码。T2A 为朗读、A2T 为转写、A2A 为显式复述诊断；原始问答式 A2A 正式训练仍需可信目标音频或编码来源。

- 有官方 split 时优先保留；例如 LibriSpeech 的 train/dev/test、COCO 的 train/val，不用重洗乱序代替官方 Test。
- 无可靠官方 split 时，从**去重后的独立媒体组**按 **90% train / 5% val / 5% test** 起步；同一音频/图片/同组对话不得跨 split；需要说话人泛化时增加 speaker-disjoint Test。
- Val 用于早停、调参和选 `best`；Test 仅在锁定 Baseline/方法后用于报告，不反复用 Test 挑方法。
- P4 历史 `64 train / 8 val` 只证明工程闭环，不得当作 B1 质量或论文式 Test。
- COCO 与 LLaVA 可能复用图片；不同数据集来源也需要跨来源去重，而不是各自切分后直接混合。

## 7. R2–R7 核心交付物：**代码就绪**与**研究验收**分离

| 阶段 | 本地代码重点 | 云端与研究产物 |
|---|---|---|
| R2 | Projector/FixedBridge/Talker/Qwen 接线、两条真实 T2A 时间轴、梯度、P5 Streaming 回归 | 不训练模型能力 |
| R3 | 全任务 Dataset/Collator、manifest、CLI、checkpoint 规范、tiny data | G3 CUDA 1–10 / 100 steps、资源 Profile、确认训练配方可行 |
| R4-A | T2A 真实宽度的 CE/Mask/Resume 代码 | 首轮 1.7B+Talker 过拟合，判断梯度链路 |
| R4-B | 正式 T2A 长短句/双语言数据配方 | 随机 Talker 收敛、能否生成可懂语音 |
| R4-C | I2T、A2T 数据及 Projector SFT 代码 | 看图/听音带来可测输出变化 |
| R4-D | 混合采样、loss、LoRA、训练器/恢复 | **B1 联合 SFT Checkpoint + 首轮 Benchmark** |
| R5 | Bridge 多方案/消融代码先 tiny 测 | 基于 B1 的公正多组重训练对比 |
| R6 | GRPO 数学、采样、8 codebook logprob/Mask/Reward 接口先 tiny 测 | SFT 与 Text/音频 GRPO 受控实验，1 GPU 起步、必要时两卡 |
| R7 | Benchmark/统计/系统调优脚本 | 统一正式复测、消融报告、Demo/技术报告与简历事实 |

### 7.1 Baseline、Benchmark、消融分别在哪里？

| 位置 | 做什么 | 对比对象 |
|---|---|---|
| R3/G3 | **系统 Profile**，不是模型 Benchmark | GPU 显存/训练吞吐 |
| R4-A/B/C | 每个任务的训练诊断与阶段评测 | 同任务自己的初步输出 |
| **R4-D 结束当场** | 冻结 **B1 SFT Baseline**、立即执行**首轮统一质量 Benchmark** | B1 自身可复现实测 |
| **R5** | **Adaptive Bridge 架构消融** | 同底座 B1 Fixed vs 其他 Bridge，经训练且同评测条件 |
| **R6** | **Multimodal GRPO 算法消融** | 同一 B1 SFT 起点 vs Text GRPO vs 可验证的 Joint/Audio |
| **R7** | **最终统一 Benchmark + 系统消融** | 全部完成的模型/训练路线 + AMP、Cache、Triton 等按需系统实验 |

**对照不能犯的错**：原版文本 Qwen 不能充当音频生成 Baseline；不同模型规模/Tokenizer/数据量不可把涨点归功于 Bridge/GRPO；只改推理 Bridge 而不重训练不能算公平消融。R5/R6 从同一个 B1 起点独立分支。

## 8. 云 GPU 资源与时长：建议而不是已证实

| 阶段 | Mac 本地 | 云端推荐 | 多卡 |
|---|---|---|---|
| R2 | 主力开发与测试 | 需要真实 1.7B 检查时再用 | 不需要 |
| R3-L / G3 | 数据脚本、fixtures | **1×H100 80GB Profile** | 不需要 |
| R4-A/B | 训练器 tiny 测 | **1×H100 80GB T2A** | 首轮不需要 |
| R4-C/D | Projector/LoRA tiny 测 | **1×H100 80GB** | 首轮不需要 |
| R5 | Bridge 算法单测 | **1×H100 80GB** 多次实验 | 不必要，串行可复现 |
| R6 | 数学单测/离线 rollout fixture | **1×80GB 起；必要时 2 GPU** | 只有实测明显瓶颈时考虑 |
| R7 | 结果分析/图表 | 同规格 GPU 跑最终 Benchmark | 不必要 |

以前讨论的时间大致为 **R4 约 63–210h、R5 约 30–120h、R6 约 20–100h（甚至更多）**，仅供探索预算上限和范围，不是实测。时间必须用 G3 实际 step time、序列长度、训练 epoch/updates、评测与存盘开销重估；若用两 GPU，要分别报 wall time 和 GPU·h。**不需要为简历强制加 DDP/ZeRO。**

## 9. 学习与自主贡献边界：CS336 怎样用到 OmniPlus

| CS336 基础 | 本项目应用 | 要掌握的真正新增能力 |
|---|---|---|
| A1 Basics | Transformer、GQA/RoPE、优化与 LM 接口、Talker 等核心组件 | Hidden 接线、Bridge、Projectors、8 路音频预测 |
| A2 Systems | AMP、KV Cache、Checkpoint、Profile、可选 Triton/DDP | 分块 Mimi/流式输出、长音频显存优化、RL rollout 的系统成本 |
| A3 Scaling（可选） | 训练效率、规模估算 | 多模态计算预算与 token/frame 统计 |
| A4 Data | 数据清洗、去重、数据划分、可追溯版本 | 图片 Caption/QA、语音 Transcript/问答、Mimi Codes/多模态对齐 |
| A5 Alignment | SFT、偏好/GRPO 的概率与目标 | 多模态联合 SFT、可验证语音奖励、8-codebook 策略概率 |

**个人重点**：模型行为与数学定义（Talker、Bridge、Loss、Mask、GRPO）亲自理解和必要时实现。**Codex 重点**：数据处理、配置/CLI、GPU 启动与恢复、调试、测试、实验记录。不要把自研 Projector/训练系统说成“从零训练 Qwen3 语言主干”。

## 10. 最终决策清单

### 已经决定

- Qwen3-1.7B 非 Base，Frozen SigLIP2/SenseVoice/Mimi；首版 Fixed Bridge 与 4×768 Talker/8 Codebooks；P0–P5 只读。
- 本地优先把 R4–R6 训练/实验代码用 Tiny Smoke 跑通，然后云端按 G3→G4→G5/G6→G7 操作。
- 各模块主要在单 GPU 上正式训练；G4-D 立即形成和 Benchmark B1；R5/R6 两条独立研究主线。
- 研究贡献需要**同底座、同数据、同评测**的真实对照；不把预期提升当作已完成成果。

### 暂时不应替用户提前决定

- Talker 随机初始化是否保留（G4B 看收敛）、T2A 正式训练数据比例（G3/G4B）。
- 固定 Bridge 读取层与 Norm 的最终优化选择（R2 先定规范，R5 再消融）。
- LoRA rank/target、联合 SFT 任务采样、正式超参数/时长（R4D Profile）。
- 最终 Adaptive Bridge 组合、RL 奖励/目标/参数更新范围（G5/G6 前）。
- RL 一卡还是两卡、是否 DDP/FSDP/ZeRO（G6 性能/显存测试后）。

**最终原则：本地先验证“我写的代码是对的”；云端再回答“模型学会了吗、创新有收益吗”。**
