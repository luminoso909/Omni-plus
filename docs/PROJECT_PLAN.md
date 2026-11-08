# OmniPlus v2.1｜重构、训练与研究执行规划

> **文档类型**：项目主规划。下文保留 v2.0 的架构、研究设计和验收标准，执行顺序见 v2.1 执行附录。<br>
> **版本 / 日期**：v2.1 / 2026-11-08<br>
> **状态**：R0–R3-L 与 R4-L 本地代码/Tiny 验收已完成。R5-L 等待授权，G3 与正式训练尚未执行。<br>
> **研究定位**：**以 Qwen3-1.7B 为主 Thinker，自主构建并训练轻量级 Thinker–Bridge–Talker Omni 模型；在可靠的多模态 SFT Baseline 上研究 Adaptive Bridge 与 Multimodal GRPO。**  
> **最重要的要求**：**继承 P0–P5、最小改造、阶段验收、严格归因、历史只读。**

本地 `R2 → R3-L → R4-L → R5-L → R6-L` 先验证代码、数学正确性与极小样本闭环。再按 `G3 → G4 → G5/G6 → G7` 开展云端资源测试、训练和实验。每个阶段分别授权、记录与验收。 [v2.1 执行附录](#附录-c-v21-执行顺序与变更记录) · [本地与云端手册](OMNIPLUS_LOCAL_FIRST_CLOUD_PLAYBOOK.md) · [架构与数据决策](OMNIPLUS_DECISIONS_AND_DATA.md)

## 给 Codex 的第一条指令（请先读）

你现在接手的是**已经完成 P0–P5 的现有仓库**，不是空仓库。不要重新初始化，不要从头再做 MiniMind-O，也不要在没有审计仓库的情况下用新代码覆盖旧实现。

1. 首先读取仓库 `README.md`、`AGENTS.md`、`docs/PROJECT_PLAN.md`、`docs/项目待办.md`、两份 v2.1 配套规划及已验收阶段报告，并检查 `omni/`、`configs/`、`scripts/`、`tests/`、`experiments/` 与 Git 状态。
2. **每次只执行用户授权的一阶段**。R4-L 本地代码与 Tiny 验收已完成，后续 R5-L、资源下载、正式训练与云任务分别取得授权。
3. 把 `docs/stages/P0.md`–`P5.md` 当作不可修改的历史证据：**不得编辑、移动、重命名、格式化、覆盖、追加内容或删除**。原始实验记录、权重与成果同样不得覆盖；历史脚本必须继续可复现。
4. 将主线从“MiniMind Dense/MoE 复现 → Qwen 扩容”更新成 **“已有 MiniMind 成果归档 → Qwen3-1.7B 主干迁移 → Omni 对齐/SFT Baseline → Adaptive Bridge / Multimodal GRPO → 评测”**。MiniMind 实现仍保留作历史兼容测试和架构参考，**不再是必须正式训练或并行研究的 Baseline**。
5. 先给出**实际文件级清单**与修改理由；完成授权阶段后执行测试、提交 `docs/stages/R0.md`（后续为 `R1.md`…），更新 `README.md` 与 `docs/项目待办.md`，报告变更与未解决问题；然后停止。
6. 用户负责核心算法的学习、实现与研究决定；Codex 负责被授权的接口包装、工程组织、测试、配置、数据 IO、运行、日志、调试辅助和文档。**不能未经明确授权直接填满用户预留的核心算法类。**

---

## 0. 总览、目标与非目标

### 0.1 项目要回答的三个问题

| 层次 | 研究/工程问题 | 必须展示的证据 |
|---|---|---|
| **模型构建** | 预训练 Qwen 如何经可训练的多模态组件获得“看、听、说”能力？ | 可训练参数、跨模态对齐与联合 SFT、可工作的文本/视觉/音频 Demo、任务评测 |
| **架构研究** | Thinker 不同层的表示如何影响 Talker？自适应 Bridge 是否有可测收益？ | 固定层 vs 其他桥接方案的**同底座**受控实验与成本对照 |
| **后训练研究** | 多模态策略优化能否改善文本与语音输出？ | SFT vs 不同 GRPO 方案，正确的联合概率/掩码/奖励定义与稳定性记录 |

支撑上述目标的系统能力：AMP、KV Cache、流式 Mimi、Profiler 与按需的 Triton / DDP。**这些技术只有在真实实现和 Benchmark 后才能写为成果。**

### 0.2 已确定的决定（Codex 不得擅自改动）

- 主 Thinker：**官方 Qwen3-1.7B 非 Base（已后训练的版本）**；官方 Qwen3-0.6B 仅用于轻量接口调试。**无需从零预训练 Qwen 语言模型，也无需先额外纯文本 SFT。**
- 初始 Talker：**参考 MiniMind-O 的 4 层、隐藏维 768、8 路 Mimi Codebook 及原始 Head/Embedding 设计**；第一版先求正确、可训练、可测，暂不扩大 Talker 或改变 Delay Pattern。
- 冻结权重：SigLIP2 Vision Encoder、SenseVoice Encoder、Mimi Encoder/Decoder；Speaker Encoder 如采用 CAM++ 则冻结。由调用方显式注入，不在 Omni 构造器中静默下载。
- 可训练模块：Vision/Audio Projector、Fixed Bridge 投影、Talker/Audio Embedding/8 路 Heads；后续联合 SFT 视需要对 Qwen 注入 **LoRA**。
- **SFT 必做的是跨模态对齐与联合监督训练**，而不是重新训练语言预训练模型；GRPO 从已验证的 Omni SFT Checkpoint 出发。
- 单 GPU 优先；GRPO 是否需要更大显存/多卡，以及奖励、Rollout、训练超参数，**到 R6 才定**。
- Adaptive Bridge：目前**只保留可替换接口**。待 R4 Baseline 成立后，在 R5 选择 Fixed/Last/Weighted/Gated 等若干方案并设计公平消融，**不预判收益**。
- MiniMind-O：保留源码实现、历史严格权重对照和历史脚本，但**不强制再训练 MiniMind Full，不作为 Qwen 研究的同底座对照**。
- 优先级：Omni Baseline（最高）→ Multimodal GRPO 与 Adaptive Bridge（两条研究线）→ 视觉 Token、更多 Backbone、复杂 MoE/分布式等（按需）。

### 0.3 不做什么

不重新写 SigLIP2 / SenseVoice / Mimi，不从零预训练 Qwen3-1.7B，不在 R0–R4 预实现复杂 Adaptive Bridge/GRPO，不强制做 DDP/ZeRO，不强制加入 Qwen3.5、不盲目追求“所有多模态组合”而忽略核心 T2A 链路。

### 0.4 v2.0 阶段关系与 v2.1 执行顺序

下图保留 v2.0 的技术依赖。v2.1 将 R3–R6 的本地实现与正式训练拆开，云端工作仍遵守图中依赖。 [执行附录](#附录-c-v21-执行顺序与变更记录)

```text
已完成历史 P0–P5 [只读保留，含旧 MiniMind 测试与 Streaming]
          │
          ▼
R0  审计现有仓库 + 更新项目文档（已验收）
          │
R1  Qwen3-1.7B 独立 Thinker Backend + Tokenizer / Hidden States / Cache
          │
R2  Projector + Fixed Bridge + Talker + Codebook + Stream 接口适配
          │
R3  数据、权重、冻结策略、单卡实验配方与数据验证
          │
R4  T2A → 视觉/音频对齐 → Omni 联合 SFT → 可测的 Qwen Omni Baseline
          │
          ├───────────────────────┐
          ▼                       ▼
R5  Adaptive Bridge 消融       R6  Multimodal GRPO（顺序由用户届时决定）
          └───────────┬───────────┘
                      ▼
R7  统一评测、系统优化、论文式报告与公开 Demo
```

**阶段互斥约束**：R1/R2 验收接口。R5-L/R6-L 可在各自获授权后验证通用代码与极小样本，正式 G5/G6 须等待 G4D 的 B1 验收。每完成一阶段等待用户授权下一阶段。

---

## 1. 已完成成果与必须继承的工程事实

> 此节基于上传的 P0–P5 报告，代表**报告时点**的状态；Codex 应用 `git status`、实际路径及测试确认当前仓库状态。**不得把计划、Smoke Test 和已验证的真实能力混为一谈。**

| 历史阶段 | 已有工作与证据 | v2.0 处理 |
|---|---|---|
| **P0** | Python 3.12、PyTorch / Transformers、`omni` 包、Ruff/pytest/CI、Git 工程；`model_lm.py` 已更名 `model_thinker.py` | **保留工程与环境锁**；无需重新初始化 |
| **P1** | 自研 MiniMind Dense/MoE：真实权重严格加载、层输出/logits/部分梯度对照、Cache、HF 回环；官方 **Qwen3-0.6B** 已在独立 HF 路径完成文本加载与推理 | **保留两种已有检验**；Qwen3-1.7B 尚未集成 Omni，不要称已完成 |
| **P2** | 冻结 SenseVoice 输出 `[L,512]`、SigLIP2 输出 `[64,768]`；真实混合 batch、padding、索引、冻结生命周期通过 | 保留 encoder、混合 batch 索引和 Feature Cache 的实现；Projector 输出改为 Qwen 隐藏维 |
| **P3** | Mimi 8-codebook codec-only 重建、Delay / STOP / PAD 边界、原版 logits 对照、可播放 WAV；兼容旧 SFT 权重 | **尽量不动 Codec 与 Delay 核心**，仅在新配置/接口周围适配 |
| **P4** | 64 train / 8 val，单设备 SFT、Loss、梯度累积、恢复、真实任务 smoke；**约 150 万参数小模型**在 600 步使训练 Loss 降低 66.46% | **继承训练框架**；此结果只证明小模型工程闭环，**不是 1.7B Omni 训练或泛化** |
| **P5** | MiniMind Omni 的 KV Cache、`stream_generate`、Mimi PCM Chunk、左 padding、自然停止/截断/取消、性能基准；主机 **151 项测试通过** | **保护流式实现与指标口径**；新 Qwen Omni 要独立通过回归，不能覆盖 P5 结果 |

历史文件定位依据（以真实仓库为准）：`omni/models/thinker/model_thinker.py`、`omni/models/model_omni.py`、`omni/models/encoders/`、`omni/data/audio_alignment.py`、`omni/training/{sft,losses}.py`、`omni/inference/{codec,streaming}.py`、`scripts/check_p{2,3,4,5}.py`、`scripts/train_sft.py`、`configs/assets.yaml`、`experiments/p{1..5}/`。**不得仅凭此段猜测所有路径和签名，R0 必须实际检查。**

### 1.1 资产与历史冻结线

- 旧仓库 `../minimind-o` **只读**；本仓库历史 MiniMind 模型与测试**保留可运行**，不主动删代码、权重、Tokenizer、实验记录。
- `docs/stages/P0.md`–`P5.md` **完全不可变**（包括自动格式化、链接改写）；允许在新 README 或 `docs/stages/R*.md` **引用**它们，不允许回写历史。
- 保留 P4 的 `64/8` 工程子集与原始 Checkpoint；R3 新建**独立正式数据 split**，不要把 8 条验证样本当作正式评测。
- 现有冻结特征缓存、Mimi Codes 和 Qwen3-0.6B 资产均有版本/哈希。更换 encoder、codec 或 tokenizer 后须**重新生成受影响缓存**，不能混用不同版本。
- R0 对工作区未提交更改先记录并保护；不能 `git reset --hard`、`git clean -fd`、自动覆盖未跟踪文件。

---

## 2. v2.0 目标架构与参数所有权

### 2.1 结构图

```text
Text → Qwen3 Tokenizer ──────────────────────────────────────────────┐
Image → SigLIP2 Vision (frozen) → Vision Projector (trainable) ───────┤
Audio → SenseVoice (frozen) → Audio Projector (trainable) ────────────┤
                                                                      ▼
                                                          Qwen3-1.7B Thinker
                                                         pretrained; first frozen
                                                         later selected LoRA SFT
                                                                │       │
                                                        Text logits    Hidden States
                                                                │       │
                                                           Text Output  ▼
                                                                Fixed Bridge
                                                             2048 → 768
                                                                │
History Mimi codes → Talker Embedding ─────────────────────────────────┤
Speaker embedding (optional/frozen encoder → trainable projection) ────┤
                                                                ▼
                                                     Talker: 4 Transformer layers
                                                        hidden=768; trainable
                                                                │
                                                     8-codebook Audio Heads
                                                       vocab=2112 per head
                                                                │
                                                 Mimi Decoder (frozen; 24kHz)
                                                                ▼
                                                       Chunk PCM / WAV

训练时：目标 WAV → Mimi Encoder（冻结）→ 8 路 codes → 延迟布局的监督标签。
```

**注**：Qwen3-1.7B `hidden_size=2048` 以官方 `config.json` 为准，实际层数、头数、KV Cache 细节在 R1 程序读取并保存到 manifest；Talker 768 来自 MiniMind-O 初始设计，不等于 Qwen 隐藏维。**切勿盲目把 Talker 扩至 2048。**

### 2.2 参数、初始化、梯度策略

| 模块 | 初始来源 | R1–R3 | R4 Baseline | R5/R6 |
|---|---|---|---|---|
| Qwen3-1.7B Thinker + LM Head | 官方已后训练权重 | 冻结 / eval 或 train 的明确约束 | 先冻结；联合多模态 SFT **按需训练 LoRA** | 在固定 SFT 起点上按实验决定是否更新 LoRA |
| SigLIP2 Vision | 官方或已验证本地权重 | 冻结、eval | 冻结 | 冻结 |
| SenseVoice | 官方或已验证本地权重 | 冻结、eval | 冻结 | 冻结 |
| Mimi Encoder / Decoder | 与 8 codebooks 匹配的已验证权重 | 冻结、eval | 冻结 | 冻结 |
| Vision Projector（768→2048） | 随机初始化；若保留旧层仅作结构迁移 | Shape/Grad Smoke | 训练 | 依实验冻结或训练 |
| Audio Projector（512→2048） | 随机初始化；若保留旧层仅作结构迁移 | Shape/Grad Smoke | 训练 | 依实验冻结或训练 |
| Fixed Bridge（2048→768） | 随机初始化；允许最小 Linear/MLP | Shape/Grad Smoke | 与 Talker 一同训练 | R5 同底座受控对比 |
| Talker / Audio Embedding / 8 Heads | 首选随机初始化并沿用 4×768 结构 | Shape/Grad Smoke | **正式训练**，先 T2A 后联合 | R5/R6 由实验方案决定 |
| Speaker Projection | 可选随机初始化；CAM++ 如使用则冻结 | 非首版阻塞项 | 视数据需要启用 | 同一设置固定 |

> **注意“冻结与 LoRA”**：`LoRA` 是参数化方法，`SFT` 是训练阶段。优先使用官方**非 Base** Qwen3-1.7B，先训练新模块，再在多模态 SFT 中启用 Qwen LoRA；GRPO 从已验证的 SFT Checkpoint（包含 LoRA 与新模块权重）继续，**不是从原始 Base 权重直接 RL**。

### 2.3 Talker 的首版固定选择

- 4 Transformer Blocks / hidden 768；8 路 Mimi codebooks；每路 logits 的词表容量 **2112（含保留特殊 ID）**。
- 使用既有 `TalkerEmbedding`、`TalkerHead`（共享 Base + 分路 Adapter、原 rank 如 256）、`codec_proj`、原始 8 路 Delay Pattern / STOP 语义。原则是**沿用结构而不是照搬 MiniMind SFT 权重**。
- **避免两次 2048→768 投影**：R2 审计已有 `talker.embed_proj` 与 `Bridge` 的边界，明确哪一个拥有投影及归一化，不可默默在二者中重复变换。建议 `Bridge` 统一接收 `[B,T,2048]`，输出 `[B,T,768]`，Talker 只接收规范后的 768 维条件；如为兼容旧键需其他内部安排，须写接口与状态映射表并测试。
- **首选随机初始化 Talker**，先做单步反传/小样本过拟合 → 短语音生成 → 质量评估。若成本/收敛不合理，**提交证据并请用户选择**是否回退到 MiniMind-O Talker 初始化。更换初始化必须单独记录实验，不可把预训练 Talker 称为从零训练。
- 不在首版改成声码器训练、音频交叉注意力、不同码本数量或动态文本/语音对齐策略。

### 2.4 必须保留的接口契约

在 R0/R1 根据真实代码敲定签名，建议统一为：

- `ThinkerBackend`：获取 `inputs_embeds`、`input_ids`/Tokenizer、前向 `attention_mask`、`position_ids/cache_position`、`past_key_values`、按需返回所选 `hidden_states`、文本 `logits`、更新后的 cache；Backend 不在构造时偷偷修改模型目录、Token ID 或下载权重。
- `Projector`：输入 `[B,N,D_in]`、有效长度/Mask，输出 `[B,N,D_qwen]`；首版不改变视觉 Token 数（64）。
- `Bridge`：输入约定的隐藏层特征（固定层或多层）、mask，输出 `[B,T,768]`；未来更换算法只替换 Bridge，不改 Talker 主接口。
- `Talker`：输入音文条件、历史声码及 speaker 条件，输出 `[B,8,T,2112]`（或文档声明的等价逐码本形式）以及 cache。
- `CodebookAlignment`：**唯一负责**延迟布局、Stop/Pad、有效 mask、shift/recover；训练/推理对称，禁止分散魔法数字。
- `OmniPlusOmni` 或新 wrapper：负责组合路由；不得把编码器下载、数据采集、ASR 评估塞入 `__init__`。

---

## 3. 现有代码迁移设计（最重要的工程约束）

### 3.1 保留 / 修改 / 新增的指导清单

| 既有代码/文件类型 | 默认动作 | 原因与操作约束 |
|---|---|---|
| `omni/models/thinker/model_thinker.py` | **保留旧 MiniMind 实现** | 现有严格权重 parity 的依据；不强制改成 Qwen 类 |
| **新 Qwen Backend**（路径由 R0 审计确定） | **新增** | 官方 HF `Qwen3ForCausalLM` 经薄适配层接入，避免逐键伪映射 MiniMind 权重 |
| `omni/models/model_omni.py` | **最小改造或并行 wrapper** | 解耦硬编码 MiniMind 词表/维度、固定层；保留旧入口及回归测试 |
| `omni/models/encoders/` | **尽量原样保留** | P2 真实权重、冻结和长度通过；新项目复用这些已验证输出 |
| Projector / Bridge | **在原模块上抽离参数化接口** | `512/768→2048` 与 `2048→768`；可替换 Bridge |
| Talker / `audio_alignment.py` | **保留算法，必要处适配配置** | 改 Thinker 不应破坏 8 路 Delay / STOP |
| `omni/data/`、`scripts/prepare_p4.py` | **复用工具，新增 Qwen 数据版本** | Tokenizer、Chat Template、占位符、代码标签时间轴必须重建 |
| `omni/training/{sft,losses}.py`、`scripts/train_sft.py` | **参数化并增添新入口/配置** | 可训练参数/optimizer 分组/LoRA/checkpoint 新格式；保留 P4 复现路径 |
| `omni/inference/{codec,streaming}.py` | **保护并最小适配** | 现有 PCM streaming、取消、尾块、Cache/质量数值回归 |
| `configs/` | **新增 v2 分支配置，旧配置不删除** | 区分旧 MiniMind 权重与 Qwen 1.7B、数据版本 |
| `tests/` | **新增 R* 测试，不弱化已有断言** | 新旧两套 smoke 都可运行；重合模块严守已验证边界 |
| `experiments/`, `outputs/` | **新建 R* run** | 每次实验可追溯、绝不覆盖 P* 原记录 |

**原则**：以上为“候选文件级映射”，不是断言仓库文件一定完全相同。R0 必须 `find`/`rg`/阅读源代码，产出“实际路径—现有类/函数—兼容/修改/新增—测试归属”的清单后再执行。

### 3.2 Qwen Tokenizer：最容易失败的兼容点

1. 使用**官方 Qwen3-1.7B 对应 tokenizer + chat template**；不可继续传 MiniMind 6400 词表 ID、BOS/EOS/PAD 约定，也不得只替换 `vocab/merges`。
2. **新图像/音频占位符**必须明确“显示 Token / 实际特殊 Token ID / 被覆盖的 embedding positions”的对应关系。优先沿用当前模型可测的数据协议，但需把独立的 Qwen 模板与占位符映射封装为一处。
3. 若使用 `tokenizer.add_special_tokens(...)` + `resize_token_embeddings(...)`：新增 embedding/head 行如何初始化、是否训练、在 PEFT checkpoint 中如何保存与恢复必须单测；冻结 Qwen 且新增 Token 行也冻结会导致模态标记训练不足。不得把正常文本里出现的字符串碰撞当作模态输入标记。
4. **不要把 Mimi Codebook Stop（如 2050）误当成 Qwen 文本 EOS**。Qwen EOS、思考模板边界与声码 STOP 属于不同词表/状态机。按模板本身而不是写死 `</think>` ID 或 MiniMind `think_end_ids`。
5. Qwen 词表与分词长度改变后，必须基于**原始文本和原始/版本一致的 Mimi Codes**重建联合标签及有效 mask；只换 Tokenizer、不重建标签时间轴的实验全部无效。

### 3.3 Qwen Hidden States、Bridge 与 Cache

- 按官方 Transformers 输出格式显式选层。**HF `hidden_states` 往往包括 embedding output（index 0）和各层输出**，因此“第 N 层”绝不能与数组 `hidden_states[N-1]` 混淆；读实际版本定义并加单测。
- 初始 `FixedBridge` 先选**固定配置层**（可先使用中间层，具体索引由 R1/R2 测试确定），选择依据/输出阶段（block 前后 norm）须记录，不能暗示它已经最优。
- 直接对照同一 Qwen3 权重的 **HF 原生前向 vs Backend** 的 logits、selected hidden、prefill / 单步 decode、padding、位置、KV Cache。引用 P1 的 MiniMind 权重对照只能证明旧实现正确，**不能证明 Qwen Backend 正确**。
- `past_key_values` 的类型/Cache 对象、`cache_position`、动态长度、左 padding、Thinker/Talker 独立缓存必须正确，不能按旧 MiniMind tuple 格式随意拼接 Qwen Cache。

### 3.4 联合文本—音频时间轴与 Streaming

- **文本 Token 不等于 12.5 Hz 的 Mimi 音频帧**。联合训练中虽然可能使用同一计算序列轴，T2A 映射并非字/词 Token 与物理声帧一一对应。R2 必须画出至少 2 条真实 T2A 样本的 Token/Code/Delay/Mask 表及张量形状。
- 真实 WAV → **已验证同版本 Mimi Encoder** → `[8,F]` code IDs → 在 `CodebookAlignment` 中生成 delayed `[8,T]`、STOP/PAD 与 loss mask；`shift→restore` **必须精确可逆**。
- 不把未来的答案音频作为当前输入；teacher forcing 仅提供因果允许的历史 codes。测试输入文本短于声码、长于声码、提前 EOS、Stop 不齐、预算截断、混合 batch 的完整性。
- Streaming 新 Qwen 生成应继续走旧 `MimiStreamingDecoder` 的已验证 chunk PCM 逻辑；Thinker/ Talker/ Mimi 三种 Cache 各有生命周期，流结束、取消、下一话轮时不串状态。

### 3.5 SFT 与 Checkpoint 迁移

- 检查旧 `losses.py` 的音频/文本损失、8 路码本归一化、Stop 加权和梯度累积；首版保留旧数学定义作为可测基线，改变时以新配置单独记录。
- optimizer 仅包含 `requires_grad=True` 的参数，列出各模块可训练/冻结数量；冻结编码器在 `.train()` 下仍 `eval()`。**特别注意：训练 Vision/Audio Projector 时，即使 Qwen 全部权重冻结，仍须允许梯度经过 Qwen 的前向计算图回传至 Projector；不能用 `torch.no_grad()` 包住这一段 Qwen forward。** 只有完全不需要回传到上游可训练输入的纯推理/预计算场景才可使用 `no_grad()`。
- LoRA 的目标层根据 **Qwen 真正 module names** 审计确定，不照抄 MiniMind 名称。保存 LoRA adapter、Projector/Bridge/Talker 新权重与必要的新增 Token embedding，并能严格恢复。
- Checkpoint **分离来源**：旧 `sft_omni_768.pth` 只能按历史架构复现；新 1.7B 的 run 记录基础模型 ID + revision、LoRA 和新模块权重，禁止通过 `strict=False` 静默漏载。
- Resume 恢复 optimizer、scaler、RNG、数据游标、冻结分组、模型/数据 revision；不同 Tokenizer/Audio Codec 的检查点不得混载。

### 3.6 v2.0 文档更新记录与不可变历史

| 文档/目录 | Codex 必须采取的行为 |
|---|---|
| `README.md` | 更新为 **Qwen3-1.7B OmniPlus v2 目标**；将 P0–P5 的已完成事实独立放入“Historical validated work”；把 Qwen Omni、正式 SFT、Bridge、GRPO 清楚标为“planned / not yet validated”；写当前第一个执行阶段 R0 |
| `docs/PROJECT_PLAN.md` | 更新为本文件的 v2.0 规划；若仓库已有 v1，先在 `docs/archive/` 保存只读副本并记录哈希，**不改 P0–P5** |
| `docs/项目待办.md` | **保留历史勾选事实**，把旧 P6/P7/P8/P9 计划标为“superseded”，新增 R0–R7 待办并链接新阶段文档；旧未完成项不是自动已完成 |
| `AGENTS.md` | 强制阶段边界、文件保护、下载/训练权限、算法分工和测试证据；“每次仅当前阶段，用户允许才前进” |
| `configs/` | 增加 v2 model/data/train/eval 配置；不删除已有 p4/P5 配置 |
| `docs/stages/P0.md`–`P5.md` | **绝对不可修改/移动/格式化/覆盖/追加/删除** |
| `docs/stages/R0.md`–`R7.md` | **新建** v2 阶段成果报告；如已有则按实际结果更新本阶段报告 |
| 其他 `docs/learning/` 与 `docs/reports/` | 不批量重写；可新增新报告或链接，历史结论保持原样 |

README 的状态表示必须使用 `✅ 已验收（旧 MiniMind 或现有工程）`、`🛠 正在迁移`、`🧪 待实测` 三种明确标签；**不能把 Qwen3-0.6B 已跑文本推理表述成 Qwen Omni 已生成语音。**

---

## 4. 权重获取、版本固定与环境

### 4.1 权重表与适配要求

| 模型 | 官方 Repo | 默认目录建议 | 使用方式 |
|---|---|---|---|
| **Qwen3-1.7B** | `Qwen/Qwen3-1.7B` | `weights/qwen3-1.7b/` | 非 Base 官方权重；主 Thinker，先冻结，后 LoRA |
| Qwen3-0.6B | `Qwen/Qwen3-0.6B` | **已存在** `weights/qwen3-0.6b/` | 旧 P1 已验收；轻量接口测试，避免重复下载 |
| SigLIP2 | `google/siglip2-base-patch32-256` | `weights/siglip2-p32-256/` 或沿用原本地路径 | Frozen Vision Encoder；**不等同于**现有裁剪的 `SiglipVisionModel` 权重目录，需加载验证 |
| SenseVoiceSmall | `FunAudioLLM/SenseVoiceSmall` | `weights/sensevoice-small/` 或沿用原本地路径 | Frozen Speech Encoder；需验证 FunASR `encoder`/frontend 的实际格式 |
| Mimi | `kyutai/mimi` | `weights/mimi/` 或沿用原本地路径 | Frozen Codec；确认 8-Codebook 配置/采样率/版本，不能盲目复用旧 Codes |
| 可选 CAM++ | `jingyaogong/campplus` | `weights/campplus/` | 说话人条件将来需要时再下载；首版非阻塞 |

### 4.2 终端下载命令（**由用户授权阶段执行，不是 R0 自动执行**）

下面为标准 Hugging Face CLI 命令，**不会在本规划生成时执行**。实际仓库使用 `configs/assets.yaml` 管理本地绝对/相对路径，正式运行必须离线读取。

```bash
# 现有项目已使用的兼容版本；R0 不要求升级环境
python -m pip install 'huggingface_hub>=0.36.2,<1'

# R1 主 Backbone：仅在用户授权 R1 时获取；下载后写入固定 revision + 文件 hash
hf download Qwen/Qwen3-1.7B --local-dir weights/qwen3-1.7b

# 0.6B 如果现有目录已完整，禁止重复下载/覆盖
hf download Qwen/Qwen3-0.6B --local-dir weights/qwen3-0.6b

# 以下优先复用 P2/P3 已验证本地资产；确需更换时再下载并重新验收
hf download google/siglip2-base-patch32-256 --local-dir weights/siglip2-p32-256
hf download FunAudioLLM/SenseVoiceSmall --local-dir weights/sensevoice-small
hf download kyutai/mimi --local-dir weights/mimi

# 可选：只有启用独立 speaker encoder 时执行
hf download jingyaogong/campplus --local-dir weights/campplus
```

**版本锁定程序**：首次批准下载时，先读取 Repo 的 commit SHA 并写入 `configs/assets.yaml` / `experiments/assets-manifest.json`；随后重现下载使用 `hf download REPO --revision <COMMIT_SHA> --local-dir DIR`。记录 `repo_id、revision、文件 sha256、license、加载类、dtype、codec codebook 数/采样率、tokenizer revision、预处理器版本`。对非 HF 的源也记录来源与转换过程。***不得因为下载了官方原始权重就默认为旧版裁剪/FP16 转换权重的数值等价。***

权重核验与许可证参照：

- Qwen：<https://huggingface.co/Qwen/Qwen3-1.7B>；0.6B：<https://huggingface.co/Qwen/Qwen3-0.6B>
- SigLIP2：<https://huggingface.co/google/siglip2-base-patch32-256>
- SenseVoice：<https://huggingface.co/FunAudioLLM/SenseVoiceSmall>
- Mimi：<https://huggingface.co/kyutai/mimi>
- CAM++（可选）：<https://huggingface.co/jingyaogong/campplus>
- MiniMind-O（架构及历史数据）：<https://github.com/jingyaogong/minimind-o>

### 4.3 环境与计算资源

- **继承已验证的本地环境**：macOS arm64 / Python 3.12.14 / PyTorch 2.15.0 / Transformers 4.57.6（历史报告快照）；R0 不自动升级、重新 lock 或安装 GPU 依赖。
- Mac 24GB：适合 unit/shape/mask、Qwen 小 batch inference、0.6B 接口调试、短样本 smoke；**不以它能够稳定完成 1.7B 联合训练为前提**。
- 正式 1.7B/Talker SFT：**单张 NVIDIA GPU 起步**。先按实际 dtype、序列长度、音频帧、梯度累积、LoRA 状态进行 10/100-step Profile，测量显存、吞吐和预计训练成本；容量不足时先缩短样本、预计算冻结特征、冻结更多模块或调小 microbatch，而不是强行上 DDP。
- AMP：CUDA BF16 优先（以设备支持为准），MPS 按实际兼容性测。DDP/ZeRO **不强制**；GRPO 是否需要多卡留待 R6 实测。
- 不伪造时间/费用估计，所有成本预测附运行环境与测量依据。

---

## 5. 数据集与数据工程

### 5.1 按“训练哪部分参数”划分数据

| 类型/阶段 | 优先备选数据 | 训练模块/目标 | 强制提醒 |
|---|---|---|---|
| **LM Pretraining**（原则上不做） | FineWeb-Edu 等 | 仅用于可选教育性 LM 实验，**不是必需** | Qwen 已有预训练语言能力；不为此消耗正式训练预算 |
| **T2A 声学建模**（首要） | **MiniMind-O T2A**、LJSpeech；扩展 AISHELL-3、LibriTTS-R | Fixed Bridge + Talker + Audio Heads；监督 Mimi Codes | 先跑短句/单说话人；原始 WAV 需同版 Mimi 编码；英文 mini 不等于中文能力 |
| **I2T 视觉对齐** | COCO Captions、LLaVA-Instruct-150K；后续 TextVQA | Vision Projector + 按需 Qwen LoRA；文本 CE | LLaVA 的图像路径需配实际图像；64 Tokens 不保证 OCR 能力 |
| **语音输入/转录对齐** | LibriSpeech、可用的中文转录数据 | Audio Projector + 按需 Qwen LoRA；ASR/转录文本 CE | ASR 训练 ≠ 开放式语音问答 |
| **A2A 语音交互** | MiniMind-O A2A；VoiceAssistant-400K（审计后可选） | Audio Projector + Thinker/LoRA + Bridge/Talker；回答文本/语音监督 | VoiceAssistant 的回答是 **SNAC**，不是 Mimi；必须取得/还原合法波形并重新 Mimi 编码 |
| **联合 Omni SFT** | 经过版本固定和去重的 T2A + I2T + A2A + 可选纯文本指令 | Projectors / Bridge / Talker + Qwen LoRA | 任务混合比例与梯度/损失权重需实测；保留文本能力基准 |
| **GRPO（R6 再定）** | GSM8K、TextVQA、经筛选的 T2A/A2A 或自建可验证语音任务 | 选定策略模块 | 先保留候选；不在 R0–R4 定死奖励或直接下载整套 |

参考入口（仅候选，**正式使用前逐一复核许可、字段与下载大小**）：

- MiniMind-O 数据：<https://huggingface.co/datasets/jingyaogong/minimind-o_dataset>
- LJSpeech：<https://keithito.com/LJ-Speech-Dataset/>；AISHELL-3：<https://www.openslr.org/93>；LibriSpeech：<https://www.openslr.org/12>
- COCO：<https://cocodataset.org/>；LLaVA：<https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K>
- VoiceAssistant-400K：<https://huggingface.co/datasets/gpt-omni/VoiceAssistant-400K>
- GSM8K：<https://huggingface.co/datasets/openai/gsm8k>；TextVQA：<https://huggingface.co/datasets/facebook/textvqa>

### 5.2 统一样本协议（应在 R3 固定）

```yaml
sample_id: string
source: repo/path + version + original_id
split: train|validation|test
task: T2A|I2T|A2T|A2A|TEXT
text_prompt_raw: string|null
text_answer_raw: string|null
image_path: string|null
input_audio_path: string|null
input_audio_sr: integer|null
target_audio_path: string|null
target_audio_sr: integer|null
speaker_id: string|null
speaker_condition_path: string|null
mimi_codes_path: string|null
mimi_model_revision: string|null
tokenizer_revision: string
license: string
quality_flags: [string]
```

**禁止**把 MiniMind Token IDs 原样当作 Qwen IDs。对于旧数据若仅有预编码 Codes 而无 WAV，必须先确认这些 Codes 采用**同一 Mimi 版本、8 路码本及特殊 Token 定义**；若不能确认，则仅作隔离实验/暂不纳入主训练。不同任务必须有明确的 loss mask，A2A 的**输入语音**绝不可混进**回答目标码本**。

### 5.3 数据处理顺序与质量门槛

1. **R3 先做 manifest / license / schema / 样本抽检**，不得自动下载数十或数百 GB 全量数据。
2. 首轮 T2A 从已有 MiniMind-O T2A 小样本/独立 LJSpeech 组成小规模 Pilot（例如数百到数千条，具体受资源与许可约束）；建立 Codec 一致性与音频长度、文字内容、语言、说话人统计。
3. 对语音任务：保存 16k 输入语音（SenseVoice）与 24k Mimi 编码路径的**不同采样率流程**，不要把它们混为一条处理链；重采样后重新生成相应特征/标签。
4. 对图像任务：检查真实媒体、caption、问答是否匹配，排除损坏图像/空标签。
5. **统一跨来源去重**：音频指纹/源音频 ID、图像 hash、文本归一化与近重复；同一个底层媒体不能分散到 train/test。
6. 固定独立 train/val/test、随机种子、revision、媒体校验哈希；P4 原 `64/8` 工程样本不能作为正式 test。
7. 在冻结 Encoder 上**允许离线缓存特征**（包含 encoder revision / dtype / processor 版本），但一旦变更增强或模型版本则缓存失效。
8. 小样本过拟合与正式质量评测分开。正式样本规模和训练预算由 R3 Profile 后决定，Codex 不自行拍脑袋固定“全量”。

---

## 6. 从 Baseline 到 SFT 的训练方案

### 6.1 阶段与可训练参数

| 训练子阶段（在 R4 内） | 模型/梯度范围 | 数据 | 通过标准 |
|---|---|---|---|
| **R4-A T2A 训练闭环** | Qwen 冻结；**Fixed Bridge + Talker + Audio Embedding/Heads** 训练 | T2A 小样本 → 合规的较大训练子集 | 真实全尺寸模型单步前后向/恢复、训练损失可下降；短语音可生成并通过结构测试 |
| **R4-B 语音质量成型** | 同上；必要时调整训练日程，不擅改架构 | T2A 扩充、英文/中文按语言分层 | CER/WER、生成失败率、人工试听，记录成功与失败；若无可懂语音，暂停并分析 |
| **R4-C 图像/语音输入对齐** | Qwen 先冻结；**Vision/Audio Projector** 训练 | I2T、A2T、初步 A2A | 真正的模态条件有效性：替换输入媒体应改变合理回答，独立评测高于对照，不仅 loss 有限 |
| **R4-D 多模态联合 SFT** | Projectors/Bridge/Talker + **Qwen LoRA（可开关）**；外部 Encoder/Codec 冻结 | T2A + I2T + A2A + 必要文本保留任务 | 单卡可重复训练、可恢复、跨任务评测、文本能力退化检查、可用 Checkpoint |

**特别说明**：R4 允许在证据充足时调换 R4-B/R4-C 的执行顺序；更重要的是持续保持可测的训练与推理接口。SFT 的文本目标和音频目标可用原 P4 定义开始：

\[
\mathcal L_{\text{SFT}}=\mathcal L_{\text{text}}+\lambda_a\mathcal L_{\text{audio}}+\lambda_{\text{aux}}\mathcal L_{\text{aux}}.
\]

最初的 `λ_a`、Stop 权重沿用 P4 定义用于 parity，随后依据实际梯度与任务性能调整；**禁止同时改多个因素却将收益归于某一项**。Prompt 的 Chat Template、采样温度、Thinking Mode、目标数据格式均记录在训练配置中。

### 6.2 必须做的训练验证

- 确认冻结参数 `grad=None`（冻结编码器之外，Qwen 是否冻结取决于当前阶段）；新模块梯度有限且非全零。
- `batch=1` 与变长混合 batch 的 mask/损失一致；音频 8 路标签、Stop 权重、padding、zero-valid 的异常行为有单测。
- **真实 1.7B + 目标 Talker** 的短样本过拟合测试必须独立重做；**不能引用 P4 的 150 万参数结果代替**。
- 训练/推理对齐：teacher-forced 目标时间轴与自回归生成步一致；短长文本、长声码、提前 EOS、生成耗尽均验收。
- 记录每轮 optimizer/scheduler、冻结分组、可训练参数量、随机种子、设备、dtype、loss curve、samples/s、峰值显存、音质与错误样本。
- 验证 `save→resume` 后模型、optimizer、scaler（若有）、RNG、LoRA 与新模块权重/配置正确恢复；恢复缺任何一组权重明确报错。

### 6.3 基线定义

- **B0（文本检查）**：原始 Qwen3-1.7B，测语言与指令能力；**不具备可比较的语音生成能力**。
- **B1（核心 Omni Baseline）**：相同官方 Qwen Backbone + 固定层 Bridge + 既定 Talker + 已完成多模态 SFT，取得可用的文本/视觉/语音功能与正式评测。
- 研究对照在 **B1 的同底座同数据**上开展；不要求再完整训练一套 MiniMind 模型，也不能“Qwen Omni 比 MiniMind 质量好”来证明 Bridge/GRPO 的贡献。

---

## 7. Adaptive Bridge：后置的架构研究

**当前阶段只需实现/固定接口**：`Bridge.forward(selected_hidden_states, attention_mask, ...) -> TalkerCondition`，并在配置中记录所选层、规范化及投影。

R4 后进入 R5 时，再选择最有解释力的若干候选：Fixed、Last、简单 Weighted、Gated 等。研究依据是不同层表征有差异，但**“多层融合一定更好”尚未得到验证**。

R5 设计必须包含：

- 同一 Qwen checkpoint、相同数据 split、近似训练预算、相同 Talker 初始化策略与采样/评估；每个 Bridge 需要经过相应训练，不能仅换推理读取层号。
- 记录 Layer 选择、额外参数量、收敛、CER/WER、UTMOS（如可用）、文本正确性、TTFA/吞吐/显存。
- 至少先完成三组有意义的单项对照，再决定是否加入 Gated；多次种子视预算决定并公开局限。
- 允许负结果；如果 Fixed 已足够好，无需为了简历“创新”强行加入门控。

相关研究背景可参考 ELMo 的任务相关层混合、SUPERB 的多层语音表示聚合；**它们只是研究动机，不是本项目已有效果的证据**。

---

## 8. Multimodal GRPO：后置的 RL 研究

**R0–R4 不制定最终 Reward/GRPO 训练配方，也不自动实现 RL 核心。** R6 由用户结合 B1 Baseline 的质量、显存和评测能力确定任务与策略。

预留设计约束：

1. 训练从同一 **SFT 基线**出发；至少比较 `SFT`、`Text-only GRPO`，具备有效语音奖励后再做 `Audio-aware / Joint GRPO`。
2. Rollout 必须明确行为分布、采样温度/top-p、Old/Current/Reference Policy、组内优势、clip 与 KL 估计；禁止在截断采样下无说明地计算未截断策略概率比。
3. 音频的 8 路 Codebook 要定义**有效步骤与 mask**。Delay Pattern 中同一步未必 8 路都有效；联合 logprob 是实际有效动作项的对数概率之和，不能把“除以 8 的平均值”当作不变的联合概率。
4. Reward 候选：文本任务可验证正确性、ASR 回转写 CER/WER、语音质量代理指标与人工试听、完整性/STOP 规则；监测 Reward Hacking（空语音、短回答、重复、异常 Stop）。
5. 是否训练 Qwen LoRA、Bridge、Talker 全部还是子集，是否单卡、梯度检查点/多卡、是否需要先 Text-only，**R6 实测后再决定**。
6. RL 结果只与**相同 SFT 初始化/相同评估集**的对照比，不能把 Backbone 规模变化造成的收益归于 GRPO。

---

## 9. 评测、工程性能与交付口径

| 维度 | 最低报告指标 | 注意事项 |
|---|---|---|
| Text | 指令准确性/任务完成率、保留文本能力 | Qwen 原始文本基线不是 Omni 语音对照 |
| Vision | Caption/VQA Accuracy、失败样本与分辨率/token 数 | 64 Token 先跑通，后期才研究动态分辨率 |
| Audio Input | ASR CER/WER、语音问答 Accuracy | SenseVoice encoder 的预训练能力 ≠ Omni 已学会使用其特征 |
| Audio Output | CER/WER、生成失败率、UTMOS/人工试听（如可用）、声音时长 | 可播放 WAV 不等于可懂/高自然度 |
| Efficiency | TTFT、TTFA、RTF、P95 chunk gap、tokens/s、GPU 峰值显存/吞吐 | 沿用 P5 清晰计时边界；同设备同 batch 同采样比较 |
| Reproducibility | run ID / Git commit / config / model SHA / tokenizer+codec revision / dataset split / seed / device+dtype | 保存原始 CSV/JSONL 和失败案例；不手填假数据 |

**研究优先级**：多模态 SFT Baseline > Bridge/GRPO 受控实验 > Triton/系统消融（按实际瓶颈）> 可选视觉效率/Backbone 扩展。绝不因为“简历需要 DDP”而牺牲主线。

最终交付：公共仓库、最小可运行脚本（数据许可允许下）、架构图、1–2 分钟 Demo、训练/评测脚本、实验与限制说明、每阶段 R* 报告、可据实填写的简历 bullet。每一项贡献写清“参考上游 / 自主实现 / 自主训练 / 可验证研究效果”。

---

# 10. Codex 分阶段任务与交付规范（**执行主章节，优先阅读**）

## 10.0 所有阶段统一执行协议（强制）

### 启动与权限

- **只执行用户指定的一个阶段**。开始阶段时先输出“目标、当前代码依据、拟改文件表、风险和测试计划”；执行后输出实测结果并停止。v2.1 本地与云端分别验收，后续顺序见执行附录。
- 只允许小规模、明确授权的模型下载/数据准备/训练；**R0 不下载、不训练、不重构大模型**；R1–R4 的下载/正式训练也只按该阶段的用户指令执行，不因计划中出现命令而自动执行。
- 不准暗改核心算法；重要变更（改变 Talker 参数规模、换 Codec、改 loss 语义、改特殊 Token 协议、弃用旧训练路径）必须说明理由并获用户决定。
- 已有工作区变更先保护；每一阶段结束保留可复现 Git commit/tag（是否推送由用户决定），不自动 push/删除历史。

### 本仓库始终有效的保护规则

```text
IMMUTABLE:
  docs/stages/P0.md
  docs/stages/P1.md
  docs/stages/P2.md
  docs/stages/P3.md
  docs/stages/P4.md
  docs/stages/P5.md
  旧仓库 ../minimind-o/**
  既有 P0-P5 experiments/ 与 outputs/（不覆盖原始结果）

MUST_PRESERVE:
  MiniMind 权重严格加载/数值一致性测试
  P2 的 encoder/mixed-batch/length 测试
  P3 的 Mimi Delay/Codec/Stop 测试
  P4 的 loss/grad accumulation/checkpoint tests
  P5 的 stream/Cache/chunk/取消测试

NEW_REPORTS:
  docs/stages/R0.md ... docs/stages/R7.md
```

**每阶段固定交付格式**（报告写入 `docs/stages/Rx.md`）：

1. 本阶段目标、实际完成范围、未做事项（写具体，不能写“已支持”但无证据）。
2. 原始文件 → 新文件/函数 的变更表；重要张量形状与接口契约。
3. 运行环境、权重/数据 revision、冻结/可训练参数、复现命令。
4. 测试清单与实际通过/失败情况；如有训练，Loss、Checkpoint、实测曲线和指标；如无训练，明确写“未训练”。
5. 风险、已知 Bug、Fallback、后续阶段的明确依赖。
6. README / 待办同步摘要、Git commit、历史只读检查结果。

**通用检查**（在真实仓库执行；若命令因环境不适用，应记录原因，不得伪称通过）：

```bash
# 审计开始：只观察，不自动清理用户变更
git status --short
git diff --name-status

# P0–P5 一律只读；R0 记录并在末尾核对 SHA
shasum -a 256 docs/stages/P{0,1,2,3,4,5}.md

# 当前仓库已有检查范围（先跑原样，再新增用例）
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m pytest -q
ruff check omni tests scripts
ruff format --check omni tests scripts
python -m pip check
# uv 若已安装且有锁文件：uv lock --check

git diff --check
# 用户已提交的历史报告不得出现变更
git diff --exit-code -- docs/stages/P0.md docs/stages/P1.md docs/stages/P2.md docs/stages/P3.md docs/stages/P4.md docs/stages/P5.md
```

**严禁将此前“151 passed”的历史计数作为新版本硬门槛**：正确标准是现有测试不退化、所有新增测试通过，并如实记录新运行中的准确数量。R* 测试默认使用 tiny fixture，真实大型权重测试必须是显式启动的独立检查脚本，不放入普通 CI。

---

## R0｜审计现有仓库、更新文档、确定迁移清单（历史验收）

### 已有基础与目标

P0–P5 全部完成并已归档；Qwen3-0.6B 仅独立文本已验收。R0 要把规划从“MiniMind 优先”转为“Qwen3-1.7B 为主”，而**不修改模型核心或下载训练资源**。

### 具体任务（按顺序）

1. **工作区审计**：阅读 `AGENTS.md`、`README.md`、项目规划、待办和 P0–P5；列出实际 Git branch / dirty files / Python/依赖版本 / CI 命令 / 模型文件路径；收集历史阶段报告 SHA-256。
2. **代码清点**：用代码搜索核实 `OmniPlusOmni`、`OmniPlusModel`、`TalkerModule`、`Projector`、`MimiCodec`、`MimiStreamingDecoder`、Dataset、SFT、Cache 的真实定义、调用方、测试与旧权重格式。制作**逐文件迁移映射**：保留/新增/最小修改/禁止修改/预期测试。
3. **风险审计**：标记 MiniMind 专用配置常量（6400 词表/Token IDs、768 维、8 层、bridge 索引、旧 Cache tuple、stop mask、音频代码等）、Qwen API 可能冲突位置、阶段性资产/数据依赖。
4. **新文档落地**：将本规划写入 `docs/PROJECT_PLAN.md`；保留 v1 原版副本于 `docs/archive/`；更新 `README.md`（历史与规划分开）、`docs/项目待办.md`（保留 P0–P5 完成勾选，旧 P6+ 计划标为 superseded、新 R0–R7）、`AGENTS.md`（阶段权限/历史保护/测试边界）。如已有同名文件，先比对再最小编辑。
5. **新增配置设计文档，不实现复杂算法**：提供 Qwen Backend、Bridge 与新资产 registry 的接口签名草案和文件级任务清单。R0 可新增 `docs/reports/R0-migration-map.md` 或在 R0 报告内包含映射表，但不得预造大量空类。
6. **回归验收**：现有 pytest/Ruff/安装检查通过；确认所有 `P0.md`–`P5.md` 文件 hash 未变化；核实 README 没有把“规划”写成“完成”。
7. 新建 `docs/stages/R0.md`，列出 R1 的逐文件计划与真实执行命令；**在 R0 交付处停止**。

### R0 验收门槛 / 不得做

- [x] 现有模块与路径真实审计、含旧功能的回归命令；文档与 AGENTS 规则符合 v2。
- [x] R1–R7 的状态、依赖、可训练参数以及资产来源完整；所有不确定项清楚标记。
- [x] **P0–P5 文件 hash 与 R0 开始一致；MiniMind 历史基线仍可运行。**
- [x] 无 Qwen3-1.7B 下载/正式训练/复杂 Bridge/GRPO 实现；不做无意义的大规模目录重排。
- [x] `docs/stages/R0.md` 已包含真实文件级 R1 迁移表。R1 已获用户确认。

2026-11-06 复验 R0 通过。原有 151 项离线测试与 MiniMind Dense/MoE 真实权重回归通过。 [R0 报告](stages/R0.md)

---

## R1｜Qwen3-1.7B Thinker Backend 与 Tokenizer / Cache 适配

### 前置条件

R0 通过；用户已选择进入 R1；具备可用的官方 Qwen3-1.7B 权重路径（用户批准下载时才获取）。现有 Qwen3-0.6B 可先测试接口，但**不能代替 1.7B 最终权重实测**。

### 要做的事

1. 读取官方 config / tokenizer / chat template；固定 revision、文件 SHA、模型真实维度/层数/头数、特殊 token、generation 参数，存 `experiments/r1/`。
2. **新增独立 Qwen Backend** 而不是将自研 MiniMind Block 强行替换成 Qwen 权重；与 R0 确定的 `ThinkerBackend` 契约对齐。优先使用已安装/经测试版本的 HF 标准模型接口。
3. 支持 `input_ids`/`inputs_embeds`、attention mask、指定层 Hidden States、文本 logits、KV Cache prefill/decode、左 padding、多轮重置。规范选层和 hidden state 是否经 norm。
4. 对照**同一官方 Qwen 模型**的原生前向/Backend：固定 seed、eval、dtype、采样，核对 logits、指定 hidden、KV Decode/Cache 开关生成一致性及位置索引；记录容差，不允许放宽后不说明。
5. 保留 MiniMind Backend 与其旧测试；新增 `tests/test_qwen_backend.py`、显式真实资产脚本（文件名按 R0 映射决定），不在普通 pytest 自动下载 1.7B。
6. 生成 Qwen 专属配置与资产 manifest；整理 R2 需要的投影输入输出、Cache 契约和特殊 Token 决议。

### 通过标准

- [x] 官方完整 Qwen3-1.7B 本地加载成功，短中英文文本生成；0.6B 可做低成本先行但最终需 1.7B 实测。
- [x] 对照原生 HF logits/隐藏层数值与 prefill/decode 一致（事先声明 dtype/误差阈值），有效 padding 和 Cache 均通过。
- [x] 旧 MiniMind 回归仍通过，无未经审计的 tokenizer resize / 模型权重转换。
- [x] `docs/stages/R1.md`、README/待办更新；**不进行多模态训练**。等待用户授权 R2。

2026-11-06 验收 R1 通过。真实 1.7B 离线通过 16 项 CPU FP32 数值对照，中英文贪心输出与原生 HF 逐 ID 一致。 [R1 报告](stages/R1.md)

**阻塞处置**：Mac 内存不足/运行很慢时先用 0.6B tiny/官方结构验接口，**将 1.7B 真实验收转移到用户批准的单卡 GPU**；不得标记未做的 1.7B 测试为通过。

---

## R2｜Omni 模块迁移：Projector、Fixed Bridge、Talker 与 Streaming

### 前置条件

R1 的 Qwen Backend 数值/Cache 已验收；P2/P3/P5 旧逻辑保持可运行。

### 要做的事

1. 将 P2 冻结编码器接入 Qwen Backend：SigLIP2 `768→2048`，SenseVoice `512→2048`，处理 mixed batch、ragged audio、有效长度、图像 64 Tokens 与 mask；不重写原始 Encoder。
2. 实现首版 `FixedBridge` 协议：选中 Qwen 指定层 `[B,T,2048]` → 投影 `[B,T,768]`，写清 Norm/Layer 索引/与 `Talker.embed_proj` 的职责界限；**只预留 Adaptive Bridge 类型接口，不写 Gated 实验**。
3. 将原 4 层/768 Talker 与 8 Head 实现接在 1.7B 条件上；按 Qwen 原始文本、Chat Template 重建 audio/text joint sequence、Stop/PAD mask 与 Teacher Forcing labels。Mimi 与 Delay 原代码原则上不改。
4. 验证**真实两条 T2A 样本**的 Token 时间轴、8-Codebook Shift/Restore、预测 mask 与目标监督；同时验证 I2T、A2T/A2A 的混合路由和缺失模态/重排。
5. 对新 `OmniPlusOmni` 路径执行 forward/backward/冻结检查，确保只有 Projector、Bridge、Talker 等预期模块参与梯度；检查全忽略与无效样本处理。
6. 将生成代码接入独立 Qwen/ Talker caches、旧 P5 的 Mimi streaming decoder；用 **codec-only** + 合成/真实声码验证 PCM 分块与整段等价、左 padding、提前 EOS、截断与取消。**本阶段不要求随机 Talker 输出自然语音**。
7. 新增单元测试及真实小样本验收脚本，新配置与缓存命名空间；若需要新增 Token，验证 embedding/head 保存恢复与冻结策略。

### 通过标准

- [x] 真实/替身多模态输入的 shape、索引、Mask、finite logits、梯度检查通过。
- [x] 同版本 Mimi 8-codebook delay `encode→shift→restore` 可逆，训练标记与生成状态机一致。
- [x] Qwen Omni 新路径的短/长输入不崩溃；PCM Chunk/尾块/Cache 生命周期正确（**音质/语义质量暂不验收**）。
- [x] 旧 P2/P3/P4/P5 真实回归通过；[R2 阶段报告](stages/R2.md) 给出关键形状图与尚未训练限制。
- [x] Qwen Omni 使用独立初始化与验收记录；R3-L 单独授权。

**阻塞处置**：如果联合时轴无法在现有 Delay 约定下正确覆盖文本和更长音频，必须记录最小失败样本、当前因果依赖和替代方案供用户决策；不得静默裁掉长音频或放弃相关 mask 测试。

---

以下 R3–R7 保留 v2.0 的研究要求与正式验收定义。v2.1 把本地代码验收提前到相应 L 阶段。下文单卡 Profile、正式训练与研究效果的前置条件分别适用于 G3–G7。L 阶段使用执行附录和手册中的本地验收标准。 [v2.1 执行附录](#附录-c-v21-执行顺序与变更记录)

## R3｜数据工程、权重清单与单卡训练准备

### v2.1 本地验收（R3-L）

2026-11-08 本地验收通过。以下是本地完成项；下方原 v2.0 的云端综合验收保持未完成。

- [x] 四任务各 8 条真实诊断样本，媒体分组与 24/4/4 划分、原始语音 16k/24k 重编码和严格版本缓存通过。
- [x] `inspect/prepare/smoke/--dry-run` 已实现；参数分组、原 P4 Loss/训练步骤与完整恢复复用通过。
- [x] 真实 tiny Batch 更新、冻结参数不变；新进程 2+2 步与连续 4 步完整状态一致。
- [x] 全仓库 183 项测试、Ruff/格式/依赖检查通过；[R3-L 报告](stages/R3-local.md) 和 [G3 操作说明](G3_CLOUD_RUNBOOK.md) 已归档。

原始预编码回答因缺乏精确 Mimi 来源而隔离；A2A 当前为语音复述诊断。正式问答数据、1.7B CUDA Profile 和 B1 尚未验收。

### 前置条件

R2 的 Qwen Omni 前向与标签协议通过；用户确认数据准备范围与单卡资源。

### 要做的事

1. 根据第 5 章创建 **T2A → I2T → A2T/A2A** 的分阶段数据配置；收集完整数据字段、许可、revision 与媒体文件大小的 manifest；先小样本，禁止自动全量下载。
2. 验证 MiniMind-O 已有 T2A mini Codes 与本地 Mimi 配置的兼容性；非同版需重编码或明确排除；LJSpeech / AISHELL-3 作为备选，数据质量与语言分布可检查。
3. Qwen Chat Template、文本/特殊 Token、图像/语音占位位置和音频标签重新生成；验证去重、划分、Speaker ID、异常短音频、Codebook 合法范围和缓存失效策略。
4. 复用 P4 的 SFT Loss / 梯度累积 / Checkpoint，新增 Qwen 模型冻结与可训练模块参数分组；预留 Qwen LoRA 但首个 T2A 试验先冻结 Thinker。校验冻结 Encoder 的 eval / no-grad。
5. 计划单 GPU 的短实际试验：先 1–10 step 真实前后向；资源满足时 100-step profile 测 GPU 显存、samples/s、音频序列长度、optimizer 状态、吞吐与初步收敛；根据实测**向用户提交正式预算选择**。
6. 设计验证用样本集及独立评测集，保存样本列表及 SHA；不要复用 P4 的小小验证集作为正式 Benchmark。
7. 编写一条命令可复现的 `train/prepare/eval` 入口（实际 CLI 参数由 Codex 实现并记录，不得虚构当前存在）。

### 通过标准

- [ ] 每个任务至少有合法的小样本并完成真实 batch、梯度与 Dataset 缓存测试；train/val/test 不泄露。
- [ ] Codec/Tokenizer/Encoder 版本在 manifest 可追溯；数据许可与公开范围已检查。
- [ ] 单卡 Qwen1.7B + Talker 的真实前反向/恢复训练可以在选定设备上进行，已测量资源；若不可行则给出实测瓶颈与降本方案，**不跳过**。
- [ ] `docs/stages/R3.md`、配置、数据审计、试运行 JSON 完成；全量下载和长时训练必须等用户批准 R4。

---

## R4｜从 T2A 到多模态联合 SFT：正式 Omni Baseline

### v2.1 本地验收（R4-L）

- [x] A：复用现有 SFT，真实 Tiny T2A 同一训练集合 Loss 下降，冻结主权重。
- [x] B：分桶、验证/早停、最佳状态、每码本 CE/STOP、预测 WAV/ASR 及明确失败记录。
- [x] C：两路 Projector 独立更新、穿过冻结 Qwen 的梯度、模态对照；真实音频对照因没有同形状配对暂未测量，fixture 已通过。
- [x] D：LoRA、任务采样/权重/分组 LR、跨进程完整恢复、独立 Adapter；真实 1.7B 零初始化 LoRA 前向误差为 0。
- [x] `train/resume/evaluate/demo/profile` 真实入口、222 项回归、[R4-L 报告](stages/R4-local.md) 与 [G4 Runbook](cloud/G4_SFT_RUNBOOK.md)。
- [ ] G4 正式训练、真实 QA A2A、可懂语音、人评、CUDA 训练与 B1 Benchmark。下方原正式验收要求保持未完成。

### 前置条件

R3 的数据与单卡真实 Profile 通过；用户确认训练预算、数据版本与优化范围。

### 要做的事（按可训练参数逐步推进）

**R4-A：真实模型 T2A 短闭环**

1. 冻结 Qwen3-1.7B、SigLIP2、SenseVoice、Mimi；随机初始化 Fixed Bridge + 4 层/768 Talker + Audio Embedding/Heads。
2. 在**完整宽度模型**上验证真实样本的 teacher forcing、CE、loss mask、Stop、gradients 和恢复；先尝试少量 T2A 过拟合，不把旧 1.5M 小模型成果当代替。需要验证 Qwen 冻结时 Bridge/Talker 仍正常更新；进入 Projector 阶段后再验证梯度可穿过冻结 Qwen 回传至 Projector。
3. 将生成语音反转写、记录 CER/可懂度、解码失败率和音频时长；**不是仅输出 WAV 就算成功**。

**R4-B：扩充 T2A，解决 Talker 稳定性**

4. 逐步加入合规英文/中文 T2A 数据；检查训练质量、变长长度偏置、speaker 条件与长句稳定性。不能无限延长训练而不检查收敛和质量。
5. 如果随机 Talker 不收敛或单卡成本不可接受，**停止并呈交失败分析**；由用户决定是否使用 MiniMind 已训练 Talker 权重初始化或调整模型规模。不同初始化用不同实验名/报告，不能混淆研究贡献。

**R4-C：视觉与语音输入对齐**

6. 使用 I2T（COCO / LLaVA 合规子集）训练 Vision Projector，使用 A2T/A2A 训练 Audio Projector；Qwen 初期冻结。用真实“换图/换音频”测试模型确实依赖输入模态。

**R4-D：联合多模态 SFT**

7. 启用经过测试的 Qwen LoRA（若必要），联合训练 Projectors + Fixed Bridge + Talker + Audio Heads + LoRA；保持 Encoder/Mimi 冻结。
8. 记录不同任务 sampling ratios、loss scaling、language capability retention，确保模态任务互不冲突。完成 T2A、I2T、A2T/A2A 的至少一组独立评测。
9. 确认 `save/resume/eval/stream demo` 可复现，形成**核心 B1 Checkpoint**与固定 Baseline 评测报告；为 R5/R6 留冻结版本与独立测试集。

### 通过标准

- [ ] Qwen Omni 的训练确由目标参数发生更新；Loss/梯度/Checkpoint 数据正确且可恢复。
- [ ] T2A 生成可懂音频（通过 ASR 回转写 + 人工试听 + 失败率记录）；如果无法达到，阶段标为未完成，不能声称 SFT Baseline 已成立。
- [ ] 图像/语音输入对文本回答产生合理、可测影响；至少完成 T2A、I2T、A2T/A2A 的明确 Smoke + 独立评测，报告难例和局限。
- [ ] 保存所有训练配置、权重引用、原始结果、设备与计算预算；P0–P5 旧测试通过。
- [ ] `docs/stages/R4.md` 输出 B1 实验结果/失败分析、README 当前能力、下一阶段前置条件；**不得自动进入 R5/R6**。

---

## R5｜Adaptive Bridge 研究与消融（R4 后设计细节）

### 前置条件

R4 的 B1 SFT Baseline 已冻结，拥有独立验证/测试集。现在才由用户选择最终候选 Bridge（例如 Fixed / Last / Weighted / Gated）。

### 任务

1. 根据 B1 固定 Backbone、Talker 初始化、数据和训练预算，写出 Bridge 实验假设及公平性矩阵。
2. 自主或经用户授权实现至少 Fixed / 另两种合理候选；评审新参数量、显存、训练代价以及 norm/隐藏层选择。
3. 对每个可学习 Bridge 重新训练相应组件；使用相同评测，尽量固定相同 seeds、数据 split 与采样策略。
4. 分别画 CER/WER、UTMOS/试听、语义一致性、Loss 收敛、TTFA、训练/推理成本曲线；记录负结果。
5. 若没有独立收益，明确推荐不保留复杂 Gate；不得为了预设“创新”而挑选有利样本。

### 验收 / 交付

- [ ] 至少三组可复现结构对照（或因预算降低须在报告解释）；实验只改变被研究的因素。
- [ ] 明确 Delta 与不确定性，不把增加参数或额外训练预算隐去。
- [ ] `docs/stages/R5.md`、结构开关/训练脚本、原始实验数据/图表完成；记录最有效方案或可信负结果。

---

## R6｜Multimodal GRPO 研究（策略/预算到此时决定）

### 前置条件

有 R4 的稳定 B1 SFT Checkpoint 与可用可验证 Reward/评测集；正式启动前**先让用户选任务、奖励、参数更新范围和单/多卡方案**。

### 任务

1. **先完成设计阶段**：选择 Text-only 基础任务、Audio-aware Reward 的可验证性、Joint Codebook 概率定义；给出算法单测计划和计算/显存 Profile。
2. 由用户实现或审查关键 RL 数学：采样和行为分布、Old/Current/Reference、Group Advantage、Clipped Surrogate、KL、序列 mask 与多码本有效 Token 对数概率。
3. Codex 辅助实现 Reward 服务、数据 IO、Rollout 管理、日志、失败样本归档、可复现运行；不替代用户对核心 RL 数学的决策。
4. 同 SFT 初始化做 `SFT / Text-only GRPO / Joint GRPO`（音频奖励验证不足时先不启动 Joint）；测 Accuracy、CER/WER、UTMOS/试听、KL、长度、Reward 分布、空语音率、训练成本。
5. 若发现 Reward Hacking、概率定义错误或音频输出退化，停止扩大训练，先修复单元测试/奖励。

### 验收 / 交付

- [ ] 算法目标的数值/概率/掩码单测、rollout 与梯度检查完成。
- [ ] 受控 SFT vs RL 实验真实执行；记录有无收益和稳定性，不捏造提升。
- [ ] `docs/stages/R6.md`、Checkpoint/配置、原始日志/对照图、失败分析；资源不足时给出可核验 partial，不虚称完成。

---

## R7｜系统 Benchmark、最终报告、Demo 与发布

### 前置条件

R4 已有可验证 Omni Baseline；R5/R6 已完成或者明确记录未完成/延期；项目当前可复现。

### 任务

1. 统一构建分任务测试集：Text、I2T/VQA、A2T/A2A、T2A，冻结 split、防泄露、核对任务数据许可。
2. 固定同设备精度、Batch、长度、采样设置与计时边界，测 TTFT、TTFA、RTF、P95 chunk gap、CER/WER、文本和视觉准确率、峰值内存与训练吞吐。
3. 借助 CS336 A2 经验仅对实际瓶颈实验：AMP、KV Cache、Triton FlashAttention、可选 DDP；记录正确性误差与性能权衡。**没实际做的技术不写进成果**。
4. 完成核心受控对照：Fixed vs Adaptive、SFT vs GRPO（仅已执行的实验）；分开报告参数/计算预算变化与架构方法本身的收益。
5. 清理 README 叙述与入门命令；完成架构图、Demo（可听语音 + 视觉/语音交互）、完整技术报告、研究限制、许可证/上游归属、模型/数据来源。
6. 用**真实实验数字**填写简历表述，保留未取得优势的实验与可复现失败案例。

### 验收 / 交付

- [ ] 终端用户能够按 README 做环境安装、选择下载资产、运行 Demo/可用评测子集；不会默认下载大资源。
- [ ] 全部项目测试与格式检查通过，历史 P0–P5 未修改，R0–R7 历史追溯完整。
- [ ] GitHub README、Demo、技术报告、Benchmark 原始 JSON/CSV、消融图表、来源与许可说明齐全。
- [ ] 不虚构性能/研究贡献；若 R5 或 R6 未完成，在最终报告中明确写成未完成。

---

## 10.9 阶段交付模板（供每次 Codex 回复复用）

```markdown
# R? 阶段交付报告

## 1. 阶段目标与事实状态
- 用户授权范围：
- 已完成：
- 未完成/阻塞：

## 2. 文件级变更（原始路径 / 新路径）
| 文件 | 原用途 | 本阶段变更 | 影响/回滚 | 测试 |

## 3. 算法/数据/权重/参数
- 关键形状/状态机：
- 冻结/可训练参数：
- 官方 checkpoint/revision 与 license：
- 数据及 split SHA：

## 4. 复现实验
- 实际命令：
- 设备、精度、seed、配置：
- 结果与原始日志/Checkpoint：

## 5. 测试与历史保护
- 普通 pytest、R* 特定测试、Ruff、pip check：
- P0–P5 SHA 是否与 R0 快照一致：
- 旧 MiniMind / 新 Qwen 兼容性：

## 6. 问题、取舍与下一阶段
- 已知 bug / 数值风险：
- 后续需要用户决定的选项：
- 下一阶段准确入口、前置文件与验收门槛：
- 本阶段已停止，等待用户授权：是
```

## 10.10 开工提示词

R2 的阶段提示词随 v2.1 执行包保留。v2.0 的 R0 开工提示词见原文存档。 [R2 提示词](../CODEX_R2_PROMPT.md) · [v2.0 存档](archive/PROJECT_PLAN-v2.0.md)

---

## 附录 A｜原始资料与版本边界

本规划整理自用户给出的现有文档：`项目待办.md`、`P0.md`–`P5.md`、`Omni-plus_完整项目开发与优化规划(3).md`、`Omni++架构升级图(3).md`、`Omni++QA.md`、`Omni最终成果(2).md`，并遵循之后确定的 Qwen3-1.7B 主线/单卡/Bridge 与 GRPO 后置决议。上述报告证明的是**截止各自阶段的已测事实**，而非新 Qwen Omni 的效果。

外部官方资源：

- MiniMind-O 架构、Talker 设计、官方数据：<https://github.com/jingyaogong/minimind-o>、<https://huggingface.co/datasets/jingyaogong/minimind-o_dataset>
- Qwen3-1.7B config 与模型卡：<https://huggingface.co/Qwen/Qwen3-1.7B>
- SigLIP2：<https://huggingface.co/google/siglip2-base-patch32-256>
- SenseVoiceSmall：<https://huggingface.co/FunAudioLLM/SenseVoiceSmall>
- Mimi：<https://huggingface.co/kyutai/mimi>
- 研究启发：ELMo <https://aclanthology.org/N18-1202/>；SUPERB <https://arxiv.org/abs/2105.01051>

**版权和可分发边界**：开源源码许可、权重许可、数据许可彼此独立。不能因上游仓库采用 Apache-2.0，就自动假定所有预训练权重、第三方语音/图片数据均能再分发；正式发布前逐项审计并保留归属。

## 附录 B｜需要留到后续阶段才能决定的事项

| 问题 | 暂定决策 | 最迟何时解决 |
|---|---|---|
| 固定 Bridge 取哪一层、与 Norm 的关系 | R1/R2 先按可测接口固定一个默认层；R5 再做层选择消融 | R2 接口完成前 |
| Talker 随机初始化是否成功 | 首选随机 4×768；真实 Pilot 不佳时可提预训练回退 | R4-B |
| 数据集实际规模和 T2A 中英比例 | 候选已列，先审计与抽样，Profile 后决定 | R3/R4 前 |
| Qwen LoRA target modules/rank | 按真实模型模块名、显存和 Pilot 决定 | R4-D |
| Adaptive Bridge 具体模型与实验次数 | 不提前实现 | R5 |
| GRPO Reward、Joint Logprob、更新模块 | 不提前定完整算法，只有正确性约束 | R6 |
| GRPO 是否多卡 / DDP / ZeRO | 单卡起步；按实测需求决定 | R6 |
| Triton / 视觉 Token 压缩 / Backbone 规模扩展 | 可选，不阻塞主线 | R7 或后续 |

---

## 附录 C v2.1 执行顺序与变更记录

| 阶段 | 本地代码与极小样本验收 | 云端正式训练与实验 |
| --- | --- | --- |
| R2 | 多模态接线、真实 T2A 时间轴、梯度、Cache 与 PCM 验收 | R2 的验收目标是架构与计算图 |
| R3 | **R3-L**：独立样本划分、数据与资产 manifest、特征与声码缓存、准备入口和 checkpoint 规范 | **G3**：单卡 CUDA 短前后向和资源 Profile，确定预算 |
| R4 | **R4-L**：极小样本验证 T2A、视觉与音频对齐、联合 SFT、梯度和恢复 | **G4**：按资源结果依次训练和验收；G4D 冻结 B1 并立即运行首轮正式 Benchmark |
| R5 | **R5-L**：验证可替换 Bridge、候选实现与对照配置 | **G5**：B1 成立后由用户决定消融方案，按相同数据、底座与评测重训练 |
| R6 | **R6-L**：验证通用 GRPO 数学、有效码本联合 logprob 与极小离线 rollout | **G6**：B1 成立后决定 Reward、更新范围与单卡或多卡方案，运行受控对照 |
| R7 | 整理已验收的代码、指标与复现材料 | **G7**：统一 Benchmark、系统复测、最终报告与 Demo |

本地通过记录在阶段报告与待办的本地栏。正式模型质量、研究效果和预算以云端实测单独记录。R3-L 以 R2 的样本协议、真实资产清单与 smoke 为入口，固定独立 split 和可复现数据命令，验证小样本的特征与声码缓存，并输出可供 R4-L 使用的批次和版本清单。 [双轨待办](项目待办.md) · [执行手册](OMNIPLUS_LOCAL_FIRST_CLOUD_PLAYBOOK.md)

2026-11-08 将本地代码实现与云端实训拆为两条验收轨道。Qwen3-1.7B、Fixed Bridge、4×768 Talker、8 Codebooks、多模态 SFT、Adaptive Bridge 和 GRPO 的研究定位继续沿用 v2.0。正式 Bridge 消融与 GRPO 实验配方等待 B1。修改前的 v2.0 主规划已逐字节归档，SHA-256 为 `0143b67b280f6d15acc2b82332bb82ce9bc50f3bb31f128dd149b2d3ae307ae4`。 [v2.0 原文](archive/PROJECT_PLAN-v2.0.md) · [v1 原文](archive/PROJECT_PLAN-v1.md)
