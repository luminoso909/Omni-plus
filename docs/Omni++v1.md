# Omni-plus：完整项目开发、学习与优化规划

> 版本：v1.0｜2026-11-03  
> 定位：基于 MiniMind-O 思路的轻量级全模态模型自主复现与算法研究  
> 用法：保存为 `docs/PROJECT_PLAN.md`；Codex 按阶段执行，不得跳阶段或自动实现用户预留的核心算法。

---

## 0. 项目目标与边界

### 最终成果

构建可复现、可扩展的 **Omni-plus**：支持文本、图像、语音输入，生成文本与流式语音；具有 MiniMind Dense/MoE 基线、可替换 Thinker 的 Qwen 适配版本，以及自主实现的跨层 Bridge、多模态 GRPO、统一基准评测与消融报告。

### 核心原则

1. **先复现再研究**：先在相同权重、Tokenizer、输入与推理条件下复现 MiniMind-O，再动算法。
2. **核心自己写，周边交给 Codex**：本人实现 Thinker、Talker、Codebook/Alignment、Bridge、关键 SFT/GRPO 数学和推理；Codex 承担环境、数据 IO、运行入口、CI、日志和图表。
3. **每阶段可运行**：每阶段具有测试、实验配置、结果与阶段报告；下一阶段不得依赖未经验证的“理论上可用”。
4. **模型替换不是换文件**：MiniMind→Qwen 必须切换完整 Backbone、Tokenizer、特殊 Token、状态映射及跨模态对齐；禁止只替换 vocab/merges/权重就声称兼容。
5. **不夸大贡献**：明确区分开源参考、本人复现、Agent 协作和个人新增算法；不把未跑的实验写成已完成。
6. **不在首轮加入新复杂度**：不新增复杂 MoE 变体、Q-Former、分布式推理框架或 Agent 功能，除非实验明确需要。

### 成功标准（按层次）

- **最低可交付**：Dense 自主复现、权重兼容、三模态输入/文本语音输出、SFT/推理、可复现 Benchmark。
- **核心完成**：再加 MoE、Adaptive Bridge、GRPO 对照实验及严格消融。
- **扩容研究**：Qwen 0.6B Backbone 适配与重新对齐，展示是否真正优于 Dense；不把“能加载语言模型”误认为“Omni 训练完成”。

---

## 1. 最终系统结构

```text
Text ─────────────── Tokenizer ───────────────┐
Speech ── SenseVoice (frozen) ── AudioProj ───┤
Image ─── SigLIP2 (frozen) ─── VisionProj ────┤
                                              ▼
                       Thinker Backend (MiniMind Dense / MoE / Qwen)
                             │                      │
                         Text Head            Hidden States
                             │                      │
                         Text Output        Bridge / Adaptive Bridge
                                                    │
                                         Talker Transformer
                                                    │
                                  Delayed 8-Codebook Predictor
                                                    │
                                     Alignment / Mimi Codec
                                                    │
                                           Streaming Audio
```

说明：Mimi、SenseVoice、SigLIP2、说话人提取器可以直接用成熟预训练模型，研究重点在其输入输出和融合接口，不要求从零重写 Codec/Encoder。Streaming 指可连续输出音频块；是否真正实时由 TTFA、RTF 和音频间隔测量决定。

---

## 2. 目录规划（渐进式创建，不预生成空类）

```text
Omni-plus/
├── configs/
│   ├── models/                 # minimind_dense / minimind_moe / qwen
│   ├── training/               # SFT / GRPO
│   └── inference/
├── omni/
│   ├── models/
│   │   ├── thinker/            # norm / rotary / attention / mlp / moe / blocks / adapters
│   │   ├── encoders/           # 外部预训练编码器包装
│   │   ├── talker/             # transformer / codec heads / alignment
│   │   ├── projectors.py
│   │   ├── bridges.py
│   │   └── omni_model.py
│   ├── data/                   # formats / alignment / collator / datasets
│   ├── training/               # sft / losses / grpo / rewards
│   ├── inference/              # generation / stream / cache
│   ├── evaluation/             # suites / metrics / ablation
│   └── utils/                  # logging / checkpoint / seed
├── scripts/                    # 极薄 CLI，调用 omni 中真正的代码
├── tests/                      # unit / integration / reference_parity
├── experiments/                # YAML 配置、JSONL/CSV 指标与运行元数据
├── docs/
│   ├── learning/              # Q&A 与推导笔记
│   ├── stages/                # 阶段性成果报告
│   └── reports/               # 最终报告
├── assets/                    # 少量可公开演示样本
├── .github/workflows/
├── AGENTS.md
├── pyproject.toml
├── README.md
├── LICENSE                      # 依上游授权审查后确定
└── .gitignore
```

运行权重、真实大型数据、原始波形、个人路径、缓存和庞大检查点一律在仓库外存储并由配置引用；仅将可公开的小样本、实验摘要和可复现配置提交 Git。

### 架构接口契约（先定接口再写实现）

- `ThinkerBackend`: `embed_tokens(...)`、`forward_embeddings(inputs_embeds, attention_mask, cache, output_hidden_states)`、`text_logits(...)`，返回统一对象（包含 logits、约定层的 hidden states、cache）。不强迫 MiniMind/Qwen 继承彼此的类。
- `ModalEncoder`: 输出 `features` 与有效长度/掩码；不偷偷改全局设备状态。
- `Projector`: 输入 `[B, N, D_in]`，输出 `[B, N', D_thinker]`，明确是否改变 token 数。
- `Bridge`: 输入所选层的 `[L,B,T,D]` 或 `[B,T,D]`，输出 Talker 所需特征及 mask。
- `Talker`: 输入条件特征、音频历史与 speaker 条件，输出八路音频 logits 和可选 cache。
- `CodebookAlignment`: 定义训练标签错位、生成序列还原、有效长度与 stop/pad mask；不得将不同阶段的 offset 写成散布的魔法常数。
- `OmniModel`: 仅负责组合模块和模态路由；编码器下载、ASR 评测、Mimi 解码器创建不写入其构造函数的隐式副作用。

---

## 3. 分工与学习方法

| 工作 | 本人（必须理解/实现） | Codex（可协助） |
|---|---|---|
| Backbone | RMSNorm/RoPE/GQA/Block/MoE 及权重接口 | pytest、权重键映射、自动测试 |
| 多模态 | 融合/Projector/Bridge/Talker/Codebook | 模型下载器、音频/视觉外部包装 |
| 数据 | 特殊 Token、位置对齐、Mask、目标构造 | Parquet/IO、采样、数据校验 |
| SFT | 文本音频 Loss、梯度与训练核心 | CLI、AMP/DDP 接线、日志和 checkpoint |
| 推理 | KV Cache 规则、生成状态机、流式码本重排 | Demo/API、音频文件输出 |
| GRPO | Rollout/行为分布/logprobs/Advantage/loss | Reward 接口、实验任务排程 |
| 科研 | 问题假设、消融对照、解释结论 | 跑重复实验、图表和报告排版 |

**学习循环**：读原实现与资料 → 手画张量路径 → 写最小代码 → 固定随机种子单元测试 → 与参考实现比对前向/梯度 → 小数据跑通 → 写 1 页阶段解释与 Q&A。

**Codex 修改规则**：不得未经同意替用户补齐核心类；遇到核心 bug 提出最小失败用例及定位建议；不得默默调整实验配置掩盖失败；不得把警告全局过滤；不得自动删旧数据或提交大权重。

---

## 4. 全阶段路线与验收

### P0｜独立仓库与环境初始化

**学习**：Python 包结构、依赖、配置、Git 管理。  
**本人**：确认模块边界与接口。  
**Codex**：创建项目骨架、Python 3.12 环境、`pyproject.toml`、Ruff/pytest、AGENTS.md、GitHub Actions、README、.gitignore；检测 MPS/CPU。  
**验收**：`pip install -e '.[dev]'`、`import omni`、`pytest -q`、`ruff check .` 成功；无模型/数据下载；旧仓库不变。  
**交付**：`docs/stages/P0.md` + 初始化 commit/tag `p0-bootstrap`。

### P1｜MiniMind Thinker：Dense 与 MoE

**学习**：RMSNorm、RoPE/YaRN、GQA、残差与 FFN、MoE Router/Top-K/Aux Loss、KV Cache 基本布局。  
**本人代码**：`thinker/` 内的基础层、Dense/MoE Block、Transformer、统一 Thinker 接口；实现权重映射需预先确定命名约定。  
**Codex**：官方权重读取、键名映射工具、参考测试。  
**验收**：
- shape、causal mask、padding mask、单步 Cache 与全量前向等价；
- MiniMind Dense 对齐前向 logits 与层输出（统一 dtype、eval、dropout、设备，报告容差）；
- MoE 对齐 Router、专家选择和 aux loss，验证实际激活参数量；
- 权重加载默认严格；明确且记录被跳过/重新初始化的参数。

**交付**：MiniMind Dense/MoE 文本推理基线、权重兼容测试、`P1.md`。

### P2｜多模态输入与特征对齐

**学习**：SenseVoice fbank/时序特征、SigLIP2 patch tokens、Projector 与占位符替换、speaker embeddings。  
**本人代码**：Projector、模态占位符/长度/位置 mask、文本+音频+视觉统一 embedding 路由。  
**Codex**：外部预训练编码器适配、数据路径、设备生命周期测试。  
**验收**：三模态组合的 shape/mask、同一 Batch 中混合有/无音频/图像的索引不乱；冻结模块 `eval` / device / dtype 正确；至少一个图文和一个语音理解前向用例。

**交付**：P2 多模态理解前向 + 诊断输出，`P2.md`。

### P3｜Talker、Mimi 与延迟音频码本

**学习**：Waveform、采样率、频谱、Neural Audio Codec、RVQ、八码本延迟排列、因果生成。  
**本人代码**：Talker Embedding/Head、Talker Transformer、固定 Bridge、CodebookAlignment 的 encode-shift/restore/mask/stop。  
**Codex**：Mimi 初始化与波形 IO、自动边界测试。  
**验收**：
- 合成码本 shift→restore 精确一致，测试起点、结束与短序列；
- Mimi codec-only 还原基准先测，避免将 Codec 缺陷归咎于 Talker；
- 同权重/同输入与旧版音频 logits 比较；
- 输出语音可播放且声学 token 顺序正确。

**交付**：完整文本→语音链路、音频对齐单测、`P3.md`。

### P4｜Dataset、Collator 与多阶段 SFT

**学习**：Parquet 样本、chat template、text/audio label、ignore index、Teacher Forcing、stop token 加权、loss 归一化。  
**本人代码**：标签与 mask 生成，text/audio loss，单步反传与训练循环关键逻辑。  
**Codex**：Dataset 读取、可复现采样、配置、checkpoint/resume、AMP/DDP 接线与 logger。  
**验收**：
- Batch 混合模态不丢原索引；
- 训练/评估增强隔离、音频变速与标签对齐不冲突；
- loss 与独立手算一致，mask 区域梯度为零；
- 32–128 条样本能够明显过拟合；
- save→resume 一致，optimizer/scaler/RNG 可恢复；
- T2A、A2A、I2T 各有 smoke test；真实大规模数据训练只在通过检查后启动。

**交付**：SFT checkpoint、loss 曲线、`P4.md`。

### P5｜推理、KV Cache 与 Streaming

**学习**：Prefill/Decode、每层 Cache 的位置语义、双路生成状态机、音频 chunk、停止与打断。  
**本人代码**：文本+音频联合生成、Cache 使用、Delayed Codebook streaming reassembly。  
**Codex**：命令行聊天、API/Demo、文件写出、性能采样。  
**验收**：Cache On/Off 输出一致（确定性采样）；stop/pad/长短回复正确；测 TTFT、TTFA、RTF、P95 chunk gap 和内存，端到端与模型耗时分开；实测设备条件记入元数据。

**交付**：本地 Demo、推理基线报告、`P5.md`。

### P6｜Qwen Thinker 扩容适配（独立分支，非最小完成条件）

**目标**：将预训练 Qwen3-0.6B 当作完整 Backbone，与原版 MiniMind Thinker 做受控比较。  
**本人**：实现或审核 `QwenBackend` 接口、Tokenizer 与 chat template/特殊 token 映射、Projector/Bridge 维度适配、文本→音频对齐策略。  
**Codex**：权重下载与复用、适配器测试、统计可训练/冻结参数。  
**训练顺序**：Qwen 文本能力验证 → 冻结 Thinker 训练 Bridge/Talker → 音频/视觉 Projector 对齐 → 必要时 LoRA → 多模态评测。  
**注意**：由于文本切分与原 MiniMind 不同，不能直接复用逐 token 文本-音频标签；必须重新生成或重新定义对齐；权重转换成功不代表语音能力可用。  
**验收**：文本任务无明显退化、图像/语音各链路可推理、至少一个完整训练集实验、与 Dense 用统一测试集报告资源/质量权衡。

**交付**：Qwen Omni 适配权重与实验报告（可将该阶段后置到 P7 后）。

### P7｜Adaptive Bridge：核心架构研究

**假设**：固定选取 Thinker 某一层不能在所有上下文下提供最适合 Talker 的信息。  
**本人**：实现 `FixedBridge`、`MeanBridge`、`LearnedLayerBridge`；可选 `TokenGatedBridge`（仅前三者有效后）。  
**Codex**：可复现实验 config、消融矩阵、可视化权重曲线。  
**对照原则**：相同 Thinker、数据、训练预算、种子和评测集，每种 Bridge **重新训练可学习组件**；不能仅在推理时改层号就声称比较结构优劣。  
**指标**：CER、UTMOS、语义正确性、TTFA、参数量/吞吐和训练稳定性。  
**验收**：至少固定/均值/可学习 3 组，报告均值与离散度；无提升也客观记录。

**交付**：核心算法代码、对比图、`P7.md`。

### P8｜Multimodal GRPO：后训练研究

**学习**：Policy Rollout、采样分布、Old/Reference Policy、优势归一化、Clip Objective、KL 估计、Reward Hacking、时序级 vs token 级目标。  
**本人**：实现 Rollout、联合 text+audio logprobs、group advantage、clipped surrogate、reference KL 的正确估计与掩码；改进 Reward（音质 UTMOS、CER、停止结构、语义完成度）。  
**Codex**：奖励服务的加载与缓存、可视化、失败样本归档、调度器。  
**验收**：
- 冻结策略时新旧 logprob 一致；采样温度/top-p 截断与训练目标明确；
- 八码本 logprob 的有效 token mask、聚合归一化不引入无说明的目标变化；
- Reward 全相同、空语音、解码失败均有明确处理；
- 对照 SFT / text-only RL / joint RL（条件允许再加 audio-only）；
- 除总 Reward 外统计 KL、长度、CER、UTMOS、文本正确率、音频失败率，防止短回答奖励投机。

**交付**：RL 实验结果与训练曲线，`P8.md`。

### P9｜系统化评测、消融、展示与发布

**本人**：提出假设、筛选有解释力的对照、审查图表和写结论。  
**Codex**：实验执行器、指标合并、置信区间、绘图、GitHub README/Demo/报告生成。  
**固定比较**：
1. Dense vs MoE vs Qwen（相同评测集，披露模型、训练预算不同导致的不可归因因素）；
2. Fixed vs Mean vs Adaptive Bridge（独立训练）；
3. SFT vs GRPO（相同初始化）；
4. Cache/precision/streaming 性能（同设备同配置）。

**评测集设计**：独立 train/val/test、无泄露，初始 smoke 10–20 条，正式测试优先 100–300 条跨任务样本；条件允许每设置至少 3 个随机种子。原项目报告里的 20 道选择题或少量音频不能直接支撑全面性能结论。  
**指标**：文本/视觉/音频理解准确率；音频 CER、UTMOS、speaker similarity、长度与生成失败率；TTFT/TTFA/RTF/chunk gap/内存；可复现实验脚本和设备信息。

**交付**：公开 GitHub 仓库、简洁 README、架构图、Demo、模型/数据来源、完整实验 CSV/JSON、技术报告、限制与失败案例。

---

## 5. 优先级与依赖关系

```text
P0 → P1 → P2 → P3 → P4 → P5 ──────────────────────────┐
                         │                           │
                         ├──→ P7 Adaptive Bridge ──┤
                         ├──→ P8 GRPO ──────────────┤──→ P9
                         └──→ P6 Qwen (可选/后置) ──┘
```

- **必须**：P0–P5、P7、P8、P9。
- **可延期**：P6 Qwen 扩容；可先做 P7/P8，再迁移到 Qwen。
- **不可并行盲目推进**：未通过 Dense 参考一致性测试前，不开展大规模训练与 Qwen 扩容。
- **明确暂不做**：从零训练语音 Codec、从零预训练 0.6B、复杂 Continuous Batching、多机器 DDP/ZeRO 重构。CS336 A2 的经验迁移到训练时即可，不强制重新实现所有系统功能。

### 计算资源安排

- **Mac MPS/CPU**：单元测试、小样本 smoke、权重映射、32–128 条过拟合、简短推理。
- **云 GPU**：正式 SFT、Qwen 跨模态适配、Bridge 多设置训练、GRPO 和可比 Benchmark。
- 不对具体显卡训练小时数或费用做未经测量的保证。先跑一组 100-step profile，再根据实际吞吐、序列长度、显存估算完整实验。

---

## 6. 旧仓库迁移与风险审计

**不整体复制旧仓库。** 保留只读原始项目及 Agent 改造版，建立新旧代码与测试映射：

| 旧文件 | 新模块 | 处理 |
|---|---|---|
| `model_minimind.py` | `thinker/` | 自主复现与权重 parity |
| `model_omni.py` | `omni_model.py`、`bridges.py`、`talker/`、`encoders/` | 拆分、依赖注入、测试 |
| `omni_dataset.py` | `data/` | 独立对齐/标签与数据 IO |
| `train_sft_omni.py` | `training/sft.py`、`losses.py` | 自主实现损失与训练步骤 |
| `train_rl_omni.py` | `training/grpo.py` | 重新审查概率与目标 |
| `rl_rewards.py` | `training/rewards.py` | 奖励分量可配置、失败显式记录 |
| `trainer_utils.py` | `utils/` | Codex 精简 checkpoint/日志/启动 |
| `make_rl_prompts.py` | `scripts/` / `data/` | 数据生成，与训练/测试隔离 |

**优先排查的风险**（源自上传的旧版实现，不代表已证实运行错误）：

1. `OmniDataset` 与 `omni_collate_fn` 中混合模态样本的原始 Batch 索引；不能只把非空音频/图像简单拼接再假设位置一致。
2. 8 路 Codec 延迟位置、首帧占位、stop token 和训练 label shift；必须用可逆合成测试覆盖。
3. 旧版 `strict=False` 加载会掩盖丢失权重；新版本要按预期加载项白名单控制并保存报告。
4. GRPO 用温度和 top-p 采样但记录未截断分布的 logprob 时，必须明确所优化的 policy 与 off-policy 修正含义；不能仅凭 ratio 约等于 1 就断定 GRPO 完全正确。
5. KL 项不能简单把同一组采样上的 logprob 差值均值当作可靠参考 KL；采用有定义的估计器并检查数值。
6. 把各码本 logprob `/8` 会改变序列概率与比值定义；应明确是 token-level 目标还是完整 action 的序列目标。
7. 外部编码器未注册为 `nn.Module` 时，`to` / `eval` / `state_dict` 生命周期需人工管理并测试。
8. 基线 TTFT/RTF/音质结果在不同旧报告里可能来自不同配置；统一 run ID、权重哈希、设备、精度、数据 split 与计时范围后再比较。

---

## 7. 实验与文档规范

### 实验记录最小字段

`run_id, git_commit, model_family, checkpoint_hash, tokenizer_revision, config_path, data_version, split, seed, device, dtype, batch_size, max_tokens, generation_params, train_steps, metric_method, metrics, timestamp`

所有原始结果保存为 JSONL/CSV，生成图表使用脚本读取原始文件；不手填报告数字，不覆盖旧结果。

### 每阶段统一报告模板（`docs/stages/PX.md`）

1. **目标与问题**：当前阶段解决什么，为什么这样设计。
2. **实现摘要**：改动的模块、数据/张量形状、关键计算或数学公式。
3. **测试与实验**：测试命令、数据规模、设置、运行环境。
4. **结果与证据**：日志、loss、指标、图表、对照结果；尚未执行的标为 TODO。
5. **问题与局限**：失败案例、已知 bug、待验证假设。
6. **下一阶段入口**：依赖、禁止自动修改的关键文件。

学习 Q&A 放 `docs/learning/`，按主题追加更新，不把大段学习记录塞进 README。

### 发布规范

README 只写已实现特性；说明继承/借鉴 MiniMind-O、第三方模型的权重与数据许可；保留适用版权声明。模型权重的再分发资格、数据许可证、示例音频授权应单独核验。AI 协助不妨碍项目署名，但应明确所做工作与来自上游的设计。

---

## 8. 最终交付与简历贡献口径

**代码**：自主实现的 Thinker（Dense/MoE）、跨模态输入映射、Talker/Codebook、SFT/GRPO/Streaming；可替换 Backbone。  
**研究**：Adaptive Bridge 的受控消融；Multimodal GRPO 的准确性与收益；MiniMind 与 Qwen 的能力/成本比较（如果 P6 完成）。  
**评测**：统一多模态评测集、音频质量与实时延迟报告、实验复现脚本。  
**展示**：简洁架构图、1–2 分钟 Demo、总报告、各阶段技术日志、清晰许可证和来源。

最终简历 bullet 应由**真实实验量化结果**填写：例如某项基线指标 `X→Y`、提升百分比及测试条件；若没有显著收益，表述成设计与验证的发现，不能捏造提升。

---

## 9. Codex 协作指令（执行约束）

> 你是 Omni-plus 的工程协作开发者。请先阅读本文件和 AGENTS.md。每次只执行用户明确指定的一个阶段，先输出具体文件变更计划，再完成环境/脚手架/测试等授权任务。`thinker/`、`talker/`、`bridges.py`、`training/sft.py`、`training/grpo.py` 和 `inference/` 的核心算法由用户逐步实现；未经批准不得填充或直接替换。完成后运行测试，汇报修改文件、命令、结果、仍存在的问题与后续可选任务。禁止用大量占位函数制造“已完成”的假象，禁止自动下载大型模型、训练或修改其他仓库。

**立即下一步**：仅执行 P0 创建仓库/环境/基础测试；不要自动进入 P1。
