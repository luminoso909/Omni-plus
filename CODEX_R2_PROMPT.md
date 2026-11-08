# 给 Codex 的指令｜OmniPlus v2.1，**当前只执行 R2**

请在现有 `Omni-plus` 仓库上工作。现在已确认的历史进度以仓库内报告为准：P0–P5 已有历史验收；`PROJECT_PLAN.md` 曾记录 R0、R1 已通过，R2 等待授权。**先检查真实 Git 状态和 R0/R1 报告，不要假设这些状态必然仍然最新。**

## 1. 先阅读（按优先级）

- `AGENTS.md`、`README.md`、`docs/PROJECT_PLAN.md`、`docs/项目待办.md`、`docs/stages/R0.md`、`docs/stages/R1.md`、`docs/stages/P0.md`–`P5.md`。
- **此次新增的两份文档**，应由我放入仓库：
  - `docs/OMNIPLUS_LOCAL_FIRST_CLOUD_PLAYBOOK.md`
  - `docs/OMNIPLUS_DECISIONS_AND_DATA.md`
- 现有 `omni/`、`scripts/`、`tests/`、`configs/`、`experiments/` 的真实源码、调用关系与测试。

如果新增文档在仓库中尚不存在，请先报告缺失，不要凭文件名猜测内容；我会提供文件或把它们放入 `docs/`。

## 2. 本次明确授权：**只执行 R2**

### A. 项目文档顺路转向 v2.1（最小改动）

将新执行共识同步到 README、`docs/PROJECT_PLAN.md`、`docs/项目待办.md`、`AGENTS.md`：**R2 正常完成之后，在本地用 Tiny 数据分阶段实现/验证 R3-L、R4-L、R5-L、R6-L 的代码，然后再上云运行 G3→G4→G5/G6→G7。** 这**仅改变代码实现/云端训练的时间顺序**，不改变 Qwen3-1.7B、Fixed Bridge、4×768 Talker、多模态 SFT、Adaptive Bridge、GRPO 的研究定位。

请严格区分“本地代码测试通过”和“云端模型/实验效果验收”，在待办里使用两套状态；保留 v2.0 主规划的版本归档与差异记录。**不要**修改 `docs/stages/P0.md`–`P5.md`、已有实验结果或旧仓库，也不自动改写已签收的 R0/R1 事实。

### B. 结构适配

1. 复用 R1 已验收的 **Qwen3-1.7B Backend**。复用 P2 冻结的 SigLIP2 `[N,768]`、SenseVoice `[L,512]`；将 Vision/Audio Projector 输出适配 Qwen hidden 2048。保留真正有效的 ragged length、mixed batch、padding、模态缺失/重排和特征冻结策略。
2. 实现或接入**唯一负责 Qwen hidden → Talker 条件**的 `FixedBridge`（2048→768）。明确 chosen hidden layer、index、norm、`Talker.embed_proj` 的职责；**不得重复做两个 2048→768 投影**。只保留未来 Adaptive Bridge 替换契约，不实现 Weighted/Gated 研究算法。
3. 沿用已有 4 层/768 Talker、8 个 Codebooks、Audio Embedding/Heads、Mimi 和 Delay/STOP/PAD 核心，不擅改结构、码本数或停止规则。
4. 用 **Qwen 自身 tokenizer/chat template** 重建图像/音频占位、回答文本、T2A 的 teacher-forcing 输入与 labels；确保原始 MiniMind Token IDs 绝不被当成 Qwen Token IDs。检查新增特殊 Token 的 embedding 初始化与保存恢复（仅当确实新增时）。
5. 对至少 **两条真实 T2A** 做可阅读的逐位置时序图和 tensor shape 证明：文本 token 与音频帧不天然一一对应；Audio `encode→shift→restore` 必须严格可逆；不能把未来回答音频放进输入，也不能静默裁掉长音频。再覆盖 I2T、A2T/A2A 的混合路由。
6. 跑完 new Omni 的 forward/backward 检查：Frozen Qwen/Encoder/Mimi 参数不更新，Projector/Bridge/Talker 只有应当更新的参数有有限梯度。**训练 Projector 时梯度需穿过冻结 Qwen；不能把该段 forward 整体包进 `torch.no_grad()`**。
7. 复用 P5 流式能力，分别验证 Thinker/Talker/Mimi Cache 生命周期、左 padding、自然结束、预算截断、取消、尾块 PCM；使用 Codec-only/真实已知 Codes 对照。**R2 不要求随机 Talker 生成自然可懂语音。**

### C. 测试/交付

- 优先新增局部小测试，不破坏旧实现。普通 pytest 不自动下载大型模型；真实资产使用显式验收脚本。
- 运行 `pytest`、Ruff、P2/P3/P4/P5 相关回归；所有失败都如实记录，不放宽断言以求通过。
- 新增/更新 `docs/stages/R2.md`：真实修改文件、张量与时间轴、冻结/可训练参数、测试与实际命令/结果、未训练声明、已知风险，以及 **R3-L 入口**。
- 同步 README 和待办中的事实状态；保护 P0–P5 文件 SHA 与原始实验输出。提交代码、配置和报告（不提交大数据或权重）；是否推送远程按我授权。

## 3. **严格禁止**

- 不进入 R3-L、R4-L、R5-L、R6-L 的实际开发；它们是**下一步**，不是本次授权。
- 不下载全量数据，不启动长时间训练，不租/操作云 GPU，不生成 B1 Checkpoint 的假结论。
- 不提前敲定 R5 的候选 Bridge 效果、R6 的奖励/更新范围/多 GPU 配方；不擅写我保留亲自实现或审核的核心算法。
- 不重新初始化项目、删除旧 MiniMind 回归、修改 P0–P5 历史报告、覆盖已存在 checkpoints 或实验日志。

## 4. 完成后回复格式

按顺序输出：
1. 真实 Git / 仓库状态与 R2 目标；
2. 文件级变更表及为何最小改动；
3. 本次 Qwen→Projector→Bridge→Talker→Mimi 的完整关键张量/标签/Mask 链；
4. 实际运行过的测试命令、通过/失败与历史 hash；
5. 当前可用的 R2 Smoke 入口命令；
6. 后续 **R3-L→R4-L→R5-L→R6-L→G3→G4→G5/G6→G7** 的两轨待办总览（只列，不执行）；
7. `docs/stages/R2.md`、README/待办更新摘要、Commit ID、未解决风险。

**执行 R2 完成后立刻停止，等待我决定是否启动 R3-L。**
