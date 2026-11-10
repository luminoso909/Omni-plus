## 一、审查结论与范围

**结论：PASS WITH REQUIRED FIXES。**

2026-11-08 独立审查：Bridge 数学、当前接口、冻结与梯度、同配置 Tiny Checkpoint/Resume 通过；发现 **2 项 P1 / Major、1 项 P2 / Minor**，没有确认的 P0 / Blocker。两项 P1 修复并复验前，不能将当前框架签收为可直接进行公平 G5 消融的完整保护层。无需重写 Bridge、Talker 或训练器。

| 审查对象 | 实际版本 / 状态 |
| --- | --- |
| R5 实现 | `046e9470fc23b843ef93aa6161a949cc014ea79a` |
| 交接与当前 HEAD | `a51b560a731d67efa136ed7fd79082044218a6f8` |
| R4-L 基线 | `f3b593f687d2c6986a41b2acdd5bd8b9e7a4728d` |
| 审查 Diff | `git diff f3b593f 046e947`，优先新增/修改代码及其调用边界 |
| 当前环境 | Mac arm64，Python 3.12.14、PyTorch 2.15.0、Transformers 4.57.6；CPU FP32 |
| 本轮写入 | 仅本报告；临时探针、测试产物在 `/private/tmp/` |

已阅读 AGENTS、PROJECT_PLAN、本地/云端执行手册、R4-local、R5-local、交接文件、G5 Runbook，以及 Bridge、Qwen Backend、Omni v2、Talker、Config、Trainer、Checkpoint、数据契约、CLI 和相关测试的实际代码。R4-L 所需训练/恢复测试重新通过；它的正式 1.7B CUDA 训练、B1 和音质验收仍未完成。

进入时存在用户未提交的旧 P5/模型/测试改动、规划文档更名及 Notebook。本轮先记录 197 个既有文件的 SHA 或不存在状态，未清理或纳入任何提交。审查既覆盖当前工作区，也在临时目录对 HEAD 的纯 Git 归档运行回归，避免把未提交文件当作 Sol 的交付内容。

## 二、需要修复的问题

### 1 P1-01：B1 与 G5 之间缺少跨阶段留出集隔离检查

**位置：** `omni/training/r3_local.py:429–444`，`scripts/bridge_ablation.py:63–75`。数据身份的现有来源为 `omni/data/sft_sources.py:135–145`；当前缓存内分组检查位于 `omni/data/r3_prepare.py:116–169`。

**现象与复现：** `initialize_bridge_base()` 校验来源 Checkpoint/Manifest 自洽及模型、Tokenizer、媒体资产兼容，但没有比较来源训练数据与目标 Validation/Test。`plan()` 只核验当前缓存与当前留出样本。使用现有 Tiny Checkpoint fixture，构造以下两个自洽契约，热启动成功且所有共享参数确实导入：

| 字段 | 来源 Fixed Checkpoint | 新候选 |
| --- | --- | --- |
| record_id | `voice-001` | `voice-001` |
| group_id | `shared-media-group` | `shared-media-group` |
| media_ids | `sha256:identical-audio` | `sha256:identical-audio` |
| split | `train` | `test` |

实际输出为 `result="ACCEPTED"`、`shared_parameters_loaded=true`。探针只模拟已保存的非零参数和数据身份，没有训练或伪造真实数据成果。文件：`/private/tmp/omni-r5-review/cross_stage_leak.py`，输出：同目录 `cross_stage_leak.log`。

**影响：** B1 已见过的媒体可以重新作为 G5 独立测试数据。即使四个候选共用相同 G5 split，也不能据此证明对未见数据的效果。现有同一缓存内去重仍然有效；本问题是 B1→G5 的跨阶段联系缺失，并非证明已记录的四组 Tiny 运行发生了泄漏。

**最小必要修复：** 给 B1 发布清单绑定可核验的训练、选择用 Validation、封存 Test 身份及来源摘要；包括其继承组件的已知训练数据。复用当前媒体哈希/分组字段，在 `plan` 和实际热启动前检查至少 `B1.train ∩ G5.{validation,test}`、`B1.validation ∩ G5.test`，跨任务和更换 record ID 仍按底层内容识别。维持已有封存 Test 的角色，不把它重新投入候选训练。允许增加真正独立的新媒体，不要求整份新旧数据完全相同；来源记录不足时应拒绝认证正式 G5 独立留出集，明确 Tiny 诊断的有限证据。

**补测与验收：** 同媒体跨角色/任务、相同字节不同 ID、传递分组及纯文本身份冲突必须拒绝；原 split 不变及独立扩充应通过；来源缺字段时明确报错。拒绝须发生在写入运行目录或导入模型参数之前。测试不得通过重新命名样本或降低哈希检查绕过。

### 2 P1-02：执行源码指纹漏掉实际生成和 Collator 代码

**位置：** `omni/training/r3_local.py:560–591`；`omni/models/model_omni_v2.py:211` 调用 `LegacyOmni.stream_generate()`；对应实现在 `omni/models/model_omni.py:207`，音频起点在 `:274`。矩阵绑定和比较源码摘要位于 `scripts/bridge_ablation.py:79–93`、`:110–118`。

**现象与复现：** `run_identity()` 的源码清单未包含 `omni/models/model_omni.py`，数据契约的 implementation 清单也不包含它。临时复制源码，把 `audio_start = None if open_thinking else 1` 改为 `... else 2`，明确改变音文生成相对时间，但完整 `run_identity()` 比较仍返回相等。文件 SHA 从 `ebb1c1d9…` 变为 `0c961a6a…`，实际输出 `identity_unchanged=true`、`file_present_in_source_identity=false`。

同样需补齐实际执行的 `omni/data/omni_dataset.py`（Collator）和 `omni/models/config.py` 等未覆盖依赖。这里依据的是运行身份和缓存身份的联合覆盖范围，而非要求每个文件在两处重复登记。探针：`/private/tmp/omni-r5-review/source_identity_probe.py`；主审重跑输出：`source_identity-root.log`。

**影响：** 改动生成时序或补齐/标签逻辑后，恢复/评估仍可能通过原身份检查；矩阵也无法据现有 `source_sha256` 区分这类变化。Git SHA 是记录项，当前没有作为严格恢复或跨候选可比性的补充门槛，而且同一个 SHA 下也可能有未提交修改。这是可复现性保护缺口，不是 Bridge 层融合公式错误。

**最小必要修复：** 统一计算覆盖模型、Collator、Loss、训练和实际生成/评测执行路径的源码指纹。可以对受控的 `omni/**/*.py` 及真实运行 CLI 建立稳定摘要，或维护有覆盖测试的完整清单；不必增加注册框架。矩阵、Checkpoint、训练与评估记录使用同一身份，缺失必要字段不能认证正式可比性。为新身份增加 schema/version，保留旧证据和旧版本复现入口，不能回填修改历史 JSON。

**补测与验收：** 分别在临时副本修改 Legacy 生成、Collator 和 Config 的可执行代码，身份必须变化，旧 Checkpoint 的严格 Resume/Benchmark 必须拒绝；不变源码下正常恢复继续通过。对共同源码不同的候选，`collect` 必须标记 invalid/non-comparable。

**已排除的扩大解读：** 探针还把临时复制的 Last 训练 manifest 的 Transformers 字段换成哨兵字符串，`collect` 仍返回 `comparable=true`，说明汇总缺少该字段的直接自洽检查。但正常运行的缓存契约已绑定 `processors.transformers`，Checkpoint identity 也绑定 runtime；因此本审查**不把这个 JSON 探针当成真实跨版本训练可以绕过全部检查的证明**。Projector、Codec 的实现也已在数据契约中登记。可在统一身份修复时显式对齐运行时记录，避免误称这些已有保护完全缺失。

### 3 P2-01：统一汇总遗漏已记录的训练资源成本

**位置：** `scripts/bridge_ablation.py:146–174`、`:248–268`；原始训练记录在 `omni/training/r3_local.py:920–923`、`:954–959`。

**现象与依据：** 训练 Manifest 已记录 `loop_seconds`、`seconds_per_step`，CUDA 运行还记录峰值 allocated/reserved；`completed_row()` 和汇总 CSV 没有导出这些字段。汇总中 `cuda_peak_allocated_bytes` 来自推理 Benchmark，不能用它判断多层 Bridge 的训练激活成本。静态检查可直接确认，未声称本机测得 CUDA 数值。

**影响：** 原始证据没有丢失，但横向比较需要手动回读训练 Manifest，容易把推理峰值误作训练峰值，削弱新增参数/额外计算成本的统一审查。

**最小必要修复：** 导出明确区分的 `training_loop_seconds`、`training_seconds_per_step`、`training_peak_allocated_bytes`、`training_peak_reserved_bytes` 与 `inference_*` 字段；保留当前“训练循环包含验证与 Checkpoint IO”的计时口径，CPU 显存留空。恢复运行须标明本段实际更新数及其覆盖区间，不能用恢复末段时间冒充完整训练预算。

**补测与验收：** 使用小 JSON fixture 给训练与推理指标赋不同值，检查来源、CSV/JSON 一致性、CPU null 和恢复区间。无需运行 GPU 即可完成工程测试；实际性能比较仍待 G3/G5。

## 三、数学、架构与梯度复核

### 1 共同契约与 Fixed / Last

令 $H_i\in\mathbb{R}^{B\times T\times D}$ 为第 $i$ 个零基 Qwen decoder block 的原始输出。Backend 在指定 block 后注册临时 hook，末层也在模型最终 RMSNorm 之前。中间层对应 HF `hidden_states[i+1]`；最后一个 HF tuple 元素已经做过最终 Norm，不能把它当作原始末层。当前实现没有混入 embedding output。

所有候选先得到 $F$，再执行同一条路径：
$$C=M\odot\operatorname{RMSNorm}(FW^\top),\qquad W\in\mathbb{R}^{768\times2048}.$$
`M[B,T]` 广播到输出通道，Norm 后清零 padding，输出 `[B,T,768]`。prefill 使用当前片段长度，decode 为 `T=1`；Qwen/Talker attention 继续使用完整 Cache 历史 mask。融合不跨 token，不引入未来信息。

Fixed 的 Linear（无 bias）→RMSNorm→mask 与旧版本逐行一致，默认仍用原 `bridge_layer=13`；Last 取原生 Block 27。显式改配置的层号是实验变量。`TalkerModule(projected_condition=True)` 的 `embed_proj` 为 Identity，2048→768 只有 Bridge 一次。Mimi、Delay、STOP/PAD 与 4 层/768/8-Codebook 结构未在 R5 改写。

### 2 Weighted 的独立推导

$$\alpha_i=\frac{e^{a_i}}{\sum_j e^{a_j}},\qquad F=\sum_i\alpha_iH_i.$$
`layer_logits` 是长度 $K$ 的 Parameter，Softmax 在 `dim=0`，归一化的是层而非 batch/token。初始化 $a_i=0$，因此均匀加权。令 $u_{bt}=\partial L/\partial F_{bt}$，已经包含公共投影、Norm 与 mask 的链式梯度，则
$$\frac{\partial L}{\partial a_i}=\alpha_i\sum_{b,t}u_{bt}^{\top}(H_{i,bt}-F_{bt}),\qquad \frac{\partial L}{\partial H_i}=\alpha_i u.$$
代码的标量广播与此一致，`zip(..., strict=True)` 拒绝权重数量与层数量不符。配置层号独立、顺序确定，Backend 返回相同时间窗、设备和 dtype；Omni 在进入 Bridge 前对齐 Talker dtype/device。低精度 Softmax 在 FP32 计算，再转换到 hidden dtype。

### 3 Gated 的独立推导

$$q=[H_1;H_2],\qquad g=\sigma(w^\top q+b),\qquad F=H_2+g(H_1-H_2).$$
$g$ 的形状为 `[B,T,1]`，是逐 token 标量，在通道上广播。权重 $w$ 有 $2D$ 个元素，bias 一个；全部初始化为零，初始 $g=0.5$，没有初始饱和。令
$$d=g(1-g)\,u^\top(H_1-H_2),$$
则
$$\frac{\partial L}{\partial w}=\sum_{b,t}d_{bt}q_{bt},\qquad \frac{\partial L}{\partial b}=\sum_{b,t}d_{bt},$$
$$\frac{\partial L}{\partial H_1}=gu+dw_1,\qquad \frac{\partial L}{\partial H_2}=(1-g)u+dw_2.$$
与实际 Linear→Sigmoid→融合实现一致。Sigmoid 低精度输入转换为 FP32 后求值；Linear 本身仍按模型/AMP 精度执行。两层完全相同等退化输入可合理产生零融合梯度，不能要求所有输入上的每个参数梯度永远非零。正式训练后的饱和风险需观测，当前没有证据说明必须改公式。

### 4 独立数值证据与参数量

临时数值脚本在进程内用 Identity 替代公共 proj/norm，隔离新融合公式；float64、seed 1108、带 padding 的 `[2,3,4]` 小张量，对全部融合参数作步长 $10^{-6}$ 的中央差分。另以完整 Tiny HF Qwen3 + Talker 检查因果与左 padding。公共 RMSNorm 的既有 FP32 实现由原回归覆盖；这里不声称整个 Bridge 通过 float64 gradcheck。

| 检查 | Weighted | Gated |
| --- | ---: | ---: |
| 独立前向公式最大误差 | 1.11e-16 | 2.22e-16 |
| 独立解析梯度最大误差 | 2.22e-16 | 2.22e-16 |
| 中央差分最大误差 | 1.11e-10 | 8.07e-10 |

差分预设阈值 `1e-8`，未放宽。四候选改变未来 token 对前三步 text/audio logits 的影响均为 **0**；左 padding 最大误差依次为 Fixed `2.98e-7`、Last `3.58e-7`、Weighted `4.17e-7`、Gated `2.38e-7`，均小于预设 `1e-5`。

| 原生 Bridge | 参数量 | 相对 Fixed 增加 |
| --- | ---: | ---: |
| Fixed / Last | 1,573,632 | 0 |
| Weighted（两层） | 1,573,634 | 2 |
| Gated | 1,577,729 | 4,097 |

公共参数为 $768(2048+1)$，Weighted 加 $K$，Gated 加 $2D+1$。新融合参数使用零初始化，不消耗额外随机数；同 seed 下公共权重一致。它们归入原 `bridge` optimizer group，实际测试验证非零梯度及更新。

## 四、训练与实验框架判断

| 事项 | 判断及限制 |
| --- | --- |
| 冻结/梯度 | 正确。Qwen 主权重 `requires_grad=False`，没有整体 `no_grad()` 截断；Projector 和启用的 LoRA 可回传。`.eval()` 不关闭 autograd；LoRA dropout 限定 0。 |
| Optimizer | 只收集预期可训练参数，Bridge 融合参数在原分组内；T2A 与 Joint 更新范围区分。 |
| 四任务与 Loss Mask | R5 没有改原标签/损失数学；全仓数据、Loss、混合任务测试通过。A2A 仍限复述诊断，可信 QA 数据依赖未被取消。 |
| 初始化 | 同一 Fixed 来源共享 Qwen/LoRA、Projector、Talker、投影与 Norm；融合参数保留零值，新 optimizer。不同 Bridge 初始函数本来就不同，必须重训练，不能直接比较未训练语音。 |
| Checkpoint/Resume | 同源码/配置下的完整保存、新进程恢复测试通过，包含模型、Optimizer、Scheduler、Scaler、RNG、采样游标；跨 Bridge 配置拒绝。源码身份完整性受 P1-02 限制。 |
| Cache/Streaming/Stop | 当前接口及已有停止、截断、PCM/Cache 回归通过。多层 hook 只保留所选层，按请求顺序返回，退出释放；未改变 Talker 输入协议。 |
| 数据与候选公平性 | 四候选只改变 Bridge；同资产、当前 split、种子、预算、终点权重与生成协议已有检查。但 B1 历史留出隔离缺 P1-01，执行源码锁缺 P1-02。 |
| CLI/Runbook | 当前 `inspect/prepare/smoke/train/resume/evaluate/demo/profile` 与 `bridge_ablation plan/collect` 均存在；G5 Runbook 的命令名和参数对应实际入口。CUDA 命令明确待验证。 |
| 报告事实边界 | R5 报告如实记录两步 Tiny、预算截断、空 ASR、CER/WER=1.0；没有把它们当成音质改善或正式 G5。 |

资源方面没有发现新增的全 Backbone 副本驻留 GPU：新增代价主要是保存被选层激活及 Gated `[B,T,2D]` 拼接。现有恢复会预检、正式加载两次完整 Checkpoint，评估还会加载含 Optimizer 的完整文件；原生模型以 FP32 主权重配合训练 AMP。它们有 IO/主机内存成本，但没有本轮证据证明 H100 必然不足，不将估算写成实测故障。

## 五、实际测试与证据边界

### 1 本轮复验

所有命令在项目根目录执行，使用 `.venv`，无下载、无正式训练。普通测试运行的 Tiny 更新属于已授权的验证。

```bash
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
OMP_NUM_THREADS=4 MPLCONFIGDIR=/private/tmp/omni-r5-review/mpl \
  .venv/bin/python -m pytest -q -p no:cacheprovider \
  --basetemp=/private/tmp/omni-r5-review/pytest
.venv/bin/ruff check --no-cache omni tests scripts
.venv/bin/ruff format --check --no-cache omni tests scripts
.venv/bin/python -m pip check
git diff --check
.venv/bin/python scripts/run_experiment.py --help
.venv/bin/python scripts/bridge_ablation.py --help
.venv/bin/python scripts/bridge_ablation.py plan --help
.venv/bin/python scripts/bridge_ablation.py collect --help
```

| 实际检查 | 结果 |
| --- | --- |
| 当前完整工作区 pytest | **269 passed, 1 skipped in 32.80s**；MPS 在本轮沙箱不可用，未复用 Sol 的 270 全通过计数 |
| 数学/Backend/Omni 定向 | **42 passed in 3.26s** |
| R3/R4/R5 训练及 Adapter 定向 | **44 passed in 22.74s** |
| 消融/原评测定向 | **17 passed in 2.11s** |
| HEAD 纯 Git 归档回归 | **262 passed, 1 skipped in 32.83s**；不含用户未提交测试 |
| Ruff / format | 通过；72 个文件无需格式修改 |
| pip check | 无损坏依赖；pip 缓存目录不可写警告不影响结果 |
| 四个 CLI help | 全部 exit 0 |
| git diff --check | 通过 |

归档初次运行没有 `.git` 元数据，产生 **4 failed, 258 passed, 1 skipped**，四项均在记录 `git rev-parse HEAD` 时失败。随后只通过 `GIT_DIR` 提供原仓库只读 Git 元数据，没有修改测试或源码，重新完整执行得到上述 262/1。原始失败日志也保留，没有把环境失败隐去。

归档复验的实际方式：

```bash
mkdir -p /private/tmp/omni-r5-review/commit
git archive a51b560 | tar -x -C /private/tmp/omni-r5-review/commit
cd /private/tmp/omni-r5-review/commit
GIT_DIR=/Users/lucius/projects/personal/Omni++/.git \
PYTHONPATH=/private/tmp/omni-r5-review/commit \
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=4 MPLCONFIGDIR=/private/tmp/omni-r5-review/mpl \
  /Users/lucius/projects/personal/Omni++/.venv/bin/python -m pytest -q \
  -p no:cacheprovider --basetemp=/private/tmp/omni-r5-review/commit-pytest-git
```

独立探针从项目根目录运行：

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 \
  .venv/bin/python /private/tmp/omni-r5-review/math_probe.py
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  .venv/bin/python /private/tmp/omni-r5-review/cross_stage_leak.py
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 \
  .venv/bin/python /private/tmp/omni-r5-review/source_identity_probe.py
```

探针退出 0 表示复现断言成立，其中两个缺陷探针断言的是“当前错误输入被接受”，不是缺陷已修复。临时脚本是本机复验材料，不是已提交的永久测试；上节已给出应加入仓库的验收断言。

### 2 未重复执行与保护核验

本轮没有重新加载真实 1.7B 权重或执行 CUDA。已检查 Sol 提交的 `experiments/r5-local/20261108-cpu-fp32/acceptance.json`：真实 1.7B 的证据是锁定 revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`、19-token CPU FP32 前向，Block 13、原始末层、末层 Norm 后及 logits 误差均为 0。此处是已有证据复核，不能写成本轮独立 GPU/全尺寸反向验收。独立数学测试使用真实 HF Qwen3 类的 Tiny 尺寸，并非替代 Qwen 的假 Backend。

审查结束核对进入时 197 个文件状态均未变化；P0–P5 的 SHA-256 也与已提交 R5 `verification.json` 完全一致。R0–R5-local、旧实验、Checkpoint 与用户未提交改动均保持原样。仅新增本报告，未 Commit、未 Push、未进入 R6-L。

## 六、Deferred：需等待 B1 / G3 / G5 的决定与风险

| 事项 | 何时决定 / 验证 |
| --- | --- |
| 最终候选层、层数、是否保留 Gated、种子数 | B1 成立后由用户确定；当前 YAML 只是模板，不预设最优方案 |
| 更新 Bridge 还是同时更新 Talker/LoRA、训练预算与学习率 | 固定共同 B1 后制定；记录初始化、额外参数、实际更新数及成本 |
| 1.7B + 4×768 的 BF16 前后向、原生恢复、峰值与吞吐 | G3/G4/G5 实测；本轮 Tiny/FP32 与已有真实前向不能代替 |
| Gate 饱和、层权重坍缩、训练稳定性 | 正式运行记录 gate/权重分布和梯度；不提前改激活或声称提升 |
| CER/WER、人工试听、自然停止、质量指标和统计区间 | B1 具备可靠输出并锁定评测后执行；当前单条、截断、空 ASR 无法排序 |
| 时延预热、重复次数、长度归一化、TTFA/RTF | G5 统一协议；当前总模型流时间无预热且排除 Mimi/ASR，不等于端到端语音延迟 |
| 真实 QA A2A 与原生联合训练 | 取得可信目标语音与合法数据后；echo 不等于 QA |
| gradient checkpointing、编译或新 HF 版本 | 当前入口明确拒绝未验收 gradient checkpointing；若后续启用须重验 hook、反向与 Cache |

这些是后续实验条件，不应通过本轮扩大架构、启动云训练或放宽旧断言来“解决”。

## 七、可直接交给 Sol 的修复清单

```text
仅修复 R5-L 独立审查指出的问题，不进入 R6-L，不运行正式 G5。

1. [P1-01，必须] 补 B1→G5 跨阶段数据身份与 split 检查。
   绑定可核验的 B1/继承组件数据来源；拒绝已训练数据进入目标 val/test、
   已用于选择的数据进入独立 test，并保护封存 test 的角色。
   在 plan 和实际热启动都校验，身份缺失不能认证正式 G5。
   补媒体/文本/跨任务/异 ID 同内容/传递分组的拒绝测试与独立扩充正例。

2. [P1-02，必须] 补齐完整执行源码 fingerprint。
   至少覆盖 LegacyOmni.stream_generate、Collator、Config 等实际依赖；
   结合现有缓存实现/processor身份，矩阵、Checkpoint、train/eval统一。
   新增身份schema并让错误/缺失身份明确拒绝；旧实验和旧checkpoint不回写。
   补临时改生成、Collator、Config后 Resume/Benchmark/collect拒绝的测试。

3. [P2-01，建议本轮一并完成] summary JSON/CSV 同时导出训练和推理成本，
   明确计时范围、恢复区间、CPU null；用不同训练/推理值验证字段来源。

4. 复跑相关测试及全仓回归、Ruff、格式、依赖检查。
   保持Bridge公式、Talker结构、Loss/Delay协议不变，不弱化旧测试。
   保护P0–P5、R0–R5-local、已有实验和用户未提交文件。
   新建修复报告，列文件/测试/修复证据；如用户授权提交，只本地提交不Push。
   完成后等待复审，不提前开展云端实验。
```

数学正确性通过；梯度/冻结机制通过；同配置 Tiny 恢复通过。未来 G5 的训练入口和候选模块已经存在，**公平比较的工程签收仍需完成 P1-01、P1-02**。本轮到此停止，等待用户决定是否授权 Sol 修复。
