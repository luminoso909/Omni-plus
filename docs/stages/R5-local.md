## 一、本地验收范围

R5-L 交付 Fixed、Last、Weighted、Gated 四种 Bridge，沿用 Qwen Omni、SFT、Optimizer、Checkpoint 与生成入口。G5 正式消融和 B1 质量验收仍待执行。本地诊断使用已有授权样本、Tiny 模型及只读的真实 1.7B 前向对照。

起点 `f3b593f`。已审查 AGENTS、主规划、v2.1 手册、数据决策、R2–R4-L 报告及现有模型、训练和测试。在该提交的独立归档重跑 `tests/test_r4_training.py tests/test_r3_training.py tests/test_qwen_omni.py tests/test_qwen_backend.py`，得到 **35 passed in 21.09s**。R4-L 的 Tiny 训练、数据版本、LoRA 与恢复接口满足本阶段前置条件。

## 二、Bridge 契约与数学

YAML 使用 `model.bridge: {kind: ..., layers: [...]}`。层号是零基 Qwen decoder block。Fixed 缺省时使用原 `bridge_layer`。Last 取末 block 的原始输出，位置在 Qwen 最终 RMSNorm 之前。原生 1.7B 有 28 层，末层索引为 27。中间 block $i$ 对应 HF tuple 的 `hidden_states[i+1]`。HF tuple 最后一项经过模型最终 Norm，原始末层则从该 Norm 的输入独立验证。Backend 只为请求层注册临时 hook，按配置顺序返回当前时间窗，调用退出后释放。 [Transformers v4.57.6 Qwen3 实现](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen3/modeling_qwen3.py)

设每层 $H_i\in\mathbb{R}^{B\times T\times D}$，原生 $D=2048$。prefill 的 $T$ 为当前 prompt 序列长度，单步 decode 的 $T=1$。Bridge 收到当前片段的 `valid[B,T]`。Qwen 和 Talker 的注意力仍使用完整历史加当前片段 mask。

| 类型 | 融合结果 $F$ | 融合参数初始化 | Tiny / 原生 Bridge 参数量 |
| --- | --- | --- | ---: |
| Fixed | $H_i$，默认原生 Block 13 | 无 | 1,056 / 1,573,632 |
| Last | $H_{27}$，原生末 Block | 无 | 1,056 / 1,573,632 |
| Weighted | $\sum_i \operatorname{softmax}(a)_i H_i$，2–4 层 | $a_i=0$，初始均匀 | 1,058 / 1,573,634（2 层） |
| Gated | $gH_1+(1-g)H_2$，$g=\sigma(w^\top[H_1;H_2]+b)$ | $w=0,b=0$，初始各半 | 1,121 / 1,577,729 |

各方案共同执行 $C=\operatorname{Mask}(\operatorname{RMSNorm}(FW^\top))$，$W\in\mathbb{R}^{768\times2048}$。无 bias 投影和 768 个 Norm scale 合计 $768(2048+1)$。补齐位置在 Norm 后置零。Talker 接收 `[B,T,768]`，内部 `embed_proj` 为 Identity。Tiny 使用 32→32，保持 4 层 Talker 和 8 路码本。原生 meta 构造验证 4×768 Talker 与上述参数量。

Weighted 使用长度 $K$ 的全局可学习 logits，Gated 使用每 token 一个标量和 $2D+1$ 个参数。低精度路径以 FP32 计算 Softmax 或 Sigmoid，再对齐隐藏态 dtype/device。融合发生在投影之前。梯度用独立 Softmax Jacobian 和 Sigmoid 链式公式对照，双精度容差 $10^{-12}$。Fixed 的默认配置、权重键 `bridge.proj.weight` 与 `bridge.norm.weight`、前向结果均有精确回归。同 seed 构造四种 Bridge 时共享初始化逐 Tensor 一致。

真实 Qwen3-1.7B 使用锁定 revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`，CPU FP32，19 tokens。两层输出均为 `[1,19,2048]`。Block 13、原始 Block 27、末层 Norm 后状态及 logits 对 HF 参考的最大绝对误差均为 **0**。

## 三、训练、初始化与实验身份

候选融合参数归入原有 `bridge` optimizer group。T2A 保持 Qwen 冻结，更新 Bridge、Talker、音频 Embedding 与 Heads。Joint 沿 R4 配方更新指定 Projector 和 LoRA，Qwen 主权重保持冻结。混合任务测试覆盖 I2T、A2T 和音频输出监督，以及穿过冻结 Qwen 的上游梯度。Cache/Streaming 测试使用确定性音频头 fixture，确保缓存开关两边都实际比较 4 帧 `[1,8,1]` 声码事件。

G5 从同一份 Fixed B1 完整训练 Checkpoint 和来源 run manifest 建立共同初始化。核验 Checkpoint SHA、Qwen/tokenizer/媒体资产、非 Bridge 架构、LoRA 配置和共享 Tensor 键、shape、dtype 与有限值。Fixed 控制组须保留来源层。所有方案继承 Qwen/LoRA、Projector、Talker 与 Bridge 投影/Norm，新增融合参数保持零初始化。使用新 optimizer 和游标开始候选训练。

同候选 Resume 继续使用 R4 完整状态。R5 加入 Bridge 选层、源码、模型 YAML 与初始化来源的版本绑定。跨类型或关键配置错误的 Checkpoint、组件导入和独立 Adapter 会拒绝。历史 R4 全状态恢复使用 `f3b593f`，R5 的显式热启动接受满足契约的 R4 权重来源。

矩阵只替换 `model.bridge`。计划记录共用数据、样本顺序、预算、初始化和 ASR 摘要。每组用独立目录保存原始指标、收敛 SVG、模型状态与失败样本。汇总校验相同总步数、训练终点 Checkpoint、硬件、数据、样本与生成协议，保留不匹配、缺失和失败行。G5 手册中的层号是待选模板。 [G5 Runbook](../cloud/G5_BRIDGE_RUNBOOK.md)

## 四、实测命令与结果

Tiny 训练复用 R3 缓存中的 6 条 T2A 训练、1 条验证和 1 条独立留出 T2A。R3 的全任务缓存为 24/4/4 个训练、验证、测试视图。语音目标由锁定 Mimi 重编码，A2A 保持 echo 诊断。来源、划分、许可与摘要沿用 R3 记录。 [R3-L 数据证据](../../experiments/r3-local/20261108-cpu-fp32/data-manifest.json)

以下从仓库根目录执行，示例使用全新结果目录。若需要本地 SenseVoice，把 `evaluation.asr_dir` 设置为已有本地目录并在矩阵生成前锁定。

```bash
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
python scripts/run_experiment.py --help
python scripts/bridge_ablation.py --help
python scripts/bridge_ablation.py plan --base-config configs/v2/r5_local.yaml --candidates configs/v2/r5_candidates_local.yaml --output-dir outputs/r5-local/recheck
MATRIX=outputs/r5-local/recheck
for ID in fixed last weighted gated; do
  CFG="$MATRIX/configs/$ID.yaml"
  RUN="$MATRIX/runs/$ID"
  python scripts/run_experiment.py smoke --config "$CFG" --dry-run
  python scripts/run_experiment.py smoke --config "$CFG" --run-dir "$RUN/train"
  python scripts/run_experiment.py demo --config "$CFG" --resume "$RUN/train/step-000002.pt" --run-dir "$RUN/benchmark"
done
python scripts/bridge_ablation.py collect --matrix "$MATRIX/matrix.json" --output-dir outputs/r5-local/recheck-summary
```

实际矩阵在 `outputs/r5-local/matrix/`，基于 `r5_local.yaml` 加本地 SenseVoice ASR 路径。Mac arm64，CPU FP32，Python 3.12.14、Torch 2.15.0、Transformers 4.57.6，4 线程，seed 42。各组训练 2 步，冻结参数摘要逐组一致，可训练参数更新。以下数值仅用于闭环核对。

| 方案 | 第 2 步训练 Loss | 留出 Loss | 融合参数最大绝对梯度 | 单条模型流时延 |
| --- | ---: | ---: | ---: | ---: |
| Fixed | 20.758842 | 21.694904 | 无融合参数 | 35.56 ms |
| Last | 20.841659 | 21.838911 | 无融合参数 | 36.74 ms |
| Weighted | 20.801406 | 21.760502 | logits 0.0165442 | 38.80 ms |
| Gated | 20.801402 | 21.760506 | weight 0.000687914，bias 0.0165464 | 35.17 ms |

所有候选在相同留出 prompt 上生成 16 帧，Mimi 解码为 30,720 个 24 kHz samples。四组都触达 24 步预算，文本未 EOS，ASR 返回空转写，实际 CER/WER 均为 1.0。时延为无预热、设备同步的完整模型声码流，排除 Mimi 解码、ASR 和文件 IO。CPU 显存字段留空。这组单样本、两步数据不支持音质或速度排序。

Weighted 的连续 2 步与新进程 1+1 步完整 Checkpoint 一致。比较了 **395 个 Tensor 叶节点**及所有非 Tensor 状态，`rtol=atol=0`。Weighted Checkpoint 尝试以 Gated 配置恢复时退出 1，未创建运行目录。未提供 B1 的原生矩阵同样退出 1。空运行目录汇总保留缺失行，完成后四个运行的配置、模型与 Checkpoint 摘要及样本身份通过汇总核验。

另用 `outputs/r5-local/warm-joint.yaml` 从真实 R4-L Tiny Joint 终点和来源 manifest 导入 Weighted `[0,1]`。导入时 134 个共享 Tensor 与来源完全一致，新增融合 logits 为零。两步联合训练使用 24 条训练视图，LoRA、两个 Projector、Bridge 和 Talker 相关组均有非零梯度，冻结组摘要一致。这个来源用于验证迁移机制，正式 B1 尚未成立。

| 检查 | 命令或口径 | 实际结果 |
| --- | --- | --- |
| Bridge 数学和 R1/R2 定向 | `python -m pytest -q tests/test_r5_bridge.py tests/test_qwen_backend.py tests/test_qwen_omni.py` | 42 passed |
| 训练与恢复 | `python -m pytest -q tests/test_r5_training.py` | 11 passed |
| 矩阵与原评测 | `python -m pytest -q tests/test_r5_ablation.py tests/test_r4_evaluation.py` | 17 passed |
| 全仓库离线主机回归 | `MPLCONFIGDIR=/private/tmp/omni-r5-mpl python -m pytest -q`，含 MPS 基础测试 | **270 passed in 30.35s** |
| 干净归档 | `git archive f3b593f` 加 16 个 R5 白名单源码、测试和 YAML，`PYTHONPATH` 指向归档 | **262 passed, 1 skipped in 31.45s**，沙箱无 MPS |
| 静态与依赖 | `ruff check omni tests scripts`、`ruff format --check omni tests scripts`、`python -m pip check`、`git diff --check` | 全部通过，72 文件格式通过 |

实际 CLI、通过/拒绝结果、数据摘要和测试记录见 [验收 JSON](../../experiments/r5-local/20261108-cpu-fp32/acceptance.json)。可比较性检查的本地输出见 [CSV](../../experiments/r5-local/20261108-cpu-fp32/summary.csv)。原始 JSON/CSV、收敛曲线、WAV、完整 Checkpoint 及各命令日志保留在 `outputs/r5-local/`。

## 五、文件级交付与历史保护

| 路径 | 变化 |
| --- | --- |
| `omni/models/bridge.py`、`omni/models/config.py`、`omni/models/model_omni_v2.py` | 候选数学、配置、统一条件接线与原 Fixed 键兼容 |
| `omni/models/thinker/backend.py`、`omni/models/thinker/qwen_backend.py` | 保留单层 API，增加按顺序取指定层的接口 |
| `omni/training/r3_local.py` | 复用现有训练与恢复，加入来源导入、严格身份、选层、梯度日志和 held-out benchmark |
| `omni/evaluation/sft.py`、`omni/evaluation/bridge_benchmark.py` | 原生成函数增加可选计时，统一样本、JSON/CSV、ASR、失败、延迟及显存字段 |
| `scripts/bridge_ablation.py` | 计划与汇总，训练仍由 `scripts/run_experiment.py` 调用 |
| `configs/v2/r5_{local,h100}.yaml`、`configs/v2/r5_candidates_{local,h100}.yaml` | 本地与原生 CUDA 候选模板 |
| `tests/test_r5_{bridge,training,ablation}.py` | 独立数学、模型兼容、训练恢复和矩阵审查 |
| README、AGENTS、主规划、待办、两份 v2.1 执行文档 | 更新阶段状态及入口 |
| 本报告、`docs/cloud/G5_BRIDGE_RUNBOOK.md`、`R5-L_REVIEW_HANDOFF.md` | 本地证据、待验 CUDA 流程及独立审查交接 |
| `experiments/r5-local/20261108-cpu-fp32/` | 小规模验收、矩阵 CSV 和历史 SHA 证据 |

初始 Git 状态和文件 SHA 保存在 `outputs/r5-local/audit/`。逐项核对 P0–P5、R0–R4-L、原有实验和既有未提交文件。阶段报告 SHA 见 [保护证据](../../experiments/r5-local/20261108-cpu-fp32/verification.json)。模型、缓存、预测音频和原始数据保留在忽略目录。

## 六、最终修复验收补充（2026-11-08）

根据用户本轮授权追加本节，前五章保留原验收事实。基于 `6cd4988` 完成两项剩余 P1 的复现、修复和正反向回归，最终结论 **PASS**。完整媒体组身份随审计契约 v3 保存，源、目标与祖先缺少证据时拒绝正式 G5。源码身份 v3 校验实际导入包、子模块及搜索路径，旧 v2 矩阵不能继续认证正式比较。P2 成本代码保持原样。

当前工作区全仓 **319 passed, 1 skipped**，仅含本次代码的干净归档 **312 passed, 1 skipped**，跳过项均为沙箱 MPS 不可用。Ruff、格式和依赖检查通过。真实 Tiny CLI 的正确源码路径完成规划、1 步更新、新进程恢复到 2 步及评测，错根路径在读取资产前拒绝。原有数学、冻结、Checkpoint、Cache 与 Streaming 测试通过。

历史报告、旧实验与 Checkpoint 保留，旧严格恢复使用原提交。正式 B1 应从完整身份缓存开始训练。当前停在 R6-L 入口，G3–G7 的云端实测与正式训练仍未执行。文件清单、反例证据、版本迁移要求与限制见 [最终修复报告](../reviews/R5-L_FINAL_FIX_REPORT.md)。
