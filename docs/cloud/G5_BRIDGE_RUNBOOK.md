## 一、启动条件

R5-L 提供 Fixed、Last、Weighted、Gated 的本地实现和实验入口。G5 需先取得 G3 资源实测和 G4-D 冻结的 B1，再由用户确认候选层、种子、训练范围及预算。以下 CUDA 命令待云端验证。优先使用单卡 H100 80GB，正式设置以 G3/G4 的显存和吞吐记录为准。

| 输入 | 固定项 |
| --- | --- |
| 代码与环境 | R5 审查修复提交、Python 3.12、同一驱动和 PyTorch/CUDA/Transformers 版本 |
| B1 | Fixed 完整 Checkpoint、对应运行清单、各继承组件的 Checkpoint 和原始缓存清单 |
| 资产 | Qwen3-1.7B revision、Tokenizer/模板、SenseVoice、SigLIP2、Mimi |
| 训练 | 同一任务、split、媒体缓存、Seed、初始化、更新范围、LR 与 optimizer step 预算 |
| 评测 | 相同 validation/test 数据、样本数量、贪心生成预算、ASR 和设备 |

## 二、固定代码与环境

将包含 [审查修复](../reviews/R5-L_REVIEW_FIXES.md) 的提交 SHA 填入 `R5_COMMIT`。若远程尚未包含该提交，先由仓库负责人发布对应提交或搬运 Git bundle。

```bash
git clone https://github.com/luminoso909/Omni-plus.git
cd Omni-plus
git switch --detach "$R5_COMMIT"
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,media,encoders,plots,data]'
python -m pip check
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
nvidia-smi
```

使用与云主机驱动匹配的 Linux CUDA PyTorch 和 Torchaudio，并把安装指令、`pip freeze` 与 `nvidia-smi` 输出存入运行目录旁。 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)

## 三、准备 B1 与实验数据

按 G4 手册放置固定 revision 的 Qwen 和媒体资产。把已验收的 B1 终点 Checkpoint 与对应 run manifest 搬到同一私有持久卷，并核对文件 SHA。B1 提供整套共享权重，包括 LoRA、两个 Projector、Talker、Bridge 投影与 Norm。新方案保留同样的共享初始化，Weighted/Gated 的新增融合参数从零初始化，optimizer 与训练游标从头建立。继续同一候选的训练使用 `resume`。

使用审计 JSONL 和固定 split，按许可与预算准备 G5 数据。所有候选共用同一缓存。保留独立 Test，仅在协议与训练预算锁定后用于汇总。联合训练包含问答式 A2A 时，仍须提供可信 QA 目标录音。数据格式和准备规则见 [G4 操作说明](G4_SFT_RUNBOOK.md)。

将 B1 及每个递归继承组件当时的缓存 `manifest.json` 原样归档。用 `shasum -a 256` 读取各 Checkpoint 的摘要，在 `outputs/g5/b1/lineage.json` 中以完整摘要为键登记。路径相对 `lineage.json`。下面的键是占位符，B1 主 Checkpoint 由训练配置指定。

```json
{
  "B1_CHECKPOINT_SHA256": {"cache_manifest": "b1-cache/manifest.json"},
  "COMPONENT_CHECKPOINT_SHA256": {
    "checkpoint": "parents/component.pt",
    "cache_manifest": "parents/component-cache/manifest.json"
  }
}
```

`plan` 和热启动都依据 Checkpoint 内的摘要核对清单，并按已记录的媒体 ID、内容摘要、文本身份与分组检查 split。B1 训练数据必须留在 G5 训练侧，B1 验证数据不得进入 G5 Test，封存 Test 保持原角色。正式矩阵要求所有来源具有审计数据契约，缺少继承证据时须找回对应原始清单及 Checkpoint。

在 `outputs/g5/configs/base.yaml` 创建派生配置。以下路径和摘要为待填值。任务、训练预算和 B1 的 LoRA 配置由 G3/G4 实测确定。

```yaml
extends: ../../../configs/v2/r5_h100.yaml
model:
  initialization:
    bridge_base_checkpoint: outputs/g5/b1/step-XXXXXX.pt
    bridge_base_manifest: outputs/g5/b1/manifest.json
    bridge_base_lineage: outputs/g5/b1/lineage.json
data:
  audited_manifest: outputs/g5/data/audited.jsonl
  audited_manifest_sha256: 填写实际SHA256
  cache_dir: .cache/g5/audited-v1
  tasks: {t2a: 20, i2t: 16, a2t: 16, a2a: 16}
evaluation:
  max_new_tokens: 128
  asr_dir: weights/media-reference/model/SenseVoiceSmall
  benchmark: {split: test, t2a_samples: 2}
```

`configs/v2/r5_candidates_h100.yaml` 给出四组候选模板。Fixed 模板用零基 Block 13，须与 B1 实际选层一致。Last 取最后一个原始 Block 27，两个融合候选共用 `[13,27]`。样本预算 20 为 T2A 保留各 2 条 validation 和 test，审计清单须提供足够的独立媒体。通过 B1 验收后锁定最终 YAML。可为不同 Seed 各建一份独立矩阵，保留样本、训练顺序和执行环境记录。

## 四、独立运行、保存与恢复

下列调用共用实际的训练 CLI。每次调用使用新运行目录。为每个候选依次执行相同流程。示例演示 Fixed 组，其他候选以矩阵生成的 run ID 替换 `fixed`。

```bash
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
python scripts/run_experiment.py prepare --config outputs/g5/configs/base.yaml
python scripts/bridge_ablation.py plan --base-config outputs/g5/configs/base.yaml --candidates configs/v2/r5_candidates_h100.yaml --output-dir outputs/g5/matrix-001

MATRIX=outputs/g5/matrix-001
ID=fixed
CFG="$MATRIX/configs/$ID.yaml"
RUN="$MATRIX/runs/$ID"
mkdir -p "$RUN"
python scripts/run_experiment.py train --config "$CFG" --dry-run
python scripts/run_experiment.py profile --config "$CFG" --steps 2 --run-dir "$RUN/profile"
python scripts/run_experiment.py train --config "$CFG" --run-dir "$RUN/train" > "$RUN/train.log" 2>&1
```

运行产物包括解析后的配置、`manifest.json`、`metrics.jsonl`、`loss.svg`、`step-*.pt`、`best-*.pt` 和 `adapter.pt`。验证曲线供事后核查。矩阵模板使用相同的终点步数作统一对照。失败保留日志与最后有效 Checkpoint。用原配置、原路径资产和新 run 目录继续，同一 optimizer/调度/随机状态及数据游标随 Checkpoint 恢复。以下步数为模板例子，按实际进度替换。

```bash
python scripts/run_experiment.py resume --config "$CFG" --resume "$RUN/train/step-000004.pt" --steps 10 --run-dir "$RUN/resumed" > "$RUN/resume.log" 2>&1
```

Checkpoint 身份校验覆盖 `omni` Python 源码、两个实验 CLI、运行时、配置、数据及初始化文件哈希。历史 Checkpoint 的严格恢复在原提交执行。若版本或设备变化，保留失败记录并从获批的共同 B1 开启新矩阵。定期将权重、日志、配置和清单备份到私有持久卷，使用文件校验确认回传完整。

## 五、统一 Benchmark 与导出

将 `CKPT` 设为该候选成功训练的终点 Checkpoint。`demo` 使用配置中的同一 held-out split 和样本预算，写出 Loss、每样本指标、失败记录和预测 WAV。推理计时口径以结果字段为准，CUDA 显存使用峰值统计。CER/WER 仅在 ASR 实测时可用，人工试听、UTMOS 和语义一致性需另行确认评测协议。

```bash
python scripts/run_experiment.py demo --config "$CFG" --resume "$CKPT" --run-dir "$RUN/benchmark" > "$RUN/benchmark.log" 2>&1
python scripts/bridge_ablation.py collect --matrix "$MATRIX/matrix.json" --output-dir outputs/g5/summary-001
```

对恢复后的运行，在 `collect --artifacts` 的 JSON 中显式指定对应训练目录、评测目录和失败日志。保留缺失或失败行并按原因复核。检查候选对照使用相同的 B1 文件 SHA、资产、数据、种子、有效步数、评测集和硬件。记录新增参数和显存/延迟成本，单独保留各 run 的原始 JSON/CSV、训练曲线和失败样本。仅汇总指标与报告进入 Git，原始语音和 Checkpoint 保留于私有目录。

汇总的 `training_*` 记录优化器循环耗时、训练峰值和实际覆盖的 step 区间。恢复运行只计本段成本。`inference_*` 单独记录模型流延迟与推理峰值。历史 Tiny 矩阵可汇总为诊断，正式比较使用新版本矩阵。
