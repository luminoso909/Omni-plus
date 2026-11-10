## 一、范围与启动条件

本手册对应 R6-L 的真实入口 `scripts/run_experiment.py → omni.training.r6_local`。本地只验收 Tiny GRPO 的概率、梯度、更新和恢复。**以下 H100、CUDA、B1 正式 RL 与质量比较均待云端验收**，本手册不构成启动云资源或正式训练的授权。执行顺序仍为 G3 资源实测、G4 冻结 B1、用户确认 G6 数据和 Reward、单卡 Pilot、正式实验。

| 输入 | 启动前必须固定 |
| --- | --- |
| 代码 | 包含 R6-L 最终验收的完整 Commit；同 checkout 安装环境 |
| 模型 | Qwen3-1.7B 非 Base 的固定 revision、原生 Tokenizer/模板、媒体资产摘要 |
| 初始策略 | 已验收 B1 完整 Checkpoint、对应 Manifest、递归继承组件和数据谱系 |
| 数据 | 有许可和可信标签的审计 JSONL、原始媒体摘要、v3 身份组、固定 split |
| 训练 | Seed、模式、可训练范围、group、生成长度、温度、KL、预算与硬件 |
| 对照 | 同一 B1、验证/测试集、采样协议、Reward/评测器版本和推理预算 |

优先使用 1×H100 80GB；单卡是否足够依据 G3/G4 和本节 Pilot 的实际峰值确定。数学定义、有效动作 Mask 和 Reference 约定见 [GRPO 数学说明](../GRPO_MATH.md)。

## 二、固定 Commit 和 CUDA 环境

先将 `R6_COMMIT` 设为阶段交付的完整 SHA。当前提交若尚未 Push，须由负责人发布或用 `git bundle` 私下搬运；不能从远程旧提交假装启动 R6。以下命令只在获批云环境执行。

```bash
set -euo pipefail
export R6_COMMIT='填写本次R6-L完整Commit'
git clone https://github.com/luminoso909/Omni-plus.git
cd Omni-plus
git switch --detach "$R6_COMMIT"
python3.12 -m venv .venv
source .venv/bin/activate

# 从已通过 G3 的环境记录填写版本及 CUDA wheel 地址。
export G6_TORCH_VERSION='填写G3通过的Torch版本'
export G6_TORCHAUDIO_VERSION='填写对应Torchaudio版本'
export G6_TORCH_INDEX='填写G3通过的官方CUDA索引URL'
python -m pip install "torch==$G6_TORCH_VERSION" "torchaudio==$G6_TORCHAUDIO_VERSION" --index-url "$G6_TORCH_INDEX"
python -m pip install -e '.[dev,media,encoders,plots,data]'
python -m pip check
python scripts/run_experiment.py --help
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); assert torch.cuda.is_available()'
nvidia-smi
mkdir -p outputs/g6/environment
python -m pip freeze > outputs/g6/environment/pip-freeze.txt
nvidia-smi > outputs/g6/environment/nvidia-smi.txt
git rev-parse HEAD > outputs/g6/environment/commit.txt
```

Python 必须为 3.12；依赖边界以 `pyproject.toml` 为准。Torch 与 Torchaudio 使用相容版本，按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 选择与驱动匹配的 CUDA wheel，不搬运 Mac 环境。上面的版本变量有意留给 G3 实测，不能用未经验证的版本填写“已通过”。

CLI 在读取配置前检查实际导入的 `omni` 包与子模块来源，Manifest 另锁定源码文件摘要与运行时。另一 checkout 的 editable install 会被拒绝；应在当前 checkout 重新安装，不能删除校验或只修改工作目录。

## 三、模型、B1 和数据搬运

### 1 模型资产

优先从 G4 已核验的私有资产目录复制。目标是新云 checkout，已有目录不覆盖：

```bash
export G4_ASSET_ROOT='填写已核验G4资产绝对路径'
test ! -e weights/qwen3-1.7b
test ! -e weights/media-reference
mkdir -p weights/qwen3-1.7b weights/media-reference
rsync -a --checksum "$G4_ASSET_ROOT/qwen3-1.7b/" weights/qwen3-1.7b/
rsync -a --checksum "$G4_ASSET_ROOT/media-reference/" weights/media-reference/
```

媒体根目录下需有 `model/SenseVoiceSmall`、`model/siglip2-base-p32-256-ve` 和 `model/mimi`，保留已验收内容及许可。若 Qwen 尚未搬运，获批后可在**尚无该模型的目录**使用以下终端命令下载官方固定版本；本轮没有执行该下载。

```bash
hf download --help
hf download Qwen/Qwen3-1.7B --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --local-dir weights/qwen3-1.7b
```

`hf download` 参数已与本地所装 CLI 的 `--help` 核对。`configs/assets_v2.yaml` 锁定模型、Tokenizer、模板、许可证及文件 SHA；准备入口逐项检查。媒体模型不能用同名但内容不同的权重替代。

### 2 B1 与审计数据包

准备经批准的私有目录，包含 `b1/`、`data/` 和 `SHA256SUMS`。`b1/` 放完整终点 Checkpoint、原始 `manifest.json`、`lineage.json` 及其递归依赖；`data/` 放审计 JSONL、媒体与许可/来源记录。摘要清单应由资产发布方提供，不能在接收后重新生成摘要冒充原始可信校验。

```bash
export G6_APPROVED_BUNDLE_DIR='填写已批准私有包绝对路径'
test ! -e outputs/g6/imported
mkdir -p outputs/g6/imported
rsync -a --checksum "$G6_APPROVED_BUNDLE_DIR/" outputs/g6/imported/
(cd outputs/g6/imported && sha256sum -c SHA256SUMS)
```

若通过下载交付，只下载已批准的有限数据包。以下 URL 与摘要由发布方提供，不是全量数据集下载入口；不要把访问令牌写进 Git 或运行报告。

```bash
export G6_BUNDLE_URL='填写已批准的数据包URL'
export G6_BUNDLE_SHA256='填写发布方提供的SHA256'
mkdir -p outputs/g6/incoming
test ! -e outputs/g6/incoming/audited-bundle.tar.gz
curl --fail --location "$G6_BUNDLE_URL" --output outputs/g6/incoming/audited-bundle.tar.gz
printf '%s  %s\n' "$G6_BUNDLE_SHA256" outputs/g6/incoming/audited-bundle.tar.gz | sha256sum -c -
tar -tzf outputs/g6/incoming/audited-bundle.tar.gz
# 确认归档内容为约定的相对路径后，解到尚不存在的新目录。
test ! -e outputs/g6/imported
mkdir -p outputs/g6/imported
tar -xzf outputs/g6/incoming/audited-bundle.tar.gz -C outputs/g6/imported
(cd outputs/g6/imported && sha256sum -c SHA256SUMS)
```

原始媒体、转写、许可和任务字段按 [G4 数据格式](G4_SFT_RUNBOOK.md) 审计。真实问答 A2A 必须具备可信回答文本及独立目标录音，`a2a_kind: qa`；echo 复述不能当作问答奖励数据。未知版本的现成 codes 不直接混用：沿用锁定 Mimi 生成 8-Codebook，SenseVoice 输入 16 kHz、Mimi 24 kHz。

正式缓存使用审计契约 v3。`identity_groups` 必须保留完整审计源中同一原始媒体的所有别名，含预算未选中和跨任务身份，不能通过换样本 ID 或删除别名重新划分。B1 训练媒体不能进入 G6 validation/test；B1 validation 不能进入 G6 test；历史 test 保持 test。身份不足、组件谱系缺失或摘要不一致时拒绝启动，不能回填旧结果来通过校验。

`lineage.json` 路径相对自身所在目录，以**实际完整 Checkpoint SHA** 为键：

```json
{
  "B1_CHECKPOINT_SHA256": {"cache_manifest": "b1-cache/manifest.json"},
  "PARENT_CHECKPOINT_SHA256": {
    "checkpoint": "parents/component.pt",
    "cache_manifest": "parents/component-cache/manifest.json"
  }
}
```

B1 与各父组件的原始缓存清单须完整留存。正式 B1 从 v3 数据重新训练得到；旧的缺证据缓存、Checkpoint 和 Manifest 不重签、不覆盖。

## 四、先准备数据，再配置 GRPO

### 1 不带 GRPO 键的数据配置

`prepare` 使用已有 SFT 数据入口，不接受 `grpo` 顶层键。创建 `outputs/g6/configs/prepare.yaml`，继承 **R4/G4** 配置，例如：

```yaml
extends: ../../../configs/v2/r4d_h100.yaml
assets:
  qwen_dir: weights/qwen3-1.7b
  reference_root: weights/media-reference
data:
  audited_manifest: outputs/g6/imported/data/audited.jsonl
  audited_manifest_sha256: 填写实际JSONL的SHA256
  cache_dir: .cache/g6/audited-v1
  tasks: {t2a: 20}
  seed: 42
  max_length: 1200
training:
  tasks: [t2a]
  task_weights: {}
  sample_weights: {}
  allow_diagnostic_a2a: false
```

20 条仅为有限 Pilot 示例，当前选样配额为 16 train / 2 validation / 2 test，满足下面 `evaluation_prompts: 2`；样本须为独立媒体组。扩展 I2T/A2T/QA A2A 时同步修改预算和训练任务，仍先审核数据。资产路径、原始数据、processor 版本与缓存合同不一致时，使用新缓存路径重新准备。

```bash
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
python scripts/run_experiment.py prepare --config outputs/g6/configs/prepare.yaml --dry-run
python scripts/run_experiment.py prepare --config outputs/g6/configs/prepare.yaml
```

### 2 三种模式的派生配置

创建 `outputs/g6/configs/text.yaml`，以如下内容替换模板路径和摘要。模型架构、Bridge 类型/选层及完整 LoRA 配置须与 B1 一致；同 rank 不同 alpha 也会被拒绝。

```yaml
extends: ../../../configs/v2/r6_text_h100.yaml
data:
  audited_manifest: outputs/g6/imported/data/audited.jsonl
  audited_manifest_sha256: 填写与prepare相同的SHA256
  cache_dir: .cache/g6/audited-v1
  tasks: {t2a: 20}
  seed: 42
  max_length: 1200
training:
  steps: 10
  batch_size: 1
  accumulation_steps: 1
  seed: 42
  max_seconds: 0.0
grpo:
  policy:
    source: sft
    checkpoint: outputs/g6/imported/b1/step-XXXXXX.pt
    manifest: outputs/g6/imported/b1/manifest.json
    lineage: outputs/g6/imported/b1/lineage.json
  evaluation_split: validation
  evaluation_prompts: 2
```

Audio 和 Joint 分别另建 `audio.yaml`、`joint.yaml`，使用相同数据与 Policy 路径，并将 `extends` 换为 `r6_audio_h100.yaml`、`r6_joint_h100.yaml`。每种模式独立 Run ID。默认 Fixed Bridge；若获批使用已训练 Adaptive Bridge，必须使用对应模型配置及完整来源，不能只替换层号比较未训练语音。

| 模式 | 默认更新范围 | 当前可执行 Reward 模板 |
| --- | --- | --- |
| Text | Qwen LoRA | Token 序列严格匹配、文本结束完整性 |
| Audio | Bridge、Talker、Audio Embedding/Heads；文本策略冻结 | 本地 Mimi 解码后 SenseVoice CER 回转写、完整性 |
| Joint | LoRA、Bridge、Talker、Audio Embedding/Heads | 文本匹配、ASR 回转写、完整性 |

这些权重是可执行模板，**不是已批准的最终研究配方**。`asr_match` 使用 `max(0, 1-CER)`；空音频得到 0，目标为空或资源缺失则报错。ASR 只接收预测音频，参考文本不进入转写器。`token_value/audio_value` 只允许 Tiny，不作为真实 Qwen 的质量奖励。语义、音质等新指标可经 `reward_score(..., evaluators=...)` 显式函数注入；当前 YAML/CLI 不提供任意插件加载器。

可配置 `group_size≥2`、正温度、clip、KL 系数、学习率、累积和生成长度。`top_k=0/top_p=1` 固定；Reference 仅支持初始 Policy。`audio` 禁止更新 LoRA/输入 Projector，保证文本条件概率抵消。Text/Joint 可以按任务启用 Projector，但当前前向不能用 `no_grad()` 切断其穿过冻结 Qwen 的梯度。

`training.batch_size × accumulation_steps × group_size` 决定一轮收集轨迹数；更新逐条反向并用全窗口分母累积。`update_epochs` 控制同一 Old Policy 轨迹的 PPO 更新次数，预算和 `--steps` 必须整除它。这里 `--steps` 始终是**含恢复步数的目标总 optimizer steps**。

沿用 SFT 配置中的 `eval_interval/patience/min_delta/stop_weight/task_weights/evaluation.*` 不控制 GRPO；本阶段没有隐式周期评估或 SFT 早停。RL 评测只通过显式 `grpo-evaluate` 和 `grpo.evaluation_*`，Reward 使用 `grpo.reward.weights`。任务采样仍使用已有 sampler 的 `sample_weights`，不能把它与 Reward 权重混淆。

## 五、本地验收与单卡 Pilot

### 1 本地 Tiny 入口

无大型资产的数学/工程回归使用 fixture；以下命令不下载资产：

```bash
python -m pytest tests/test_r6_math.py tests/test_r6_rollout.py tests/test_r6_training.py tests/test_r6_rewards.py tests/test_r6_integrity.py tests/test_r6_review.py -q
```

下面命令依赖已有 R3/R4 合法缓存和固定资产。**本次原配置引用的外置硬盘未挂载，旧真实资产 Dry Run 因缺少依赖失败，不能将这些原配置命令写成已通过。** 本轮实际 CLI 验收使用隔离、自创合法 Tiny Fixture；具体路径、命令与结果见 [R6 阶段报告](../stages/R6-local.md)。恢复原资产后可运行以下复现入口。若缓存尚未准备，先用 `configs/v2/r4d_local.yaml` 运行上述 `prepare` 流程；自定义路径时，为数据配置和三个 RL 配置设置相同的 `assets/data`，不要向普通 `prepare` 传 R6 YAML。

```bash
python scripts/run_experiment.py grpo-smoke --config configs/v2/r6_text_local.yaml --dry-run
python scripts/run_experiment.py grpo-smoke --config configs/v2/r6_text_local.yaml --run-dir outputs/r6/text-tiny-001
python scripts/run_experiment.py grpo-smoke --config configs/v2/r6_audio_local.yaml --run-dir outputs/r6/audio-tiny-001
python scripts/run_experiment.py grpo-smoke --config configs/v2/r6_joint_local.yaml --run-dir outputs/r6/joint-tiny-001
```

### 2 单卡 1–10 Step 与 Profile：待 CUDA 验收

先用 Dry Run 核对真实资产、可训练参数、数据身份、B1 谱系和上下文预算。它按实际命令选择 train 或配置的评估 split，也检查目标步数、输出目录及传入 Checkpoint 的恢复身份；不执行 forward，不能替代 CUDA 数学验收。

```bash
CFG=outputs/g6/configs/text.yaml
mkdir -p outputs/g6/logs
python scripts/run_experiment.py grpo-smoke --config "$CFG" --dry-run > outputs/g6/logs/text-dry-run.json
python scripts/run_experiment.py grpo-smoke --config "$CFG" --steps 1 --run-dir outputs/g6/text-pilot-001 > outputs/g6/logs/text-pilot-001.log 2>&1
python scripts/run_experiment.py grpo-profile --config "$CFG" --steps 10 --run-dir outputs/g6/text-profile-001 > outputs/g6/logs/text-profile-001.log 2>&1
```

按相同流程分别跑 `audio.yaml` 和 `joint.yaml`，使用全新目录。1-step 例子要求 `update_epochs=1`；其它设置按完整轮次填写。`grpo-smoke` 最多 10 步；`grpo-profile` 最多 100 步，且**实际进行参数更新**。获批 100-step Profile 时先另建预算为 100 的配置，在该配置下启动新 run；不能修改已保存 run 的配置后冒充 Resume。

记录 `loop_seconds/seconds_per_update`、有效更新数、Rollout 数、生成长度和 CUDA `peak_allocated_bytes/peak_reserved_bytes`。计时包含采样、奖励、更新与保存，是端到端循环成本，不能称为纯模型 kernel 吞吐。CPU 上的 ASR/Mimi 奖励也可能主导耗时。

验收要求：Reward/Loss/KL/梯度有限；首次 Old 重算一致；期望参数有更新；冻结参数摘要相同；Checkpoint 在新进程可恢复；预测 Code/STOP/截断有原始记录。Cache 与完整 teacher-forcing 当前使用 `rtol=atol=1e-4`，BF16 若失败须定位，不直接放宽断言。

## 六、正式训练、保存与恢复：待 G6 授权

完成 Pilot 后再锁定正式预算和 Reward。每种模式使用独立配置与目录，分别从同一 B1 开始，不把 Text 的额外训练悄悄加入 Joint 的起点。

```bash
python scripts/run_experiment.py grpo-train --config outputs/g6/configs/text.yaml --run-dir outputs/g6/text-train-001 > outputs/g6/logs/text-train-001.log 2>&1
python scripts/run_experiment.py grpo-train --config outputs/g6/configs/audio.yaml --run-dir outputs/g6/audio-train-001 > outputs/g6/logs/audio-train-001.log 2>&1
python scripts/run_experiment.py grpo-train --config outputs/g6/configs/joint.yaml --run-dir outputs/g6/joint-train-001 > outputs/g6/logs/joint-train-001.log 2>&1
```

中断后用**原配置**、原源码/环境、原资产/数据和新结果目录恢复，例如总预算 10、已经完成 1 步：

```bash
python scripts/run_experiment.py grpo-resume --config "$CFG" --resume outputs/g6/text-pilot-001/step-000001.pt --steps 10 --run-dir outputs/g6/text-resume-001 > outputs/g6/logs/text-resume-001.log 2>&1
```

完整 Checkpoint 保存模型/LoRA、Optimizer、Scheduler、Scaler、Python/NumPy/Torch/设备 RNG、任务/数据游标、配置/源码/资产身份、Reference 快照及 Policy 版本。周期保存须同时满足完整 Rollout 更新边界和 `training.save_interval`，正常终点另保存一次；例如 `update_epochs=2/save_interval=3` 的周期存盘点为第 6、12 步。只恢复完整边界，下一批重新采样；不把更新后的概率写回旧轨迹。Reference 与冻结底座必须仍等于最初 B1。

`adapter.pt` 用于独立模型加载，不含完整训练状态，不能替代 Resume。不同模式/训练范围/Reward/Seed/预算或设备版本的配置不可直接恢复同一 Checkpoint。需要新实验时保留原记录，另行授权明确热启动来源。

整批优势与 KL 梯度均为零时，结果为 `no_update`，不会通过 AdamW weight decay 制造更新，也不推进 step。当前策略是停止并保留诊断，其非完整边界快照拒绝恢复。排查 Reward 饱和、全截断、过短生成或数据问题；不能把 `no_update` 写成成功训练。

## 七、B1 对照与统一评估

B1 和 RL 使用同一配置的 `evaluation_split/evaluation_prompts`、Seed、温度、长度、Group Size 和 Reward 版本。评测统一使用 `training.seed + 1,000,003`，输出 `evaluation_seed/evaluation_record_ids` 供对照。先在 validation 调参，正式测试集只在协议锁定后使用。split 是严格配置身份的一部分，正式 test 实验需在启动训练前锁定，不能事后改原配置再恢复。两次调用都使用全新目录：

```bash
python scripts/run_experiment.py grpo-evaluate --config "$CFG" --baseline --run-dir outputs/g6/text-b1-eval-001 > outputs/g6/logs/text-b1-eval-001.log 2>&1
python scripts/run_experiment.py grpo-evaluate --config "$CFG" --resume outputs/g6/text-resume-001/step-000010.pt --run-dir outputs/g6/text-rl-eval-001 > outputs/g6/logs/text-rl-eval-001.log 2>&1
```

`--baseline` 直接评估配置锁定的初始 B1，不做一次额外训练生成“B1 checkpoint”；其 `step=0`，不写新 Checkpoint 或 Adapter。RL 使用该 run 的完整 checkpoint；两者都不更新参数。Audio/Joint 同理。不同模式的 Reward 标度不直接横向比较，先分别与本模式 B1 对照。

`evaluation.json` 和 `rollouts.jsonl` 是当前评估入口的原始结果，Reward 汇总在 `evaluation.json.metrics`；`metrics.jsonl` 仅训练生成。逐样本保留 Token、声码、Mask、EOS/STOP/截断及 Reward 分项。文本严格匹配不等于一般语义正确；ASR CER 不等于自然度，仍须在 G6/G7 增加固定协议的独立 Benchmark、音质评测和人工试听。当前 Reward API 可返回解码波形给评测器，不承诺 CLI 自动保存每条 WAV。

检查真实 Tokenizer 对未定义词表对齐行的解码行为，空或不可解码输出记录为失败；不得把模型矩阵大小当作全部 Token 都有可读语义。比较时报告额外可训练参数、有效更新数、数据访问量、Reward 调用成本与硬件，不把额外训练收益直接归因于 GRPO。

## 八、结果回传、失败处理与双卡边界

| 文件/字段 | 用途与验收 |
| --- | --- |
| `resolved_config.yaml`、`manifest.json` | 完整配置、资产/源码/数据身份、更新范围、冻结摘要、硬件和计时 |
| `metrics.jsonl` | Reward、KL、Loss、Ratio、clip fraction、梯度、学习率及行为版本 |
| `rollouts.jsonl` | 逐条动作、Mask、Old Logprob、原样本 ID、Reward 分项和停止状态 |
| `step-*.pt`、`adapter.pt` | 完整恢复与独立适配器；SHA 核对且禁止覆盖历史 run |
| `evaluation.json`、外部 Benchmark 结果 | 与 B1 的同协议对照，未测指标留空并说明 |
| `error.json`、stdout/stderr | 异常类型、进度与相关样本；失败不删除、不补造指标 |

把所有运行结果备份到获批私有持久卷，回传前生成实际摘要，接收端重新核验。以下命令在仓库根运行；`G6_RETURN_DIR` 必须是新目标路径。

```bash
find outputs/g6 -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > outputs/g6/SHA256SUMS
export G6_RETURN_DIR='填写新的私有回传目录'
test ! -e "$G6_RETURN_DIR"
mkdir -p "$G6_RETURN_DIR/outputs"
rsync -a --checksum outputs/g6/ "$G6_RETURN_DIR/outputs/g6/"
(cd "$G6_RETURN_DIR" && sha256sum -c outputs/g6/SHA256SUMS)
```

Git 只保存配置、测试、报告和小规模汇总，不提交模型、原始音频、缓存或 Checkpoint。OOM 后保留错误和最后完整 Checkpoint；先降低新实验的 group/batch/生成预算，再重新 Profile。配置变更须新 run，不能静默改变 Resume 条件。`gradient_checkpointing` 尚未实现的开关保持拒绝，不宣称用它降低显存。

当前训练器是单设备执行，没有可靠的 DDP/FSDP/ZeRO 或 Rollout/Training 双卡实现。可选后续设计是 GPU0 冻结行为快照收集、GPU1 更新，显式传递 Policy 版本、动作、完整 Mask、实际行为 Logprob 和输入身份，更新前等待采样屏障；Reference 仍固定。该设计须另行开发、测试通信/同步/恢复并 CUDA 验收，不能用 `torchrun` 启动当前 CLI 冒充已支持多卡。

只有真实单卡 Pilot、跨进程 Resume、数据隔离、B1 公平对照和获批质量指标均取得证据，才能记录 G6 通过。本地 R6-L PASS 不替代这些正式验收。
