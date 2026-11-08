# OmniPlus v2.1｜本地代码先行与云 GPU 实验执行手册

> **文档性质**：对 `docs/PROJECT_PLAN.md`（v2.0）的**执行顺序补充**，不是推翻模型架构与研究目标。  
> **使用对象**：本人 + Codex；先本地实现、极小样本验证所有训练/评测模块，再在云端按顺序启动真实训练与研究实验。  
> **状态更新（2026-11-08）**：R2 架构与 R3-L 本地数据/训练准备已验收；下文保留规划命令，实际可用入口以 [R3-L 报告](stages/R3-local.md) 和 [G3 操作说明](G3_CLOUD_RUNBOOK.md) 为准。云端各阶段尚未执行。
> **下一阶段**：R4-L 等待用户授权。

---

## 0. 一句话方案与冲突优先级

**先把 R2 接口适配好 → 在本地把 R3 数据处理、R4 SFT、R5 Bridge 实验框架、R6 GRPO 算法与实验框架分别用 tiny 数据跑通 → 再租云 GPU，按照 R3 资源 Profile → R4 训练 B1 → R5 结构消融 → R6 强化学习 → R7 最终 Benchmark 的顺序执行。**

以下三条不能因“代码先行”而改变：

1. **代码完成 ≠ 模型有能力 ≠ 研究结论成立**。本地小样本验证仅证明接口、张量、损失、梯度、Checkpoint、流程可以运行，不证明 Talker 能说清楚、视觉能识别或 GRPO 有收益。
2. **R5/R6 的最终实验配方在 R4 的 B1 Baseline 成立后决定**。可提前实现可替换 Bridge、通用 GRPO 数学与测试，但具体层选择、Reward、音频策略与训练资源必须保持配置化和可替换；禁止预先把候选当成最终方法。
3. **历史保护绝对有效**：不得修改、覆盖、重命名 `docs/stages/P0.md`–`P5.md`、既有 P* 实验及旧仓库；保留 MiniMind 回归；不提交大权重、数据或原始语音到 Git。

冲突处理：本文件优先于 v2.0 **R3–R6 的代码实现与云端实训的时间顺序**；v2.0 的模块结构、参数、研究目标、质量验收、用户授权与 R7 标准继续有效。如果调整主文档，以「v2.1 执行附录」补充并在变更日志说明，不要直接抹掉既往版本。

## 1. 双轨阶段：本地先行，云端按闸门推进

| 阶段 | **L：本地实现与 Tiny Smoke**（代码就绪） | **G：云端真实执行与验收**（模型/实验就绪） | 必须区分的状态 |
|---|---|---|---|
| R2 | Qwen1.7B + Projector + Fixed Bridge + Talker + Mimi/Streaming 接口、Mask、梯度检查 | R2 无强制云端正式训练；若 Mac 资源不足，可另行授权真实设备检查 | 仅架构/计算图正确 |
| R3 | Dataset/Collator、分组去重、特征/Codec 缓存、模型与数据 manifest、统一 CLI/配置、Checkpoint 规范 | **G3**：1–10 步真实 CUDA 前后向和可选 100-step Profile，确定显存/吞吐/预算 | R3-L 完成不等于 v2.0 R3 完整验收 |
| R4 | **L4A/B/C/D**：训练入口、冻结分组、联合损失、LoRA、Resume、Eval；每项 tiny 2–10 step | **G4A/B/C/D**：依次完成 T2A Pilot→T2A 正式→视觉/语音对齐→联合 SFT，产出 B1 | 只在 G4D 后认证 B1 |
| R5 | **L5**：可替换 Bridge、Fixed/Last/Weighted/Gated 候选实现或接口、单测、实验矩阵生成器 | **G5**：以 B1 为基准，授权后确定候选方案、重训练并做受控消融 | 本地无 B1 时只能测算法/接口 |
| R6 | **L6**：通用 text GRPO、分组优势、old/current/ref、KL、clip、可变长 mask、8 路 Codebook 联合 logprob 单测、rollout/checkpoint 框架；tiny 离线奖励样例 | **G6**：先 Text GRPO 试验；有可靠语音 Reward 与资源证据后才做 Audio-aware/Joint GRPO | 未确认奖励不能声称完成音频 RL |
| R7 | 测试/报告汇总、指标与绘图、Demo 接口 | **G7**：统一质量/效率 Benchmark、必要系统优化、最终报告 | R4D 之后已经要先进行首轮 B1 Benchmark |

**推荐工作节奏**：R2 单独完成并停下 → 本地 R3-L → 本地 R4-L → 本地 R5-L → 本地 R6-L（每步均独立验收停下）→ 统一云端 G3/G4/G5/G6/G7。实际顺序可以在用户明确授权下调整，但不可跳过依赖或把本地测试写成“已完成训练”。

### 1.1 代码完成的统一定义

每个 L 阶段同时满足：
- 至少 **1 个完全离线 tiny fixture**（伪造张量/合法格式）和 **1 个在许可允许下的真实小样本**；核心数学有可手算/对照的单测；Mac 内存不足时真实 1.7B 只做最小接口验收。
- loss 有限且可反传；冻结模块不更新；需要回传 Projector 时，梯度可以穿过冻结 Qwen（**不能把整个 Qwen forward 包进 `no_grad()`**）。
- `save → new process → resume` 严格恢复模型/LoRA/optimizer/scheduler/RNG/数据游标/配置；单测验证 step 连续性和缺权重报错。
- `pytest`、Ruff、P2/P3/P4/P5 老回归通过；输出各自 `docs/stages/R*-local.md` 或阶段报告中“local readiness”小节，**不得覆盖已签收历史报告**。
- 一条真实、可复制的 CLI 命令能完成对应工作；错误时非零退出并给出失败原因，**不能只留下伪代码/空实现**。

### 1.2 小样本规范

| 任务 | 本地 fixture 规模（建议上限） | 必须检查 |
|---|---:|---|
| T2A Talker | 8–32 对短文本/音频，训练更新 2–10 步 | 8 路 codes、STOP/PAD/Delay、mask、梯度、长短句、不读取未来音频 |
| Vision I2T | 8–32 张许可合规小图片，更新 2–5 步 | 64 tokens、特征注入、模态调换测试、Qwen 梯度路径 |
| Audio A2T/A2A | 8–16 条短语音，更新 2–5 步 | 16k SenseVoice 与 24k Mimi 分开、源语音/目标语音隔离 |
| Joint SFT | 三任务各 4–8 条，更新 2–5 步 | loss 比例、按任务采样、LoRA on/off、checkpoint |
| Bridge 消融 | 合成 hidden states + 8–16 T2A fixture | 相同输入/维度、权重初始化、反向与固定实验配置 |
| Text GRPO | 2–4 个 prompts × group 2–4 | advantage、概率比、clip、KL、可变长 mask、数值边界 |
| 8-Codebook 联合 GRPO | 2 个 prompts × 2 条假想 rollout | 仅有效声码 logprob 求和、Delay 对齐、0 有效 token、STOP、采样行为分布一致 |

> Tiny 的“跑通”只可证明工程与数学正确性；尤其 R5/R6 不能使用未训练模型产生的随机输出来证明算法改善。

---

## 2. 需要 Codex 提供的稳定 CLI 契约（**待实现，并非当前仓库已存在**）

2026-11-08 已落实 `scripts/run_experiment.py inspect/prepare/smoke --config ...`、`--dry-run` 和 `smoke --resume`。参数统计为 meta 模型；恢复写新 run，保留原记录。下文 `train/resume/evaluate/profile/export` 独立命令仍是后续设计，不能直接当作已实现入口。真实本地复现命令见 [R3-L 报告](stages/R3-local.md)。

先审计并尽量复用已有 `scripts/train_sft.py`、`scripts/prepare_p4.py`、`omni/training/`、`omni/inference/`。**推荐新增一层薄 CLI 包装**，而不是重写模型/训练器。最终命令名称应由 Codex 实施并写入 README 和每阶段报告；以下是要求落实的**目标接口**，执行前必须在真实仓库中通过 `--help` 和 smoke 证明它们存在。

```bash
# 以下命令是“Codex 需要实装的命令契约”，现在不是已验收的仓库命令。
python scripts/run_experiment.py inspect  --config configs/v2/model.yaml
python scripts/run_experiment.py prepare  --config configs/v2/data/t2a.yaml --max-samples 32
python scripts/run_experiment.py smoke    --config configs/v2/r4a_t2a_smoke.yaml --device cpu
python scripts/run_experiment.py profile  --config configs/v2/r3_cuda_profile.yaml --steps 10
python scripts/run_experiment.py train    --config configs/v2/r4a_t2a.yaml --run-id r4a-run001
python scripts/run_experiment.py resume   --run-dir "$RUN_ROOT/r4a-run001" --checkpoint latest
python scripts/run_experiment.py evaluate --config configs/v2/eval_b1.yaml --checkpoint "$RUN_ROOT/r4d-b1/checkpoints/best"
python scripts/run_experiment.py export   --run-dir "$RUN_ROOT/r4d-b1"
```

**统一选项必须支持**：`--config`、`--run-id` 或 `--run-dir`、`--device`、`--max-samples`（仅准备/Smoke）、`--steps`（仅 Smoke/Profile）、`--dry-run`（对下载/训练有实际影响的命令）、`--checkpoint`；CLI 内按 config 实际路由到 SFT/GRPO/Bridge，不需要为每阶段复制一套训练脚本。

**建议配置布局（新增，旧配置一律保留）**：

```text
configs/v2/
  model.yaml                  # Qwen、Projector、FixedBridge、Talker、codec 版本
  assets.yaml                 # repo/revision/path/license/processor/sha
  data/{t2a,i2t,a2t,a2a,joint}.yaml
  r3_cuda_profile.yaml
  r4a_t2a_smoke.yaml          # 本地/云端短验证
  r4a_t2a.yaml                # 全尺寸模型短闭环
  r4b_t2a.yaml                # T2A 正式配方
  r4c_vision.yaml
  r4c_audio.yaml
  r4d_joint_sft.yaml
  r5_{fixed,last,weighted,gated}.yaml
  r6_{text,codebook_sanity,joint}.yaml
  eval_{t2a,vision,audio,b1,bridge,grpo}.yaml
```

上面是**建议命名**，R2 不应提前生成无实现的空目录；R3-L/R4-L 创建时可随实际代码微调。所有 YAML 字段要在 parser 中验证，错误的 Key 必须报错而不是静默忽略。

---

## 3. 云端统一环境：只需搭建一次

**资源策略**：首选 **1×H100 80GB** 作训练，条件允许先比较 A100 80GB 的成本/吞吐；R6 可能需要 2 GPU，但应先经过单卡 Profile。以下示例基于一台可 SSH 登录、Linux + NVIDIA 驱动 + 足够持久盘空间的云主机，**不依赖某个具体云厂商**。实际 CUDA/PyTorch wheel 要与镜像及锁文件核对，不能盲目用 Mac 的安装方式替代。

### 3.1 本地检查 Git 代码，推送经授权的提交

```bash
git status --short
git diff --check
# 跑完整 pytest / ruff / 受保护历史 SHA 检查，确认无问题后提交
# 用户决定是否 push；此处不自动操作远端仓库
```

### 3.2 SSH 到云 GPU 并克隆

```bash
# 云端终端示例：已有 SSH/Git 凭据；私有仓库请用 SSH URL 或规范的凭据管理方式
export REPO_URL='https://github.com/luminoso909/Omni-plus.git'
export PROJECT_ROOT="$HOME/Omni-plus"
git clone "$REPO_URL" "$PROJECT_ROOT"
cd "$PROJECT_ROOT"
git rev-parse HEAD
```

**重要**：项目 URL 来自 P0 报告，Codex 须核对当前远端及权限；如果仓库私有/已迁移，要先按 `git remote -v` 修正，而不是硬编码这个地址到所有脚本。每个训练 run 记录 Git commit SHA；运行期间不自动 `git pull`。

### 3.3 创建独立数据/模型/结果位置和 Python 环境

```bash
# 按云盘实际挂载点调整；确保 /workspace 是持久卷，不是关机即清空的临时盘
export WORK_ROOT=/workspace/omniplus
export MODEL_ROOT="$WORK_ROOT/models"
export DATA_ROOT="$WORK_ROOT/data"
export RUN_ROOT="$WORK_ROOT/runs"
export HF_HOME="$WORK_ROOT/hf-cache"
mkdir -p "$MODEL_ROOT" "$DATA_ROOT" "$RUN_ROOT" "$HF_HOME"

cd "$PROJECT_ROOT"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev,media,encoders,plots,data]'
# 仅在 Codex 审计确认版本与镜像匹配时补装 PEFT / TRL / CUDA PyTorch，不静默破坏旧依赖锁
python - <<'PY'
import torch
print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
PY
nvidia-smi
python -m pip check
```

云镜像自带的 CUDA Toolkit 版本不等于 PyTorch 编译使用的 CUDA 版本。`torch.cuda.is_available()` 不为 True 时必须修好环境，禁止标记 Profile 成功。依赖修改保存在独立的 `configs/v2/environment_cuda.md`/锁文件记录，不能覆盖 P0–P5 复现配方。

### 3.4 模型权重下载：先复用，缺什么补什么

```bash
# hf download 用法已由官方文档核实；正式 run 应固定 commit revision。
python -m pip install 'huggingface_hub>=0.36.2,<1'
# QWEN_REV 由下载前的资产 manifest 填入，禁止留空或假造 SHA
: "${QWEN_REV:?必须先核实 Qwen3-1.7B 的官方 commit revision}"
hf download Qwen/Qwen3-1.7B \
  --revision "$QWEN_REV" \
  --local-dir "$MODEL_ROOT/qwen3-1.7b"

# 若需要且经核实版本/许可，再按相同模式下载：
# google/siglip2-base-patch32-256
# FunAudioLLM/SenseVoiceSmall
# kyutai/mimi
# 可选 jingyaogong/campplus
```

- SigLIP2、SenseVoice、Mimi **优先复用已通过 P2/P3 验收的相容权重和预处理**；云端若重新从官方仓库获取不同格式/版本，必须重测 shape/数值/码本，必要时重建缓存。
- 模型下载命令由 `assets.yaml` 生成或在 cloud guide 中明示；每次记录 `repo_id、revision、hash、dtype、processor、sample_rate、codec_codebooks、license`。
- 官方 `hf download` 的 `--repo-type dataset`、`--revision`、`--local-dir` 和 `--include` 支持按仓库和文件精确下载，避免把完整数据集误下载进临时根盘。参考：<https://huggingface.co/docs/huggingface_hub/guides/cli>。

### 3.5 数据集下载与小批量准备

**不能把候选数据的完整下载命令当作默认命令**。先用 API 检查远端文件结构、许可证、样本字段和文件大小，再由 `prepare` 读取用户批准的数据切片或清单；对于 parquet rowgroup 特别大等情况复用 P4 的有界读取经验。

```bash
# 远端数据文件清单检查（不下载数据本体）
python - <<'PY'
from huggingface_hub import HfApi
for repo in ['jingyaogong/minimind-o_dataset',
             'liuhaotian/LLaVA-Instruct-150K']:
    print('\nDATASET', repo)
    files=HfApi().list_repo_files(repo, repo_type='dataset')
    for name in files[:30]:
        print(name)
    print('total files:', len(files))
PY

# 数据准备命令：须由 Codex 在 R3-L 真实实现并测试
python scripts/run_experiment.py prepare \
  --config configs/v2/data/t2a.yaml --max-samples 128
python scripts/run_experiment.py prepare \
  --config configs/v2/data/i2t.yaml --max-samples 64
```

**绝不**将 MiniMind Token IDs 直接喂入 Qwen；对原始文本重新 tokenization。对已有 Mimi Codes 检查同版本/8 codebooks/采样率；VoiceAssistant-400K 的 SNAC/CosyVoice 码不能冒充 Mimi。按原始图片/语音媒体分组去重；官方 split 优先，否则 90/5/5，正式 Test 永不用于调参。

### 3.6 统一输出目录、日志、Checkpoint（所有云任务都必须满足）

```text
$RUN_ROOT/<run_id>/
  resolved_config.yaml           # 真正生效的配置，含 git SHA / dataset & model revisions
  manifest.json                  # 设备/dtype/可训练参数/数据 split SHA/开始结束时间
  logs/{stdout.log,stderr.log}
  metrics/{train.jsonl,val.jsonl,system.jsonl}
  checkpoints/{step-000100/,step-000200/,best/}
  eval/{summary.json,per_sample.jsonl,failures.jsonl}
  samples/                       # 经许可证及隐私审查后可选保存短 WAV/图片
  charts/
  status.json                    # running/completed/failed/interrupted + last checkpoint
```

- `run_id` 唯一，永远不覆盖旧 run；resume **追加**同一 run 的训练记录并明确 resume 事件，不重置全局 step。
- **SFT 可恢复状态**：模型新增模块权重、LoRA adapter、optimizer、scheduler、GradScaler（适用时）、随机种子与全部 RNG 状态、sampler/数据游标、配置、模型/数据/代码 SHA、epoch/step；完整恢复与仅加载推理权重是两种明确模式。
- **GRPO 可恢复状态**：在上述基础上还需记录 policy/ref/old 策略版本、reward 版本、组采样配置、已完成/待更新 rollout 批次边界、分布式 world_size/rank 信息。**只在参数更新边界保存一致快照**；中断在采样中段时，重新生成该未提交 rollout 批次，不可把旧/新策略数据混合为一个 batch。
- `best` 由固定 Validation 指标决定，不能以 Test 选最优；必须可以 `resume --checkpoint latest` 和只读 `evaluate --checkpoint best`。
- 云端断连：用 `tmux`/提供商持续任务容器，并将结果**同步到持久卷和独立备份**。不把未及时备份的临时容器盘当唯一副本。

### 3.7 日志与结果带回本地

```bash
# 本地 Mac 运行：SERVER 由用户替换为 SSH host；非敏感结果可直接 rsync
export SERVER='user@your-cloud-host'
export REMOTE_RUN='/workspace/omniplus/runs/r4d-b1'
mkdir -p ./cloud-results/r4d-b1
rsync -av --partial \
  --exclude='checkpoints/' \
  "$SERVER:$REMOTE_RUN/" ./cloud-results/r4d-b1/

# 大型 checkpoints 单独搬运至有访问控制的对象存储/硬盘；必须验证完整性
# 不要 git add 数据、权重、音频原件和大型 checkpoints
```

### 3.8 首次云端校验：G3，唯一的资源门槛

```bash
python scripts/run_experiment.py profile \
  --config configs/v2/r3_cuda_profile.yaml --steps 10
# 通过后：可选 100 step，测 1.7B + 正式 4×768 Talker 的显存和吞吐
python scripts/run_experiment.py profile \
  --config configs/v2/r3_cuda_profile.yaml --steps 100
```

记录 BF16/FP32、microbatch、音频帧长度分布、LoRA on/off、可训练参数、显存高水位、samples/s、checkpoint 恢复。**R3-G 失败就先修工程或调整配置；不得跳到 G4 大规模训练。**

---

## 4. 按任务执行的云 GPU 作业清单

> **以下每行的 config 与 CLI 是 Codex 必须提供并在云环境验证的目标命令，不代表现在已能运行。** 同一类任务通过更换 `--config` 调整数据比例、显存预算、训练步数和参数冻结，无需远程手改核心代码。正式运行需要用户显式批准数据下载/预算。

| 云任务 | 输入/数据起点 | 训练参数 | 推荐 GPU | 目标 CLI（统一运行器） | 输出与停止标准 |
|---|---|---|---|---|---|
| **G4A T2A 过拟合** | 小 T2A、原始 WAV 或兼容 Mimi Codes | Frozen Qwen；Bridge+Talker+Heads | 1×80GB | `python scripts/run_experiment.py train --config configs/v2/r4a_t2a.yaml --run-id r4a-001` | 梯度/重启恢复、loss 可降；不宣称语音质量 |
| **G4B T2A 正式** | T2A mini→经审计 full 子集 | 同 G4A | 1×80GB | `... train --config configs/v2/r4b_t2a.yaml --run-id r4b-001` | 音频可懂度、ASR CER/WER、失败率、试听；不收敛则停 |
| **G4C-Vision** | COCO Caption→LLaVA | Vision Projector，Qwen 初期冻结 | 1×80GB | `... train --config configs/v2/r4c_vision.yaml --run-id r4cv-001` | Caption/VQA 和换图敏感性 |
| **G4C-Audio** | LibriSpeech/A2A | Audio Projector，Qwen 初期冻结 | 1×80GB | `... train --config configs/v2/r4c_audio.yaml --run-id r4ca-001` | ASR/语音 QA，换语音敏感性 |
| **G4D 联合 SFT** | 去重后的 T2A+I2T+A2T/A2A+可选文本 | Projectors/Bridge/Talker/Heads + 可选 Qwen LoRA | 1×80GB | `... train --config configs/v2/r4d_joint_sft.yaml --run-id r4d-b1` | 保存可复现 B1 SFT Checkpoint |
| **B1 首轮 Benchmark** | 冻结 Validation 与独立 Benchmark | 不训练 | 1×80GB | `... evaluate --config configs/v2/eval_b1.yaml --checkpoint "$RUN_ROOT/r4d-b1/checkpoints/best"` | 立刻保存 B1 全任务和效率指标，不等到 R7 |
| **G5 Bridge** | 与 B1 相同数据/评价协议 | 按方案重训练 Bridge/指定模块，控制参数和预算 | 1×80GB | `... train --config configs/v2/r5_weighted.yaml --run-id r5-weighted-001` | Fixed/Last/Weighted/Gated 受控对照；不只换推理层 |
| **G6 Text GRPO** | GSM8K 等可验证 prompt，R6 再定 | 基于 B1 更新指定 LoRA/策略参数 | 1×80GB 起 | `... train --config configs/v2/r6_text.yaml --run-id r6-text-001` | 分布、概率、奖励/稳定性与 SFT 对照 |
| **G6 Joint GRPO** | 用户批准的可验证音频任务 | R6 另定 | 1×80GB 起，必要时 2 GPU | `... train --config configs/v2/r6_joint.yaml --run-id r6-joint-001` | 8 码本有效步联合 logprob、反作弊、独立质量评测 |
| **G7 统一评测** | 固定 benchmark splits | 不训练；按瓶颈做系统小实验 | 1×80GB | `... evaluate --config configs/v2/eval_b1.yaml --checkpoint ...` | 统一对比、性能 Profile、报告与 Demo |

`...` 是表格中重复的 `python scripts/run_experiment.py` 缩写，**不是可直接复制执行的命令**；正式文档要在 Codex 实装后输出逐条不含省略号的完整命令。

### 4.1 推荐参数配置字段（不得由 Codex 擅自锁死数值）

```yaml
experiment:
  stage: R4B
  task: T2A
  run_id: r4b-001
model:
  thinker: Qwen/Qwen3-1.7B
  thinker_revision: REQUIRED_PINNED_REVISION
  thinker_frozen: true
  lora: {enabled: false}
  bridge: {type: fixed, output_hidden: 768, source_layer: TO_BE_VALIDATED}
  talker: {layers: 4, hidden_size: 768, codebooks: 8}
data:
  manifest: REQUIRED_DATASET_MANIFEST
  train_split: train
  val_split: validation
  max_audio_seconds: TO_BE_PROFILED
train:
  seed: 42
  precision: bf16
  micro_batch_size: TO_BE_PROFILED
  grad_accum_steps: TO_BE_PROFILED
  max_steps: TO_BE_APPROVED
  checkpoint_every_steps: TO_BE_APPROVED
  validate_every_steps: TO_BE_APPROVED
  resume: null
output:
  root: '${RUN_ROOT}'
```

这是**配置协议示意**。`REQUIRED_*`、`TO_BE_*` 是强制待填写占位值，解析器必须拒绝这些占位符，禁止默默按默认值开大训练。每次运行保存解析后的 resolved YAML。

### 4.2 GRPO 单卡与多卡机制

- **优先单 GPU**：短 prompt、小组大小和生成预算，文本 GRPO 先证明算法；通过梯度检查点、参考策略顺序计算、避免同时保留不必要大对象控制显存。
- **必要时两 GPU**：选择（A）两个训练 rank 使用 DDP/FSDP/ZeRO（当代码确实支持，且对吞吐或状态分片有意义）；或（B）一张 Rollout/Reward、一张训练 GPU（需要明确权重版本同步、队列一致性和采样策略）。两者不应同时盲目加入。
- **仅当支持多进程且已通过一致性单测时**，才可使用下面的 PyTorch 官方启动方式：

```bash
# G6 的可选双卡命令；默认不可运行，直至 distributed test 通过
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc-per-node=2 \
  scripts/run_experiment.py train \
  --config configs/v2/r6_joint.yaml --run-id r6-joint-2gpu-001
```

`torchrun` 启动 N 个独立进程；**并不会自动把单卡训练程序变成正确的分布式 GRPO**。必须具备 rank0 checkpoint、全局 group/advantage 的正确分组、分布式 seed、同步梯度、失败恢复、度量聚合和 reference/rollout version 协议。DDP **不会**为每卡分片完整模型状态；显存不足需重新选择 offload/FSDP/ZeRO 或 pipeline 方案。PyTorch 官方：<https://docs.pytorch.org/docs/stable/elastic/run.html>。

### 4.3 费用/时间估算（**未经过 R3-G 实测，不作为承诺**）

| 阶段 | 暂供预算讨论的 1×H100 80GB 墙钟量级 | 可信度 | 主要不确定因素 |
|---|---:|---|---|
| G3 Profile | 约 1–4h | 低 | 环境/数据下载是否已有缓存 |
| G4A | 约 3–10h | 很低 | 模型/数据/输出长度 |
| G4B | 约 30–100h | 很低 | 随机 Talker 收敛、语音时长 |
| G4C | 约 10–30h | 很低 | 视觉/音频任务比例 |
| G4D | 约 20–70h | 很低 | LoRA、长序列、联合任务吞吐 |
| G5 | 约 30–120h | 很低 | 候选数量、是否重复训练、种子数 |
| G6 | 约 20–100h 或更多 | 极低 | GRPO rollout/音频奖励/双卡模式 |
| G7 | 约 5–20h | 很低 | benchmark、优化实验规模 |

这是此前讨论的**规划级范围**，并非实际 GPU 测试数据；**不可据此直接预充值或承诺项目周期**。在 G3 的相同硬件/序列长度/任务配置测得：

`预计训练时长 ≈ (目标 optimizer updates × 实测每 update 秒数) + 数据/评测/定期保存开销`。

若使用双卡，应分别报告墙钟与 GPU·小时，不能把两者混为一谈。

---

## 5. 训练、实验与 Benchmark 的位置

| 时间点 | 产出 | 不能做/不能声称 |
|---|---|---|
| R2-L / R3-L | 可执行的结构、数据和训练框架、tiny 回归 | 不宣称多模态能力 |
| R4-L / R5-L / R6-L | 可单测的 SFT、Bridge、RL **代码** | 不宣称完成正式模型训练、消融效果或 GRPO 收益 |
| **G4D 完成** | **B1 = Qwen3-1.7B + FixedBridge + Talker + 联合 SFT Checkpoint** | 不把原生文本 Qwen 当语音 Baseline |
| **B1 紧随 G4D** | **首次正式 Baseline Benchmark**，锁定模型/数据/评测协议 | 不等到 R7 才第一次评测 |
| G5 | Fixed vs Last/Weighted/Gated 等**架构消融** | 不凭只替换前向层号声称改善 |
| G6 | SFT vs Text GRPO / 有效的 Audio-aware / Joint GRPO **RL 消融** | 奖励不可验证则不强行上音频 RL |
| G7 | 统一 Benchmark、系统性能消融、最终技术报告 | 不把新改的架构未经重训直接和旧 B1 比 |

R5 与 R6 以**同一个冻结 B1**为研究起点，可以分别执行；两者组合实验可选，且需清楚归因。验证集用于调参，独立 Test 用于锁定后的报告，不得反复用 Test 选 checkpoint。

---

## 6. Codex 分工、实施顺序与文档维护规则

**本人主导/审查**：Bridge 定义与选层、音文对齐的数学语义、Talker 关键结构、8-codebook 联合对数概率、GRPO objective/reward、实验变量与最终方法选择。  
**Codex 获授权可做**：训练/评估/云端薄脚本，数据/缓存、CLI、yaml、CI、测试 fixture、环境/版本清单、日志、恢复/同步、GPU Profile、图表、文档与代码辅助调试。**不可因为“先全写完”就自作主张锁定尚未设计的 Reward、算法或覆盖用户保留的核心实现。**

### 6.1 每次给 Codex 的交付协议

1. 先读取真实仓库和最新 `AGENTS.md`、`PROJECT_PLAN.md`、本文件；列出拟改文件/当前测试/运行命令/保护文件 SHA。
2. 只执行用户本次授权的 **一个 L 或 G 单元**，避免一次性开启长时训练、全量数据下载、下一阶段。
3. 创建独立 `experiments/r*/` 新 run/manifest；对 `README.md`、`docs/项目待办.md`、`AGENTS.md` 仅同步真实状态。已签收历史 stage 报告不回写。
4. 交付代码文件表、可以实际复制的 CLI（含 `--help` / smoke 输出）、测试证据、无效样本/边界用例、已解决和未解决风险、下一单元入口。
5. 云端实验额外交付硬件、GPU 型号/数量、dtype、实际 wall time、GPU·hours、版本 SHA、单卡/多卡启动命令、Checkpoint 和结果下载方式。
6. 完成当前单元立即停止。没有用户明确授权，禁止自动跑 G4B/G4D/G5/G6 正式训练。

### 6.2 文档同步建议

| 文档 | 行为 |
|---|---|
| `docs/PROJECT_PLAN.md` | 保留 v2.0 研究架构/阶段定义，新增 v2.1 执行顺序说明和本手册链接；若实质修改先保留 archive 与 diff |
| `docs/OMNIPLUS_LOCAL_FIRST_CLOUD_PLAYBOOK.md` | 将**本文件**作为新的可操作云端手册，Codex 在实际接口落地后更新命令的“verified”状态 |
| `docs/OMNIPLUS_DECISIONS_AND_DATA.md` | 放入配套“项目认知汇总”文件，作为模块/数据/参数/实验依据 |
| `README.md` | 明确代码已实现、tiny smoke 已通过、云端正式训练/研究是否完成三种状态 |
| `docs/项目待办.md` | R2-L/R3-L/R4-L/R5-L/R6-L 与 G3/G4/G5/G6/G7 **两套状态栏**，不把 L 勾选等同于 G |
| `AGENTS.md` | 加入 stage gate、历史保护、真实命令验收、云端预算和 secrets 规则 |
| `docs/stages/P0.md`–`P5.md` | **严禁改动**；`R0/R1` 已验收证据也优先只读引用 |

### 6.3 什么时候可以宣称最终成功？

- **工程完成**：各 L 阶段稳定运行，旧测试不回退，真实样本 tiny test 有证明。
- **模型完成**：G4 有可懂语音、真实视觉/语音输入条件效果、独立评测与 B1 检查点。
- **算法研究完成**：G5/G6 在锁定 B1 上完成实际训练和受控对照；无收益也如实报告。
- **可复现交付完成**：G7 的 README、可运行云端步骤、结果 manifest/图表/模型来源、Demo/技术报告完整。

---

## 7. 风险登记（不得跳过）

| 风险 | 最低保护动作 |
|---|---|
| **随机 Talker 不收敛** | G4A/B 通过小样本过拟合+音频可懂度评估；不达标暂停并向用户汇报是否允许 MiniMind Talker 初始化迁移 |
| **Qwen 字词 token 与音频帧不一一对应** | R2 必须两条真实 T2A 的逐步时间轴、causal mask、STOP/Shift/Restore 测试；不默默裁长音频 |
| **冻结 Qwen 阻断 Projector 梯度** | 只冻结参数；需要 Projector 更新时保留 Qwen 的前向计算图 |
| **LoRA/新增特殊 Token 保存缺项** | adapter + 新模块 + embedding/head（如变更）严格恢复，缺项失败，不用 `strict=False` 掩盖 |
| **GRPO 行为策略概率不一致** | 记录 temperature/top-p/mask、old/ref/current、8 codebook 有效 action logprob，单测验证 |
| **数据泄漏/版本不一致** | 按媒体/来源 ID 分组去重；record model/processor/codec/tokenizer SHA，缓存版本不符就失效 |
| **云端断连/临时盘丢失** | 持久卷 + 每若干步 checkpoint + 定期独立备份 + resume 测试 |
| **多 GPU 看似启动却不正确** | torchrun 只是进程启动器；distributed group/mask/聚合、rank0 保存/恢复必须做一致性测试 |
| **预算失控** | G3 Profile 之前不启动正式长训练；run 配置需最大步数/时长/存储限额/手动批准 |

## 8. 外部操作依据

- Hugging Face CLI 的仓库、revision、数据集下载参数：<https://huggingface.co/docs/huggingface_hub/guides/cli>
- PyTorch `torchrun` 单节点多进程启动语义：<https://docs.pytorch.org/docs/stable/elastic/run.html>
- Qwen3-1.7B 官方模型：<https://huggingface.co/Qwen/Qwen3-1.7B>
- Hugging Face PEFT/LoRA 集成：<https://huggingface.co/docs/transformers/peft>
- 本项目现有历史事实与正式研究边界：`docs/PROJECT_PLAN.md`、`docs/stages/P0.md`–`P5.md`、`docs/stages/R0.md`、`R1.md`（以实际仓库为准）。

**当前下一步：R4-L，等待授权；随后逐个验收 R5-L、R6-L，再进入云端 G3–G7。**
