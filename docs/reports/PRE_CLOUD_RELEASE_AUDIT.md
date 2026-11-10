## 一、结论与范围

**READY FOR G3：具备进入云端部署和有限资源验收的工程条件。尚不具备直接启动正式 G4–G6 训练的条件。**

审计日期：2026-11-10。基线为 `30bb068c52d212683bcf2169ee4a9a40453897a7`，另有本报告列明的小型工程修复。发布时采用包含本报告的实际提交，不能把当前带有用户未提交修改的工作区整体复制后声称是该提交。

R5-L 的两个 P1 已关闭，成本汇总 P2 保持通过；R6-L 的数学、Tiny 更新及恢复仍为本地 PASS。本轮没有发现阻断有限 G3 的未关闭 P0/P1。正式训练前的独立评测、数据供应等缺口继续保留，见第四章；它们不因本结论而被勾选完成。

**当前不能直接执行默认真实资产配置。** 本机外置媒体盘未挂载，`weights/media-reference` 不存在，也尚未生成本轮 G3 审计数据包。进入 G3 后先完成资产、环境和数据准备，再经实际预检启动获批的 1–10 步。本文没有启动云主机、下载完整数据、正式训练或 Push。

## 二、源码与阶段验收核对

已核对 `AGENTS.md`、README、主规划、v2.1 本地云端手册/决策文档、待办、R0–R6-L 已存在报告及 G3–G7 手册。历史报告与前轮已完整读取的内容逐文件核对 SHA；判断同时依据实际源码、配置、测试和产物，而非仅依据阶段勾选。

| 范围 | 实际路径与证据 | 当前结论 |
| --- | --- | --- |
| R5 数据隔离 | `omni/data/sft_sources.py`、`bridge_splits.py`；`test_r5_data_identity.py`、`test_r5_bridge_splits.py` | v3 契约在预算选样前保留完整媒体组及未选别名，跨 B1/目标/祖先检查；缺身份拒绝。丢别名、换 ID 的原 P1 已关闭。 |
| R5 源码身份 | `omni/training/run_source.py`；两 CLI 与 `run_identity()`；`test_r5_source_origin.py` | 检查实际模块 `__file__`、spec origin、包搜索路径和源码摘要，不只看 cwd/Commit；跨 checkout 的原 P1 已关闭。运行中源码须保持固定。 |
| R5 成本 | `scripts/bridge_ablation.py`、`omni/evaluation/bridge_benchmark.py`；`test_r5_ablation.py` | 训练与推理分列，恢复区间、失败结果和 CPU 空值保留；本轮未改成本算法。 |
| R6 数学与更新 | `omni/training/grpo*.py`、`r6_local.py`；`test_r6_math.py`、`test_r6_rollout.py`、`test_r6_training.py` | Text/Audio/Joint 共享真实 GRPO；Old 固定、Reference 冻结、优势/裁剪/精确条件 KL、八码本有效 logprob 求和均有独立期望测试。仅支持完整支持集正温度采样，拒绝 Top-k/Top-p 截断。 |
| R6 恢复与隔离 | `sft.py`、`r6_local.py`；`test_r6_integrity.py`、`test_r6_review.py` | 保存完整更新边界、Reference、策略版本、Optimizer/Scheduler/Scaler/RNG/游标，继承媒体及源码隔离。跨进程恢复包含在回归中；CUDA 未验。 |
| 模型与流式接口 | `model_omni_v2.py`、`thinker/qwen_backend.py`、`bridge.py`、`talker/model_talker.py`、`inference/{codec,streaming}.py` | Qwen 2048 → Bridge 768，无重复投影；4×768 Talker、8-Codebook、Mimi/PCM/Cache 接口已验。Fixed 默认；Last/Weighted/Gated 可训练。真实资产证据来自 R2，非本轮重新实测。 |
| SFT A/B/C/D | `r3_local.py`、`sft.py`、`adapters.py`、`configs/v2/r4*.yaml` | T2A、独立 Vision/Audio Projector、联合 LoRA、多任务采样、验证/早停、完整恢复和独立 Adapter 入口存在。冻结 Qwen 不切断 Projector 梯度。 |
| 数据与标签 | `omni/data/{qwen_omni,audio_alignment,sft_sources,r3_prepare}.py` | 原生模板、Shift/Mask/Delay、16kHz 输入与 24kHz 编码分路，媒体组 split/去重、版本化缓存和不兼容 Mimi codes 拒绝已实现。QA A2A 与 echo 诊断区分。 |
| 评测与产物 | `omni/evaluation/{sft,bridge_benchmark}.py`、`r6_local.py`、`bridge_ablation.py collect` | 现有 SFT Loss/STOP、T2A WAV/ASR、RL Reward、Bridge 汇总可用；不等于完整 Text/Vision/Audio 统一 Benchmark 或实时 Qwen Demo。 |

R5 原问题状态以 [最终修复报告](../reviews/R5-L_FINAL_FIX_REPORT.md) 为准；R6 以 [阶段报告](../stages/R6-local.md) 和 [同轮自审](../reviews/R6-L_FINAL_REVIEW.md) 为准。本轮的交叉代码检查及全仓回归未重新宣称正式音质、训练收益或外部独立算法认证。

### 1 云手册与工作流

| 手册 | 实际入口 | 本轮核对与边界 |
| --- | --- | --- |
| G3 | `inspect / prepare / smoke / resume / profile` | 原跟踪文件 `docs/G3_CLOUD_RUNBOOK.md` 被用户移到 `docs/cloud/`，尚未提交；保留该改动。旧 schema 1/未实现 CLI 描述过时，**本报告第六章及 `g3_release.yaml` 是当前启动步骤**。 |
| G4 | `train / resume / evaluate / demo` | A/B/C/D 模板、组件初始化和完整恢复存在；B1 首次 Benchmark 必须在 G4D 完成时执行。 |
| G5 | `bridge_ablation.py plan/collect` + 共享 SFT 入口 | 独立配置/Run、B1 谱系、同协议终点和成本检查存在；候选及预算等 B1 成立后确定。 |
| G6 | `grpo-smoke/train/resume/evaluate/profile` | 三模式和 Baseline 只读 Reward 对照存在；本轮将示例数据数改为 20，使 16/2/2 split 满足两条评测 prompt。 |
| G7 | 现有 `evaluate/demo/grpo-evaluate/collect`，其余明确标待实现 | 手册覆盖最终 Benchmark/性能/Demo/报告/回传；不把未实现的统一评测器、Triton 或 `export` 命令包装成已可运行。 |

真实 CLI 使用 `--config/--run-dir/--resume/--steps/--dry-run`，RL 另支持 `--baseline`。旧总规划里的 `--checkpoint/--run-id/--device/--max-samples` 或 `eval_b1.yaml` 是目标设计，不能直接运行。`profile` 和 `grpo-profile` 会更新参数，必须按训练预算授权。

## 三、本轮小型修复

| 修改 | 修前证据 | 修后边界与测试 |
| --- | --- | --- |
| SFT dry-run 复用实际初始化/恢复校验 | 在 `30bb068` 的隔离 Tiny CLI 中，错误 `.pt` 和缺失 test split 均输出 `validated` | `r3_local.py:inspect_sft_run` + CLI：检查实际 split/任务、数据语义、初始化组件/Bridge、Checkpoint 身份/config/参数/step/冻结与恢复状态。错误输入明确拒绝且不创建 run；正向恢复、评测与初始化通过。 |
| 预检交叉复核 | 初版遗漏评测/validation 缺任务、空 Optimizer 状态，以及 legacy torch.save 与 mmap 的兼容边界 | 补充对应正反例；不加载真实 Qwen，不执行 forward，不恢复全局 RNG。完整实际 Resume 仍必须在 G3 新进程验证，预检不证明 CUDA 数值和设备可用。 |
| CI 统一入口 | `.venv/bin/pytest --collect-only ... tests/test_r5_ablation.py` exit 2，`ModuleNotFoundError: scripts` | `.github/workflows/ci.yml` 使用 `python -m pytest -q`，与本地命令一致；本轮未运行远程 CI。 |
| 有界 G3 配置 | 老 `g3_h100.yaml` 是 schema 1，不能拿 100 步说明当作已支持 | 新 `configs/v2/g3_release.yaml` 继承 schema 2/R4A，真实 Qwen、CUDA BF16、冻结主权重、8 条 T2A、最多 10 步、每步保存/每两步验证、新缓存。摘要占位值必须替换为已核验数据摘要。 |
| 文档 | G6 示例 16 条只留出 1 条 validation，却配置评测 2 条 | 改为 20 条、16/2/2；更新 G7 的预检范围和 G3 导航，README/待办记录当前审计。 |

新测试位于 `tests/test_precloud_preflight.py`。未改模型核心、Loss/GRPO/Bridge 算法、P2 成本汇总或历史实验。本轮源码摘要自然变化；历史 Checkpoint 继续在其原提交/配置/环境恢复，不删除身份门槛，也不重签旧产物。

预检只核对 Optimizer 分组/参数数量、Scheduler/Scaler 基本结构和 CPU RNG 的私有实例恢复；设备 RNG 仅作结构检查，不代表真实 CUDA 恢复已验收。新版 zip Checkpoint 使用 CPU mmap；旧格式仅允许 **≤64 MiB** 的 CPU 回退，更大文件明确要求另存新版，不自动修改源文件。正常训练初始化路径不变。

## 四、剩余问题分级

下表沿用 [前轮 A–D 审计编号](OMNIPLUS_DEVELOPMENT_AUDIT.md)。区分 G3 的有限试运行和后续正式训练门槛，避免把未实现的评测功能误报为已交付。

### 1 云训练前必须完成

| 优先级 / 编号 | 当前状态及路径 | 必须做什么、何时做 |
| --- | --- | --- |
| P1 / 启动条件 | `configs/v2/g3_release.yaml`；媒体目录及本轮审计 JSONL 缺失 | **真实 G3 步骤前**：选定干净提交，搬运锁定资产/许可与 8 条独立合法录音，核验摘要、填云路径，新建云缓存。环境/数据未就位不能开始 smoke；这不是已经完成的云端实测。 |
| P1 / A1 | SFT `evaluate_run`、RL `prepare_run` 严格绑定完整运行配置 | **正式 B1 配置冻结前**：实现独立只读评测加载，分别锁定训练来源与评测协议，保留祖先隔离。不能通过放松完整 Resume 来支持训练后任意切换 test/AMP/长度。有限 G3 不依赖此新入口。 |
| P1 / A2 | `sft_sources.py` 接受已审计 JSONL，非所有上游格式下载器 | **各正式数据源入训前**：获批来源的转换清单、官方 split、全部媒体别名/转码关系、许可与 transcript 审计；真实 QA A2A 补独立可信目标语音。当前 echo 不算问答能力。 |
| P1 / C1、C2 | `evaluation/sft.py`、`bridge_benchmark.py` 仅部分指标 | **G4D 首次 B1 Benchmark 前**：Text/I2T/A2T/QA A2A prompt-only 生成评测、统一评分/失败覆盖率与基础效率计时。G7 是复核，不能拖到 G7 才首次测 B1。 |
| P1 / A4，规模条件 | 当前准备/加载整个 split 到内存 | 扩容前实测 RAM/磁盘；超过可容纳 Pilot 才补分片/懒加载和可恢复预处理。8 条 G3 不要求重写数据框架。 |
| P2 / 公开交付 | 既有 G3 文件移动、P5 未跟踪材料；项目自身许可未指定 | 用户单独决定历史文件归档及许可；不能混入本轮提交。完整公开 G7 交付前解决，不阻断私有的干净提交 G3。 |

A3 的 dry-run 工程缺口本轮关闭。没有将 A1/A2/C1 或正式 CUDA 验收改为完成。

### 2 云端待验证

| 阶段 | 当前未测内容 | 通过条件 |
| --- | --- | --- |
| G3 | 真实 1.7B + 4×768、CUDA/BF16、实际媒体加载、资源/存盘/恢复 | 有限 Loss/梯度、预期参数更新、冻结摘要不变；同配置连续与新进程 Resume 的状态/下一步对照；峰值显存/RAM/磁盘、吞吐和失败记录。 |
| G4 | 原生 Talker 收敛、视觉/语音条件作用、LoRA 联合冲突、可信 QA | 获批数据、独立验证和 B1 首轮 Benchmark；实际可懂度/自然度与文本保留，不能以 Tiny Loss 下降代替。 |
| G5 | 同一 B1 的重训练公平消融 | 数据/预算/种子/评测协议一致，报告额外参数、训练与推理成本、失败/负结果；最终候选等 B1 后决定。 |
| G6 | 真实 Reward、Mimi/ASR、长序列概率重放与 Reference 资源 | 三模式实际 CUDA 更新/恢复，奖励有效性、失败样本、独立质量对照；BF16 数值失败先定位，不直接放宽既有断言。 |
| G3–G7 | 环境锁、持续运行、备份和跨阶段来源 | 保存完整配置/模型与数据指纹、真实软件/硬件、原始指标和所有祖先清单；正式 B1 从 v3 身份建立，不能把诊断旧缓存重签为正式数据。 |
| G7 | 跨方法汇总、统一实时性能、Demo/人评/最终报告 | 按 [G7 手册](../cloud/G7_FINAL_RUNBOOK.md) 实现尚缺入口并执行；当前仅有原始产物基础，不声称已经自动贯穿全部阶段。 |

### 3 可选优化

Triton、DDP/FSDP/ZeRO、双卡 Rollout/训练分离、激活重算、额外 Backbone 和服务 UI 均按 G3/G4 实测需要另行决定。当前不支持的 gradient checkpointing 会明确拒绝；不能用 `torchrun` 冒充已有多卡支持。

## 五、G3 资产、许可证与环境

### 1 必要资产

| 资产 | 云端位置 / 固定依据 | 当前事实 |
| --- | --- | --- |
| Qwen3-1.7B 非 Base | `weights/qwen3-1.7b/`；revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`；`configs/assets_v2.yaml` 的逐文件 SHA | 本机目录存在，许可证 SHA 与锁一致；本轮不重新做 1.7B 推理。须搬运完整权重、Tokenizer、配置、模板和许可。 |
| SenseVoice/SigLIP2/Mimi | `weights/media-reference/model/{SenseVoiceSmall,siglip2-base-p32-256-ve,mimi}/`；`experiments/r2/20261108-cpu-fp32/manifest.json` 中资产身份 | 本机目标目录和旧外置路径均缺失；保留相容转换版本及 Processor/配置/许可，不以同名模型静默替换。即使 T2A，契约仍检查三类模型；编码只使用 Mimi。 |
| 有限原始 T2A | `outputs/g3/data/audited.jsonl` 及相对引用的 WAV；8 个独立媒体组，6 train / 1 validation / 1 test | 本轮未制作、替换或下载数据。允许复用已有合法录音，但不能重复/转码凑数或导入未证兼容的旧 codes。 |
| 初始化 | G3 R4A：随机 Talker、Fixed Bridge，Qwen 主权重冻结 | 不需要尚不存在的 B1。G5/G6 才需要真实已验收 B1 及完整祖先谱系；G3 输出不称 B1。 |

T2A JSONL 每行需非空 `sample_id/source/revision/license/task/split/prompt/answer`；目标为 `target_audio={path,sha256,media_id}`，`target_provenance=original_recording`，`target_transcript` 与 `answer` 相同。具体 schema 以 `read_sources()` 和 G4 手册为准。保留完整源清单中的未选别名；哈希不能识别未登记且字节不同的同源转码，须人工/供应流程补证。

Qwen 锁声明 Apache-2.0，本机 `LICENSE` SHA 为 `832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e`。MiniMind-O 归属保留；本地 Mimi 转换资产历史记录为 CC-BY-4.0。旧数据卡 Apache-2.0/GPL-3.0 未逐文件区分，不能据非空 `license` 字段证明可公开再分发。**媒体资产当前缺失，未重新核实其随附许可文件**；云端交付包须附发布方记录，并限定获批用途。项目自身许可尚未指定，公共发布前解决。

### 2 推荐的候选 CUDA 环境

单卡 H100 80GB、Linux x86_64 Ubuntu 22.04/24.04、Python 3.12；候选固定 **PyTorch/Torchaudio 2.8.0、cu126、Transformers 4.57.6**。这是满足项目约束的待测组合，不是 CUDA 已通过或最新版本声明。官方提供 2.8.0 的 cu126 安装组合及 Python 3.12 Linux wheel。 [PyTorch 官方版本表](https://pytorch.org/get-started/previous-versions/) · [Torch wheel](https://download.pytorch.org/whl/cu126/torch/) · [Torchaudio wheel](https://download.pytorch.org/whl/cu126/torchaudio/)

使用云服务商与 CUDA 12.6/PyTorch 配套的 NVIDIA 驱动；不要仅凭 CUDA 12.x 的最低驱动兼容表推断所有 PTX/特性可用。启动时做真实 CUDA Tensor 运算并记录驱动、runtime 和 BF16 支持。 [NVIDIA 兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)

建议初始主机 RAM 64GB、持久盘预留约 150GB；这只是为完整 FP32 状态、Optimizer、缓存及多次 Checkpoint 留余量的规划值，须在首步测实际峰值和文件大小后核定。当前每步保存完整状态，不能只按新增 Bridge 参数估算磁盘。没有核定正式训练耗时/费用。

云端目录约定：

```text
/workspace/omniplus/repo/            干净固定提交与当前 checkout 的 .venv
  weights/                          经核验的完整资产，只读使用
  .cache/g3-release/t2a-v1/          当前云环境生成的新缓存
  outputs/g3/data/                   审计 JSONL 与获批有限媒体
  outputs/g3/configs/release.yaml    实际派生配置
  outputs/g3/environment/ logs/      环境、命令、stdout/stderr
  outputs/g3/step1/ resume2/         独立运行；Checkpoint 直接在 run 根目录
```

不要复用 Mac 的缓存或另一个 checkout 的 editable install。现有严格 Resume 绑定源码、配置、资产和运行环境；搬迁不是删除校验的理由。

## 六、第一组真实终端命令

**以下是下一阶段操作说明，本轮未执行云端部分。** 先取得云资源/有限数据授权。使用不含用户脏改动的提交；不要求本轮 Push。

### 1 本地交付准确 Git 提交

在本轮提交完成后，从仓库根执行；Git bundle 只包含提交，不包含权重、数据或用户未提交文件：

```bash
set -euo pipefail
PRE_CLOUD_COMMIT=$(git rev-parse HEAD)
G3_BUNDLE="/private/tmp/omniplus-${PRE_CLOUD_COMMIT}.bundle"
test ! -e "$G3_BUNDLE"
git bundle create "$G3_BUNDLE" HEAD
git bundle verify "$G3_BUNDLE"
printf '%s\n' "$PRE_CLOUD_COMMIT"
```

由用户把 bundle 与输出的完整 SHA 私下传至云端。若用户选择自行 Push，也可从已核实的 `https://github.com/luminoso909/Omni-plus.git` 克隆后 checkout **同一 SHA**；远程没有该提交时不能假装已同步。

### 2 云端环境与来源

```bash
set -euo pipefail
: "${G3_COMMIT:?填写本地交付的完整SHA}"
: "${G3_BUNDLE:?填写已传到云端的Git bundle绝对路径}"
G3_CHECKOUT=/workspace/omniplus/repo
test ! -e "$G3_CHECKOUT"
mkdir -p /workspace/omniplus
git clone "$G3_BUNDLE" "$G3_CHECKOUT"
cd "$G3_CHECKOUT"
git switch --detach "$G3_COMMIT"
test "$(git rev-parse HEAD)" = "$G3_COMMIT"
test -z "$(git status --porcelain)"

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e '.[dev,media,encoders,plots,data]' 'transformers==4.57.6'
python -m pip check
mkdir -p outputs/g3/environment outputs/g3/logs
python -m pip freeze > outputs/g3/environment/pip-freeze.txt
git rev-parse HEAD > outputs/g3/environment/commit.txt
nvidia-smi > outputs/g3/environment/nvidia-smi.txt
export CUDA_VISIBLE_DEVICES=0
python - <<'PY'
import torch
import torchaudio
import transformers
assert torch.__version__.split('+')[0] == '2.8.0'
assert torchaudio.__version__.split('+')[0] == '2.8.0'
assert transformers.__version__ == '4.57.6'
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
x = torch.ones(2, 2, device='cuda', dtype=torch.bfloat16)
assert torch.equal(x @ x, 2 * x)
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
PY
python scripts/run_experiment.py --help
```

如安装解析改变固定版本或 CUDA 测试失败，停止并记录，不能直接继续。本轮没有在 Mac 安装 CUDA 依赖或修改本地 Torch 版本。

### 3 准备获批资产与 8 条数据

发布方包必须已经过来源/许可审核，有相对路径 `SHA256SUMS`，包含 `weights/qwen3-1.7b`、`weights/media-reference`、`data/audited.jsonl` 和引用媒体。摘要只证明搬运一致，不替代合法性/原始媒体身份审核。

```bash
: "${G3_ASSET_BUNDLE:?填写已批准且含SHA256SUMS的私有资产包目录}"
(cd "$G3_ASSET_BUNDLE" && sha256sum -c SHA256SUMS)
test ! -e weights/qwen3-1.7b
test ! -e weights/media-reference
test ! -e outputs/g3/data
mkdir -p weights outputs/g3/data
rsync -a --checksum "$G3_ASSET_BUNDLE/weights/qwen3-1.7b/" weights/qwen3-1.7b/
rsync -a --checksum "$G3_ASSET_BUNDLE/weights/media-reference/" weights/media-reference/
rsync -a --checksum "$G3_ASSET_BUNDLE/data/" outputs/g3/data/
python - <<'PY'
import hashlib
from pathlib import Path
import yaml
manifest = Path('outputs/g3/data/audited.jsonl')
with manifest.open('rb') as stream:
    checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
path = Path('outputs/g3/configs/release.yaml')
path.parent.mkdir(parents=True, exist_ok=True)
with path.open('x') as stream:
    yaml.safe_dump({
        'extends': '../../../configs/v2/g3_release.yaml',
        'data': {'audited_manifest_sha256': checksum},
    }, stream, sort_keys=False)
PY
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
python scripts/run_experiment.py inspect --config outputs/g3/configs/release.yaml > outputs/g3/logs/inspect.json
python scripts/run_experiment.py prepare --config outputs/g3/configs/release.yaml --dry-run > outputs/g3/logs/prepare-preflight.json
python scripts/run_experiment.py prepare --config outputs/g3/configs/release.yaml > outputs/g3/logs/prepare.json
```

`inspect/prepare --dry-run` 检查配置、资产与原始数据契约，不执行 forward，也不证明 GPU 可用。`prepare` 才真正编码并保存云环境缓存。上述派生 YAML 的相对 `extends` 以配置文件目录解析，其余资产路径按仓库根解析。

### 4 有限首步与新进程恢复

```bash
CFG=outputs/g3/configs/release.yaml
python scripts/run_experiment.py smoke --config "$CFG" --steps 1 --dry-run > outputs/g3/logs/smoke-preflight.json
python scripts/run_experiment.py smoke --config "$CFG" --steps 1 --run-dir outputs/g3/step1 > outputs/g3/logs/step1.log 2>&1
python scripts/run_experiment.py resume --config "$CFG" --resume outputs/g3/step1/step-000001.pt --steps 2 --dry-run > outputs/g3/logs/resume-preflight.json
python scripts/run_experiment.py resume --config "$CFG" --resume outputs/g3/step1/step-000001.pt --steps 2 --run-dir outputs/g3/resume2 > outputs/g3/logs/resume2.log 2>&1
```

检查两个 run 的 `manifest.json`：Loss/梯度有限、`frozen_before == frozen_after`、训练参数摘要变化、恢复段 `start_step=1,step=2`，实际设备为 CUDA。Checkpoint 必须包含模型、Optimizer/Scaler/RNG、Scheduler/采样游标和身份。`--steps` 是含已恢复步数的目标总步数。**Checkpoint 位于 run 根目录，不是 `checkpoints/` 子目录。**

在剩余 G3 预算允许时，用相同配置/种子另开 `smoke --steps 2 --run-dir outputs/g3/continuous2`，比较连续与恢复的模型、Optimizer/Scheduler/RNG/游标及下一步。先验证恢复，再另行批准更长 Profile；可用实际入口：

```bash
python scripts/run_experiment.py profile --config "$CFG" --steps 10 --run-dir outputs/g3/profile10 > outputs/g3/logs/profile10.log 2>&1
```

所有 run 累计计算预算，10 步是单次配置上限，不是无限次重试许可。100 步需新配置/授权，不能修改旧配置冒充 Resume。Profile 计时包括验证和 Checkpoint IO，不等于纯 kernel 吞吐。

### 5 保存与回传

保留每个 run 的 `resolved_config.yaml`、`manifest.json`、`metrics.jsonl`、`loss.svg`、`step-*.pt`、`best-*.pt`（若产生）和 `adapter.pt`。保留缓存 manifest、完整数据清单/别名和许可、Git/软件锁/执行日志；G5/G6 还需 B1 与所有祖先的 Manifest/Checkpoint/lineage，不能仅传单个权重文件。

待所有写入进程结束后冻结 `outputs/g3`，按以下系统命令创建相对路径摘要。新接收目录不得覆盖旧实验：

```bash
test ! -e outputs/g3/SHA256SUMS
test ! -e outputs/g3/prepared-cache
cp -R .cache/g3-release/t2a-v1 outputs/g3/prepared-cache
(cd outputs/g3 && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
: "${G3_RETURN_DIR:?填写全新的私有挂载或回传目录}"
test ! -e "$G3_RETURN_DIR"
mkdir -p "$G3_RETURN_DIR"
rsync -a --checksum outputs/g3/ "$G3_RETURN_DIR/"
(cd "$G3_RETURN_DIR" && sha256sum -c SHA256SUMS)
```

`prepared-cache` 包含本次有限 8 条数据的缓存 Tensor、selections 和原 manifest；只复制日志会漏掉严格恢复所需的缓存证据。恢复时在同一固定环境按原配置还原缓存相对路径并核验摘要，不重新编码后重签身份。macOS 接收端可在包内运行 `shasum -a 256 -c SHA256SUMS`。远程传输时由用户使用实际主机/存储路径执行 rsync/scp，校验后再清理获批临时副本。Checkpoint/数据不入 Git；只把无敏感信息的小规模摘要、结果表和阶段报告提交。失败/OOM/缺指标同样保留，不把未执行项写零值或已通过。

## 七、本轮实际验证

本机 arm64，Python 3.12.14、PyTorch 2.15.0、Transformers 4.57.6；CPU 离线 fixture，沙箱 MPS 不可用。以下仅记录本轮实际执行，不借用历史全仓结果充数。

```bash
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
export MPLCONFIGDIR=/private/tmp/omni-precloud-k1250h07/mpl
.venv/bin/python -m pytest -q -p no:cacheprovider --basetemp=/private/tmp/omni-precloud-k1250h07/pytest
.venv/bin/ruff check --no-cache .
.venv/bin/ruff format --check --no-cache omni tests scripts
.venv/bin/python -m pip check
git diff --check
```

| 实际检查 | 结果 |
| --- | --- |
| 当前工作区全仓 pytest | **441 passed, 1 skipped in 126.87s**；唯一跳过为沙箱 MPS 不可用。 |
| `git archive 30bb068` 加本轮白名单源码/测试/配置的干净交付树 | **434 passed, 1 skipped in 127.93s**；排除用户既有未提交修改，仅跳过 MPS。 |
| 新预检和 R3/R4/R5 定向回归 | **76 passed in 45.89s**，其中新预检 42 项；实际命令为 `python -m pytest -q tests/test_precloud_preflight.py tests/test_r3_training.py tests/test_r4_training.py tests/test_r5_training.py`。随后包含在全仓回归中。 |
| Ruff / format / pip check / diff | 全部通过；89 个 Python 文件格式符合；无损坏依赖。pip 的本机缓存权限提示不影响依赖检查。 |
| CLI 与云手册 | `run_experiment.py --help`、`bridge_ablation.py plan/collect --help`、`hf download --help` 均 exit 0；原 G3–G7 的 36 个 shell 代码块及本报告共 **44 个 shell 块** 通过 `bash -n`，嵌入 Python 通过 AST 解析；参数/输出路径另与源码核对。这不等于云端命令实测。 |
| G3 YAML / 缺资产闸门 | `read_settings` 解析为 schema 2、Qwen/CUDA BF16、8 条 T2A/10 步/save1/eval2。实际 `python scripts/run_experiment.py inspect --config configs/v2/g3_release.yaml` exit 1，明确缺 `outputs/g3/data/audited.jsonl`，未创建 run；不把解析或预期拒绝称为真实资产通过。 |

G3 配置另实际完成 `read_settings → build_model(meta=True) → configure_parameters`：Qwen 1,720,574,976 参数全部冻结；Bridge 1,573,632、Talker 30,680,834、Audio Embedding/Heads 各 7,520,256 可训练，共 **47,294,978**。两个输入 Projector 和 Speaker 冻结。全部参数均为 meta；不计外部编码器/Mimi，也不代表权重加载或显存实测。

干净交付树复验的实际命令（从 `/private/tmp/omni-precloud-k1250h07/release` 执行）：

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 \
MPLCONFIGDIR=/private/tmp/omni-precloud-k1250h07/mpl-clean \
PYTHONPATH=/private/tmp/omni-precloud-k1250h07/release \
GIT_DIR=/Users/lucius/projects/personal/Omni++/.git \
/Users/lucius/projects/personal/Omni++/.venv/bin/python -m pytest -q -p no:cacheprovider \
  --basetemp=/private/tmp/omni-precloud-k1250h07/clean-pytest
```

轻量反例证据位于系统临时目录 `omni-precloud-a3-before-after-fz5nx9cg`；永久回归随代码提交。本轮全仓日志与保护基线位于 `/private/tmp/omni-precloud-k1250h07`，临时目录可能被系统清理，长期事实以上表及已提交测试为准。未执行真实 1.7B/Mimi/ASR、CUDA 或远程 CI。

## 八、文件范围与历史保护

本轮白名单：`.github/workflows/ci.yml`、`configs/v2/g3_release.yaml`、`omni/training/r3_local.py`、`scripts/run_experiment.py`、`tests/test_precloud_preflight.py`、本报告、README、项目待办、G6/G7 手册。无模型核心、依赖定义或历史阶段报告修改。

开工基线记录 237 个已有源码/文档等文件的字节 SHA/缺失状态，以及 710 个既有输出/实验/权重/缓存文件的大小与 mtime。复核 230 个非白名单已有文件状态完全不变，7 个已有白名单文件按计划修改，710 个资产/实验文件大小与 mtime 均不变；另外只新增配置、测试和本报告。用户的旧模型/推理/测试改动、Notebook、G3 文档移动、P5 未跟踪文件等保持原状且不暂存。

P0–P5 下列 SHA 与开工及 R6 历史保护记录一致；R0–R6-L 阶段报告和原审查报告也保留原字节。

| 文件 | SHA-256 |
| --- | --- |
| P0.md | `01ac1cb1d4bad5c2d938fc5ad60f06d68ec49e11060e01442f38e0d31fcd993f` |
| P1.md | `ecb9052d9162d77d632210c2d5bf18c7ff5b52fcda5fe8a70a6be6df51757473` |
| P2.md | `b1345c087866ad07026c33f0506d82d5a633e8c3ffb33e1eeb8dbe393cbe04b3` |
| P3.md | `14ed2a69dd9340f4373ce35b02f750d7fd372cad5dfeedb9777de46aebcf811a` |
| P4.md | `37b80cf84f4c512ab916cc43a03ae2e761e98a8fa6aabdd3c865ebb01df9040f` |
| P5.md | `302dd9b5ba88afc4ed21079af94777b2960052c2b9aa3ce6992c6dc4196537ed` |

本轮仅本地提交，不 Push。停止在 G3 部署入口；正式训练、研究结论与公共发布均未执行。
