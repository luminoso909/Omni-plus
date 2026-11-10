# G3｜单卡云端验证操作说明

本说明准备未来 1×H100 80GB 的真实 Qwen3-1.7B + 4×768 Talker 验收。本轮仅验证本地配置、CLI 和 tiny 恢复；CUDA、显存、吞吐、100 步 Profile 与正式训练均未执行。G3 须单独授权。

## 一、代码与环境

当前远端为 `https://github.com/luminoso909/Omni-plus.git`。本地阶段不会自动 Push；先由用户推送已验收提交，或用 Git bundle 搬运同一提交，再在云端固定 checkout。不能直接拿远端旧版本声称复现 R3-L。

```bash
git clone https://github.com/luminoso909/Omni-plus.git /workspace/Omni-plus
cd /workspace/Omni-plus
# G3 启动前设置为用户选择的完整已验收提交 SHA。
: "${OMNI_COMMIT:?设置已传至云端的完整 Git SHA}"
git checkout --detach "$OMNI_COMMIT"
git status --short
python3.12 -m venv .venv
source .venv/bin/activate
```

使用适合镜像驱动的 CUDA PyTorch 与同版本 torchaudio；安装命令从 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 的 Linux/Pip/CUDA 选项取得。不要复制 macOS 的 wheel，也不要把已有 Mac 的 `pip freeze` 当作已验证的 CUDA 环境。接着安装本项目并记录真实解析结果：

```bash
python -m pip install -e '.[dev,media,encoders,plots,data]'
python -m pip check
mkdir -p outputs/g3/environment
python -m pip freeze > outputs/g3/environment/requirements.txt
nvidia-smi > outputs/g3/environment/nvidia-smi.txt
python - <<'PY'
import torch
assert torch.cuda.is_available(), "需要可用的 CUDA PyTorch"
assert torch.cuda.device_count() == 1, "本次验收限单 GPU"
print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_device_name(0))
assert torch.cuda.is_bf16_supported(), "候选配置要求 BF16"
PY
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 pytest -q
ruff check omni tests scripts
ruff format --check omni tests scripts
```

当前依赖约束见 `pyproject.toml`。云端包版本改变时，须根据缓存版本检查结果重建对应缓存，并对照 R2 数值/接口；本说明不预先声称任何具体镜像已经兼容。

## 二、资产与有限样本

| 资源 | 位置/来源 | 准备与核验 |
| --- | --- | --- |
| Qwen3-1.7B | `weights/qwen3-1.7b/` | R1 的 revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`，完整 HF 目录；核对 `configs/assets_v2.yaml` 文件锁 |
| SenseVoice/SigLIP2/Mimi | `weights/media-reference/model/` 下原目录名 | 从已验收本地资产私有传输；不是任意同名的新官方模型。检查权重、配置和 processor 指纹 |
| 有限原始样本 | `cache/p4/sources-v2/` 的三份小 Parquet | 已有 96 条候选与原始来源/许可；按历史摘要检查文件 SHA，不下载完整数据集 |
| 小样本缓存 | `.cache/r3-local/tiny-v1/` 或新的专用路径 | 配置和 processor/编码器/tokenizer 版本匹配才可复用；不匹配则显式用新目录重建 |
| 数据和模型清单 | `experiments/p4/`、`experiments/r2/`、`experiments/r3-local/` | 随 Git 带入摘要，不带原始图片/语音或权重 |

Qwen 资产已在本地时优先私有传输。若 G3 另行批准补下载，已验证的本机 `hf download --help` 支持以下固定 revision 命令：

```bash
hf download Qwen/Qwen3-1.7B \
  --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --local-dir weights/qwen3-1.7b
```

数据卡声明 Apache-2.0 与 GPL-3.0，未逐文件区分；本地诊断子集不作为公开发布数据包。导入的历史回答声码没有逐样本 Mimi 编码版本，R3-L 将其隔离；可训练的诊断音频来自原始语音经本地锁定 Mimi 重新编码。A2A 复述诊断不等同开放式语音问答训练。

## 三、预检与执行入口

`configs/v2/g3_h100.yaml` 与本地 tiny 配置分开：原生 1.7B、四层 768 Talker、CUDA BF16、T2A、冻结 Qwen/Projector、训练 Bridge/Talker/Embedding/Heads、最多 10 步。按实际云盘修改资产与缓存路径，保留解析后的配置。

```bash
python scripts/run_experiment.py --help
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python scripts/run_experiment.py inspect \
  --config configs/v2/g3_h100.yaml
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python scripts/run_experiment.py prepare \
  --config configs/v2/g3_h100.yaml --dry-run
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python scripts/run_experiment.py prepare \
  --config configs/v2/g3_h100.yaml
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python scripts/run_experiment.py smoke \
  --config configs/v2/g3_h100.yaml --steps 1 --dry-run
```

`inspect` 的参数统计来自 meta 模型，只证明形状和冻结分组；`--dry-run` 不执行 forward、不分配训练模型，不证明 CUDA 设备能运行。上述资源存在且用户批准 G3 后，才运行下列真实 GPU 命令：

```bash
mkdir -p outputs/g3/logs
set -o pipefail
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python scripts/run_experiment.py smoke \
  --config configs/v2/g3_h100.yaml --steps 1 --run-dir outputs/g3/step1 \
  2>&1 | tee outputs/g3/logs/step1.log
```

完成后从该 run 的 manifest 读取实际 checkpoint 路径，以 `smoke --resume <checkpoint> --steps 10 --run-dir outputs/g3/resumed10` 在新进程延续；总步数包含已完成步数。恢复写新 run，保留旧 Checkpoint 和日志。当前完整 checkpoint 包含冻结 Qwen，本地不会伪称只保存新模块；云端先测实际文件大小并预留足够持久存储。

`train / resume / evaluate / profile` 独立子命令仍属于后续统一接口；本阶段只实现 `smoke --resume` 的恢复验收。不要把手册中的目标命令当成已有功能。可选 100 步 Profile 应在 G3 获准后实现采集与预算配置，先验证当前 1–10 步路径，不能把普通 `smoke` 计时冒充完备 Profile。

## 四、G3 必须补交的证据

| 验收 | 记录内容 |
| --- | --- |
| 设备与预算 | GPU 型号/数量、驱动、CUDA/PyTorch、BF16 支持、持久盘和用户批准的最大步数/时长 |
| 真实训练 | 1–10 步 Loss、非零梯度、冻结参数不变、训练参数变化、有效 token/frame 与 batch 分布 |
| 独立恢复 | 新进程恢复模型/optimizer/scheduler/scaler/RNG/epoch/cursor；和不中断运行对照 |
| 资源测量 | CUDA 同步后的 step wall time、峰值 allocated/reserved 显存、samples/s、checkpoint 大小和保存开销 |
| 归档 | Git SHA、源码/配置/数据/资产摘要、运行日志、成功与失败记录、运行目录 |
| 扩大运行 | 仅在上述通过后，按新批准预算采集 100 步数据并估算正式训练成本 |

## 五、结果回传

下面命令由用户在本地终端执行；保持私有传输，源和目标目录不同，禁止覆盖历史结果：

```bash
: "${OMNI_SERVER:?设置用户批准的 SSH host}"
mkdir -p outputs/cloud-results/g3
rsync -av --partial --exclude='*.pt' \
  "$OMNI_SERVER:/workspace/Omni-plus/outputs/g3/" outputs/cloud-results/g3/
```

大型 Checkpoint 单独传入忽略目录并核对 SHA-256。Run manifest、日志和统计经检查后可归档到新的 `experiments/g3/`；原媒体、权重和 checkpoint 保持不提交。G3 完成之前，所有 CUDA 性能与正式训练任务继续标为未执行。
