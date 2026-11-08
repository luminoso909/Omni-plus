## 一、适用范围与验收边界

这是 R4-L 已实现入口的云端操作说明。Mac 本地只验证 Tiny 更新、真实小样本 IO、数学与恢复；**G3、G4-A/B/C/D 均未执行，B1 尚未成立**。先取得云任务授权并完成 G3，再按下面顺序运行。H100 配置默认只有 10 步 Pilot，不代表批准正式训练规模。

| 任务 | 配置 | 更新范围 | 正式验收仍需完成 |
| --- | --- | --- | --- |
| G4-A | `configs/v2/r4a_h100.yaml` | Bridge、Talker、Audio Embedding/Heads | 真实 1.7B + 4×768 Talker 小集合过拟合、有效声码与生成 |
| G4-B | `configs/v2/r4b_h100.yaml` | 同 A；接续 A 或经审计的兼容初始化 | 长短句、中英文、验证集选择、可懂语音及人工试听 |
| G4-C Vision | `configs/v2/r4c_vision_h100.yaml` | 仅 Vision Projector | 图像输入对回答的贡献、独立验证与模态调换 |
| G4-C Audio | `configs/v2/r4c_audio_h100.yaml` | 仅 Audio Projector | 真实转写/听音对齐；梯度穿过冻结 Qwen |
| G4-D | `configs/v2/r4d_h100.yaml` | 两个 Projector、Bridge、Talker/Heads/Embedding、Qwen LoRA | 多任务联合训练、可信 QA A2A、冻结 B1 并做首轮 Benchmark |

所有任务都复用 `scripts/run_experiment.py → omni/training/r3_local.py → sft.train_step`。不另建训练器。普通测试不加载大型资产；只有显式训练、恢复或 Profile 命令执行参数更新。

## 二、固定代码与安装环境

当前 R4 提交只在本地。上传前先由用户 Push，或用 `git bundle` 搬运；远程未收到该提交时，不能把旧仓库当作 R4 版本。

```bash
git clone https://github.com/luminoso909/Omni-plus.git
cd Omni-plus
# 将 R4_COMMIT 设置为本次本地提交的完整 SHA，确认远程已经包含它。
git switch --detach "$R4_COMMIT"
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,media,encoders,plots,data]'
python -m pip check
python scripts/run_experiment.py --help
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

选用与云主机驱动对应的 CUDA PyTorch/Torchaudio 组合，安装方法以 [PyTorch 官方页面](https://pytorch.org/get-started/locally/) 为准。不要复制 Mac 的 wheel 或把 CPU wheel 当作 GPU 环境。将 `pip freeze`、`nvidia-smi`、Python/Torch/Transformers 版本保存到独立 run 旁；本地版本记录不是 Linux CUDA 兼容性证明。

## 三、权重、数据与缓存

### 1 固定权重

- Qwen：`weights/qwen3-1.7b/`，官方非 Base，revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`；完整 HF 目录和 Qwen 原生 tokenizer/template。文件摘要见 `configs/assets_v2.yaml`，入口逐个核验。
- `assets.reference_root` 下保留 `model/SenseVoiceSmall`、`model/siglip2-base-p32-256-ve`、`model/mimi`。使用 R2 已锁定的本地转换资产，按许可私下搬运；不能用名字相同、内容不同的下载模型直接顶替。
- 可选 ASR 使用完整本地 SenseVoice 目录。`evaluation.asr_dir` 为空时 CER/WER 为未测量；不写 0。ASR 只接收预测波形，参考答案不传入解码器。
- 准备和训练入口都离线加载，不会下载缺失模型。资产缺失/哈希不同直接失败。

### 2 审计后的统一数据清单

新增入口接受**已审计、已规范化的本地 JSONL**，可标记 MiniMind-O、LJSpeech、AISHELL-3 等来源；它不是这些站点的自动下载器或原生压缩包解析器。先按获批预算下载、核实许可、整理媒体及标签，再建立清单。完整数据下载与原问答式 A2A 的目标录音补证仍待执行。

每行必须含：`sample_id/source/revision/license/task/split/prompt/answer`。`task` 为 `t2a/i2t/a2t/a2a/text`；`split` 为 `train/validation/test`。媒体字段按任务选择 `image/input_audio/target_audio`，每个字段均为 `{path, sha256, media_id}`；相对路径从 JSONL 所在目录解析。

```json
{"sample_id":"audited-id","source":"your-reviewed-source","revision":"fixed-version","license":"verified-license","task":"t2a","split":"train","prompt":"请读出下面这句话。","answer":"已核实的录音转写。","target_audio":{"path":"audio/recording.wav","sha256":"填入实际文件SHA256","media_id":"来源命名空间:原始媒体ID"},"target_provenance":"original_recording","target_transcript":"已核实的录音转写。"}
```

这只是字段示例，替换为真实摘要和授权记录后才能使用。目标录音必须有 `target_provenance: original_recording`，`target_transcript` 与 `answer` 相同；这依赖清单编制者做语义核验，程序的文件摘要不能证明转写真实性。问答式 A2A 另需 `a2a_kind: qa`，输入和目标录音的媒体 ID/字节必须不同。复述仅能标 `echo`；云端 D 默认禁止使用它。

同一原始媒体 ID 或 SHA 跨角色/来源合并为同组，不能跨 split。优先保留官方划分，固定 Seed；每任务预算至少保留 3 条以覆盖 train/val/test。输入音频独立重采样至 16 kHz 供 SenseVoice；目标波形重采样至 24 kHz 后用锁定 Mimi 生成 8 路 codes。未知版本旧 codes 不进入此入口。

### 3 创建自己的实验配置

复制并修改一份 YAML 放在 `outputs/g4/configs/`，使用 `extends: ../../../configs/v2/r4a_h100.yaml`（其它阶段替换文件名）。相对资产/缓存路径从仓库根目录解析，`extends` 相对当前 YAML 解析。

```yaml
extends: ../../../configs/v2/r4a_h100.yaml
assets:
  qwen_dir: weights/qwen3-1.7b
  reference_root: weights/media-reference
data:
  audited_manifest: outputs/g4/data/t2a.jsonl
  audited_manifest_sha256: 填入实际JSONL文件SHA256
  cache_dir: .cache/g4/t2a-v1
  tasks: {t2a: 16}
  seed: 42
  max_length: 1200
training:
  steps: 10
  batch_size: 1
  accumulation_steps: 2
  eval_interval: 2
  save_interval: 2
evaluation:
  max_new_tokens: 128
  asr_dir: weights/media-reference/model/SenseVoiceSmall
```

这份清单、媒体和缓存留在 Git 忽略目录。完整配置必须在训练之前固定；正式步数、学习率、长度上限、batch 和数据预算由 G3 结果确定。字段拼写错误、未知任务、非法码本和版本不匹配都应明确失败。Mac 缓存包含处理器版本与原始路径契约，搬到云端需按该环境重新 `prepare`，不能删除版本校验。

## 四、真实命令与阶段衔接

以下 `RUN_CONFIG` 指向已经审核并准备好本地文件的实验 YAML，`RUN_DIR` 必须是全新目录。每条命令都由现有 CLI 处理。

```bash
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
RUN_CONFIG=outputs/g4/configs/g4a.yaml
python scripts/run_experiment.py inspect --config "$RUN_CONFIG"
python scripts/run_experiment.py prepare --config "$RUN_CONFIG" --dry-run
python scripts/run_experiment.py prepare --config "$RUN_CONFIG"
python scripts/run_experiment.py train --config "$RUN_CONFIG" --dry-run
python scripts/run_experiment.py profile --config "$RUN_CONFIG" --steps 2 --run-dir outputs/g4/a-profile-001
python scripts/run_experiment.py train --config "$RUN_CONFIG" --run-dir outputs/g4/a-001
python scripts/run_experiment.py evaluate --config "$RUN_CONFIG" --resume outputs/g4/a-001/step-000010.pt --run-dir outputs/g4/a-eval-001
python scripts/run_experiment.py demo --config "$RUN_CONFIG" --resume outputs/g4/a-001/step-000010.pt --run-dir outputs/g4/a-demo-001
```

`profile` 实际进行更新并保存 checkpoint，测量整段更新循环（包括期间验证/存盘）的时间与 CUDA 峰值显存，最多 100 步。它不是 kernel profiler，不把含存盘的平均时间叫纯训练吞吐。`--steps` 是目标总步数，上限由配置决定；运行 100-step Profile 需先将配置预算显式设为 100 并取得授权。

### 1 A/B：初始化、验证与音质

默认 `model.initialization.talker: random`。`talker: pretrained` 还需 `talker_checkpoint`，只接受与当前 Talker 完全同键/形状的状态；不自动改 MiniMind 权重名或丢掉不匹配层。

B 接续 A 使用 `components_checkpoint` 导入 A 的多模态组件，创建新的训练 run 和优化器。完整 Resume 则必须使用原配置与 `--resume`，二者不可混淆。验证周期 `eval_interval`、保存周期 `save_interval`、`patience/min_delta` 决定最佳状态与早停；`patience: 0` 禁用早停。最终模型不一定是验证最好的模型，按 manifest 的 `best_step` 选择对应 `best-XXXXXX.pt`。

`evaluate` 输出各码本 CE、有效 token 数、STOP precision/recall；`demo` 从 prompt 自回归生成并解码预测 codes，写出 WAV、文本、ASR 与失败原因。生成达到预算、没有音频帧、ASR 空转写等必须保留。当前流式 API 不暴露每路音频 STOP 明细，评测明确标记该项不可观测，不能将预算结束当成自然停止。试听另行填写清晰度、内容一致性、噪声和截断，Tiny 生成文件不证明有可懂语音能力。

### 2 C：两路 Projector 分别训练

Vision 配置只选 I2T、Audio 配置只选 A2T。它们仅解冻各自 Projector，沿冻结 Qwen 的计算图反传，不能用整段 `no_grad()`。分别保存在独立 run；可从 B 的全组件状态起步。`evaluate` 用相同形状、不同内容的两个媒体特征做调换，只测监督回答位置；验证集不足两条或形状不同则报告未测量，不裁切媒体冒充对照。

### 3 D：联合训练与 Adapter

先用 `model.initialization.components_checkpoint` 导入 B 的全组件，再用 `vision_checkpoint` 和 `audio_checkpoint` 分别导入 C-Vision、C-Audio 的 Projector，避免一个 C checkpoint 覆盖另一条已训练路径。正式 D 的 `allow_diagnostic_a2a` 保持 `false`；缺少可信 QA 目标录音时不得把本地 echo 冒充完整任务验收。

`sample_weights` 控制确定性任务轮转比例，`task_batch_sizes` 控制各任务 batch，`task_weights` 控制 Loss；三者分开配置。`group_lrs` 可分别设置 Bridge、Talker、Embedding、Heads、Projectors 和 LoRA 的 LR。LoRA 明确注入原生 Qwen 的 `q_proj/v_proj` 等指定 Linear，默认本地 rank=4、alpha=8、dropout=0；正式 rank/targets/LR 待 Profile。`gradient_checkpointing: true` 当前会明确拒绝，不能把未生效的开关当作已支持。

每个新 run 保存 `adapter.pt`，其中含 LoRA 与全部多模态模块（含冻结组件），不含 Qwen 主权重。独立推理导入使用 `omni.training.adapters.load_adapter`，必须绑定相同基础资产、架构和 LoRA 配置；Tiny 还需相同基座 seed。它不包含优化器，不能代替训练 Resume。

## 五、恢复、保存与结果回传

中断示例：原配置总预算为 10，首次显式运行到第 2 步，再在**新目录**恢复到第 10 步。

```bash
python scripts/run_experiment.py train --config "$RUN_CONFIG" --steps 2 --run-dir outputs/g4/a-part-001
python scripts/run_experiment.py resume --config "$RUN_CONFIG" --resume outputs/g4/a-part-001/step-000002.pt --steps 10 --run-dir outputs/g4/a-resume-001
```

完整 checkpoint 保存 model、LoRA、optimizer、scheduler、scaler、Python/NumPy/Torch/设备 RNG、任务轮转/分桶游标、验证历史与早停状态。源码、解析配置、数据清单、资产或运行时不一致会拒绝恢复；迁移环境应重新执行兼容性验证，不删校验绕过。

| 文件 | 保留用途 |
| --- | --- |
| `resolved_config.yaml`、`manifest.json` | 参数、来源身份、Git、设备、梯度、冻结摘要与测量条件 |
| `metrics.jsonl`、`loss.svg` | 实际训练过程与验证曲线 |
| `step-*.pt`、`best-*.pt` | 完整恢复与验证最佳状态；禁止覆盖旧 run |
| `adapter.pt` | Qwen 基座之外的独立模型组件 |
| `evaluation.json`、`audio/prediction.wav` | 真实验证、生成与 ASR，另附人工试听记录 |

将 stdout/stderr 重定向到 run 旁的唯一日志文件，失败也保留；训练中按获批备份策略把 checkpoint 同步至私有持久卷。回传配置、版本、源数据摘要、日志、指标、试听清单、checkpoint/adapter 及校验值。Git 只提交小规模摘要与报告，原始语音、缓存、权重、checkpoint 不提交。先独立验证恢复，再清理临时云资源；本次未连接云 GPU，也没有云端预算或音质实测结论。
