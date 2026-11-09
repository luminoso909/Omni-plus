## 一、结论与范围

**R6-L PASS（本地代码、数学与 Tiny 工程验收）**。2026-11-09，基于 `c07f1dff5ed564c390f962754b53bf3962ee5077`，完成设计、实现、同轮自审、修复和最终回归。不将本轮自审称为外部独立验收；未启动 G3–G7、CUDA、正式训练或下载大型数据。

前置检查确认 R5 最终报告的两项 P1 已关闭，P2 成本实现保持原样。本轮将完整媒体身份与实际源码根检查复用到 RL，新增正反向回归。开工及执行中已有用户改动均排除出交付，历史报告保持只读。

| 目标 | 实际结果 |
| --- | --- |
| Text-only / Audio-aware / Joint | 同一更新器、真实 Tiny Qwen/Talker 采样、概率重算、Reward 和参数更新通过 |
| 数学 | 组优势、PPO 分段梯度、精确 KL、八码本概率乘积、长度归一化有独立期望值与差分检查 |
| 模型边界 | LoRA、两个 Projector、Fixed/Weighted/Gated Bridge 与 Talker 梯度通过；冻结 Qwen 不变 |
| 工程 | 实际 CLI、保存/跨进程恢复、Adapter、数据隔离、源码身份、失败记录通过 |
| 正式研究 | B1 未成立；正式 Reward、音质、CUDA/BF16、资源与多卡均未验收 |

## 二、算法与接口

完整公式、Shapes、时间轴和手算例见 [GRPO_MATH.md](../GRPO_MATH.md)。Qwen 和八路音频头在同一历史状态上预测并行动作；按物理决策步相加文本与有效音频 logprob，绝不把八路平均 CE 当作联合概率。文本 EOS 后允许音频继续；延迟/已停止 PAD 不计概率，EOS/STOP 本身计入；已采样的残缺帧动作仍计 Loss，解码只输出连续完整帧。

Old Logprob 在采样时 detached 保存，一轮采样结束后才能更新。同批多个 epoch 固定 Old；Reference 固定为初始 SFT，Tiny 可用随机初态。Reference 仅保留可变参数 CPU 快照，共享不可变底座；先无梯度参考前向，恢复 Current 后建图并完成本条反向，再处理下一条。Dropout 关闭，但 Current 保持 autograd。

采样、Old、Current、Reference 均使用相同合法支持集及正温度。Top-k/Top-p 截断、零温度明确拒绝。KL 使用精确分类 `KL(current || reference)`，按有效头相加，保留 Current 概率与 logprob 的梯度；这是旧轨迹访问状态上的条件正则，不冒充完整新策略轨迹 KL 的无偏估计。

`sequence_mean` 每条轨迹等权；`token_mean` 每个有效物理步等权，累积窗口使用全局分母。零方差优势为零；无有效动作、非有限 Reward/概率/梯度拒绝。整批零梯度标为 `no_update`，不靠 AdamW 衰减制造更新。单设备优先，未实现梯度检查点或多卡。

## 三、参数范围与配置

六份 YAML 为 `configs/v2/r6_{text,audio,joint}_{local,h100}.yaml`。现有 `run_experiment.py` 路由到 `r6_local.py`，继续使用 SFT 的模型构建、任务采样、Optimizer/Scheduler、完整 Checkpoint 与 Adapter。没有另造模型或平行 SFT 框架。

| 模式 | 默认可训练组 | Tiny 32 维 / 原生 meta 可训练参数 |
| --- | --- | ---: |
| Text | Qwen LoRA | 896 / 802,816 |
| Audio | Bridge、Talker、音频 Embedding/Heads | 450,018 / 47,294,978 |
| Joint | 上述音频组与 LoRA | 450,914 / 48,097,794 |

原生 meta 是配置统计，**不是加载/训练验收**：Qwen 1,720,574,976，Fixed Bridge 1,573,632，Talker 30,680,834，音频 Embedding/Heads 各 7,520,256；总模型含 LoRA/Projector/说话人投影为 1,779,841,026，不含外部冻结编码器/Mimi。Talker 仍为 4×768、8 路码本。

Text/Joint 可以显式训练输入 Projector；混合 T2A/I2T/A2T 测试验证梯度穿过完全冻结 Qwen。Audio 模式必须冻结文本策略及输入 Projector，才能消去文本条件的概率比。Weighted/Gated 的零初始融合参数实际更新，非 Bridge 状态逐项不变。正式 SFT 初始化要求模型/Bridge 与完整 LoRA 配置一致；RL Checkpoint 不能冒充 SFT，必须走 Resume。

## 四、实际 Tiny 与有限数据证据

### 1 隔离 CLI

原配置资产检查失败：`/Volumes/TOSHIBA/项目/minimind-o/model/SenseVoiceSmall/model.pt` 不存在，外置盘未挂载。未删除校验或替换主仓库资产锁。

真实 CLI 在 `/private/tmp/omni-r6-local/portable-fixture` 执行：干净源码归档加本轮文件，三条自创文本分别属于 train/validation/test，WordLevel 测试 Tokenizer、16 维 Tiny Qwen、4 层 16 维 Talker、8 路测试码本。隔离目录中的固定资产元数据明确标记 fixture；惰性媒体权重仅供身份检查且**从未加载为模型**。模型前向、采样、Loss、反向和优化器均为真实实现，没有替换训练算法。

共 **17 条实际命令全部 exit 0**，覆盖 prepare/cache hit、三模式 dry-run/smoke、train、resume、baseline/evaluate、profile。命令逐条记录在 [cli-commands.json](../../experiments/r6-local/20261109-cpu-fp32/cli-commands.json)。原始源码归档、输入、日志、Checkpoint 保留在上述独立临时目录，不进入 Git。

| 模式 | 更新数 | 第 2 步梯度范数 | 第 2 步条件 KL | 冻结/更新 |
| --- | ---: | ---: | ---: | --- |
| Text | 2 | 0.096239537 | 0.00000299321 | 冻结摘要一致；LoRA 改变 |
| Audio | 2 | 1.915324569 | 0.00427393926 | 冻结摘要一致；音频四组改变 |
| Joint | 2 | 1.671904206 | 0.00389483160 | 冻结摘要一致；LoRA 和音频四组改变 |

策略项接近零是当前等权组优势均值接近零的结果；非零梯度和参数摘要变化独立证明真实更新，不能用单个 Loss 的正负解释性能。Tiny Token/Code 均值奖励只提供可复现数学信号。

### 2 真实历史 T2A 缓存

只读 `.cache/r3-local/tiny-v1`，核对缓存 manifest、Tensor 文件 SHA 和 contract_id，选择两条已许可 T2A prompt，长度分别为 34、25。沿用真实官方 Qwen Tokenizer 契约、151,936 词表及 8×2048 声码接口，以 32 维 Tiny 模型、group=3 分别执行三模式一次更新。

| 模式 | Old 与重算最大绝对误差 | 梯度范数 | 更新 |
| --- | ---: | ---: | --- |
| Text | 9.536743e-7 | 0.104520425 | 通过 |
| Audio | 7.629395e-6 | 2.561798811 | 通过 |
| Joint | 1.525879e-5 | 2.468580484 | 通过 |

统一 `rtol=atol=1e-4`，没有放宽断言。所有指定组梯度非零，冻结参数不变。这是缓存/API 数值验收：缺失的原始媒体和编码器权重没有重新验证，也没有实际 Mimi/ASR 音质验收。来源、许可限制、revision、文件 SHA、Reward 与梯度见 [cached-t2a.json](../../experiments/r6-local/20261109-cpu-fp32/cached-t2a.json)，原始 prompt token 未再分发。

## 五、Checkpoint 与评测

保存模型/LoRA、Optimizer、Scheduler、Scaler、RNG、任务/数据游标、Policy 版本、Reference 快照及其摘要、冻结底座摘要、配置/数据/源码身份。周期保存同时满足 `save_interval` 与完整 rollout 更新边界，正常终点始终保存。恢复后重新生成下一批，不携带过期 Old 概率。

实际 CLI 的连续 Joint 两步与新进程一步加恢复一步，**519 个 Tensor 和 1775 个非 Tensor 叶节点精确一致**。单测另覆盖 Adapter 加载、保存间隔、半轮拒绝、Policy 版本篡改、Reference/冻结底座篡改与配置错误。

`grpo-evaluate --baseline` 直接评测初始策略，不制造伪 Baseline Checkpoint；终点用 `--resume`。两者 seed 均为 `training.seed + 1,000,003`，相同 split、样本、生成和 Reward 参数，均无参数更新。原始核对见 [cli-verification.json](../../experiments/r6-local/20261109-cpu-fp32/cli-verification.json)。

## 六、测试命令与结果

Mac arm64，Python 3.12.14、PyTorch 2.15.0、Transformers 4.57.6，CPU FP32。未执行远程 CI 或 CUDA。

```bash
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
export MPLCONFIGDIR=/private/tmp/omni-r6-local/mpl
.venv/bin/python -m pytest -q -p no:cacheprovider --basetemp=/private/tmp/omni-r6-local/pytest-final
.venv/bin/ruff check --no-cache .
.venv/bin/ruff format --check --no-cache omni tests scripts
.venv/bin/python -m pip check
git diff --check
```

| 检查 | 实际结果 |
| --- | --- |
| 当前工作区全部 pytest | **399 passed, 1 skipped in 129.02s** |
| 只含本次修改的干净归档 | **392 passed, 1 skipped in 119.37s** |
| 新增数学 / Rollout / Reward / 训练 / 身份 / 补充审查测试 | 分别 22 / 19 / 10 / 19 / 7 / 3 项通过，共 80 项 |
| Ruff 全仓、源码格式 | 全通过；工作区 88 个、干净树 85 个 Python 文件格式通过 |
| 依赖、Diff | 无损坏依赖，Diff 空白检查通过 |
| 实际 CLI | 17 条通过；旧真实资产 dry-run 另记 1 条依赖缺失失败 |

两个全仓运行均只跳过同一个 MPS 环境测试。干净归档为 `git archive c07f1df` 加本轮生产/测试文件，保留原资产锁和 R2 manifest，排除用户未提交的模型/Streaming 改动；不同于 CLI 合成资产归档。因此本次实现不依赖用户未提交代码。精确命令、检查结果及元数据见 [acceptance.json](../../experiments/r6-local/20261109-cpu-fp32/acceptance.json)。

开发时曾有 Reward 解码 fixture 过早 STOP，未形成完整帧；修正独立固定采样种子后保留原断言，补充 ASR 手算 CER=1/3 正例。直接 `pytest` 曾因既有 `scripts` 导入路径收集失败，最终与 README/CI 一致使用 `python -m pytest`，未改旧测试。

## 七、文件级交付

| 文件 | 本轮修改 |
| --- | --- |
| `omni/training/grpo.py` | 组优势、精确 KL、Clipped Objective 和两种归一化 |
| `omni/training/grpo_rollout.py` | 真实采样、严格重算、动作 Mask、Delay/停止与完整帧提取 |
| `omni/training/grpo_rewards.py` | 确定性诊断、文本匹配、完整性、Mimi/ASR 接口和显式指标注入 |
| `omni/training/r6_local.py` | 共用训练设施、固定 Reference、更新、检查点、评测和预检 |
| `omni/training/r3_local.py` | 可选 GRPO 配置入口与 Reward 字典替换语义，原 SFT 缺省不变 |
| `scripts/run_experiment.py` | 增加五个 GRPO 路由与 Baseline 评测参数 |
| `configs/v2/r6_text_local.yaml`、`r6_audio_local.yaml`、`r6_joint_local.yaml` | 三模式本地配置 |
| `configs/v2/r6_text_h100.yaml`、`r6_audio_h100.yaml`、`r6_joint_h100.yaml` | 单卡正式入口模板，B1/数据/Reward 待获批配置 |
| `tests/test_r6_math.py`、`test_r6_rollout.py`、`test_r6_rewards.py` | 独立数学、条件概率与 Reward 测试 |
| `tests/test_r6_training.py`、`test_r6_integrity.py`、`test_r6_review.py` | 更新、恢复、历史 P1 防回归、多任务和可替换 Bridge 测试 |
| `docs/GRPO_MATH.md`、`docs/cloud/G6_GRPO_RUNBOOK.md` | 算法定义与实际 CLI 操作步骤 |
| 本报告、`docs/reviews/R6-L_FINAL_REVIEW.md` | 同轮最终验收与风险 |
| README、AGENTS、PROJECT_PLAN、项目待办、两份 v2.1 执行文档 | 同步本地 PASS 与云端未执行状态 |
| `experiments/r6-local/20261109-cpu-fp32/` | 小规模实际命令、参数、数值、恢复与保护摘要 |

## 八、保护与下一步

开工记录 207 个既有文件 SHA/缺失状态，686 个已有资产/实验文件的大小和 mtime；本轮只提交白名单。P0–P5 与 R0–R5-local SHA 不变。执行中另出现用户 G3 文档移动，移动前后字节相同，本次未纳入提交或撤销。详见 [保护证据](../../experiments/r6-local/20261109-cpu-fp32/protection.json)。

当前停在 **G3 云 GPU 部署入口**。下一步需用户授权单卡 CUDA Profile，恢复/搬运已核验的媒体资产，按 G3/G4 验收 B1 后再进行 G6。待验内容包括真实 1.7B BF16 Cache 重放误差、长序列显存、Mimi/ASR 吞吐、可信 QA A2A、最终 Reward、正式音质/文本对照及单双卡决策。没有把 Tiny 改善或 fixture 随机语音写成研究成果。
