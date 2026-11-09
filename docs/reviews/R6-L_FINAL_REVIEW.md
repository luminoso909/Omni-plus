## 一、最终结论

**PASS：R6-L 本地算法与工程验收。** 2026-11-09，审查以实际代码、独立数值期望、真实 Tiny 更新、严格恢复和全仓回归为依据。此为同轮自我审查，不称外部独立审查。无已知 P0/P1 阻断；正式 B1/G6、CUDA、音质收益仍未执行。

## 二、数学与接口核对

| 项目 | 审查依据与结果 |
| --- | --- |
| 联合分解 | `grpo_rollout.py` 与实际 Omni forward：当前文本和八路音频头仅依赖过去动作；物理步联合 logprob 求和，不按声学帧或路数平均。独立概率乘积例验证八码本 Ratio=256。 |
| 组优势 | 总体方差、每 prompt 独立标准化；`[0,2,4]` 和零方差手算一致；无效 Reward 拒绝整组。 |
| PPO | 正负优势四个 clip 分支值与解析梯度正确；Old/Advantage detached；float64 gradcheck 通过。 |
| KL | 精确 `KL(current || reference)`，方向、数值和 softmax Jacobian 对照通过；Reference 无梯度，Current 的概率权重保留梯度。 |
| Mask 与时间轴 | EOS/STOP 自身有效，延迟和结束 PAD 无效；文本 EOS 后音频继续，残缺帧仍保留动作概率；Cache、左 padding、截断回归通过。 |
| 行为分布 | 合法完整支持集的温度 Softmax 同时用于采样和三种概率重算；拒绝动态 Top-k/Top-p，未混用原始 softmax。 |
| 归一化 | 每轨迹等权与有效物理步等权分别验证；累积覆盖整个更新窗口，未按微批均值错误平均。 |
| 参数边界 | Joint 的两个 Projector 穿过冻结 Qwen；Audio 冻结文本策略；Weighted/Gated 融合参数从零更新；无重复 Bridge 投影或 Talker 改写。 |
| 恢复 | 同批多 epoch 固定 Old，完整边界保存；Reference/冻结底座验证，恢复重采样；实际新进程 519 Tensor 与 1775 非 Tensor 叶逐项精确一致。 |

关键路径为 `omni/training/{grpo,grpo_rollout,grpo_rewards,r6_local}.py`；公式与限制见 [算法说明](../GRPO_MATH.md)。多模态当前方差、长序列资源和质量不能由 Tiny 测试外推。

## 三、本轮发现并闭环的问题

| 问题 | 根因与最小处理 | 验收 |
| --- | --- | --- |
| RL 来源伪装 SFT | 仅复用原 SFT 谱系会遗漏 `grpo.policy` 的祖先链；初始来源同时拒绝 `progress.algorithm=grpo-v1` 和配置含 `grpo`，RL 只能走 Resume。 | 两种伪装均拒绝；合法 SFT 正向逐 Tensor 加载。 |
| 同形状 LoRA 语义不同 | rank 相同而 alpha 不同仍可加载 Tensor；增加完整 LoRA 配置相等检查。 | alpha 改动明确失败，正常来源通过。 |
| Baseline 评测不公平 | 不应额外训练一次制造基线，或沿用训练后的不同 RNG；新增真实 `--baseline`，两边重新固定独立评估 seed。 | 相同配置、split、样本 ID、seed，无参数更新。 |
| 保存间隔与 dry-run 不完整 | GRPO 初版未沿用间隔，静态检查未选实际 split/核对 Resume；抽取共享预检，完整 rollout 且命中间隔才保存，终点仍保存。 | 间隔/半轮、评估缺文件、smoke/profile 超预算、错误版本和身份正反例通过。 |
| 检查点开关静默无效 | eval 模式下 HF gradient checkpointing 不生效；显式拒绝未支持开关。 | 配置/API 均明确失败，未改变 dropout/策略协议。 |
| 解码测试缺完整帧 | fixture 均匀 STOP 分布过早结束；固定能形成帧的独立种子，保留原断言，并补 ASR `abc→adc` 手算。 | Reward 10 项测试通过，未放宽数值容差。 |

以上在本轮完成修复再运行最终验收，未延后成另一轮开发。R5 的完整媒体身份与实际源码根算法保持原样，R6 加 7 项入口回归，覆盖丢未选媒体别名、改样本 ID、跨任务身份、缺证据、跨 checkout 导入及合法正例。P2 成本汇总代码未修改。

## 四、最终证据

- 当前工作区：**399 passed, 1 skipped in 129.02s**；干净交付树：**392 passed, 1 skipped in 119.37s**。均仅跳过沙箱 MPS 测试。
- 全仓 Ruff、源码格式、依赖及 Diff 检查通过。新增 80 项测试通过，旧 SFT、Bridge、Checkpoint、Streaming 与 MiniMind 回归包含在全仓结果内。
- 隔离、声明合成资产身份的实际 CLI **17 条通过**，Text/Audio/Joint 各两步；模型与优化算法真实执行，未加载伪媒体权重或声称官方 1.7B 实测。
- 两条已许可的真实 T2A 缓存 prompt，官方 Tokenizer 契约、Tiny 32 维、8×2048 码本，三模式分别一次更新通过。Old 重算最大误差为 `9.54e-7 / 7.63e-6 / 1.53e-5`，未放宽 `1e-4` 断言。
- 原资产 CLI 因外置媒体盘未挂载失败，作为未验依赖保留；未绕过生产身份检查。真实 Mimi/ASR 的重新加载与质量验证未完成。
- P0–P5、R0–R5-local、旧 Checkpoint/资产和用户既有改动受到 SHA/状态保护。本次不 Push。

完整命令、配置、参数量与证据索引见 [R6 阶段报告](../stages/R6-local.md) 和 [验收 JSON](../../experiments/r6-local/20261109-cpu-fp32/acceptance.json)。

## 五、Deferred：G3–G6 必须实测的边界

| 内容 | 后续验收条件 |
| --- | --- |
| 正式 1.7B 与 CUDA/BF16 | 原生 4×768 Talker、1–10 步数值/冻结/恢复、峰值显存和吞吐；Cache 重放失败先定位，不静默放宽断言。 |
| Reference 与全词表资源 | 当前 CPU 可变参数快照交换、完整分类 KL 的复制和激活成本，经单卡 H100 Profile 决定优化。 |
| Mimi/ASR 与音频质量 | 恢复资产并验证真实解码、16k/24k转接、CER及独立音质/人评，不把诊断均值或完整性当能力。 |
| RL 数据和 Reward | B1 成立后锁定可信问答 A2A、合法 split、最终指标/权重/采样比例；字节摘要不能替代未登记转码的语义审计。 |
| 生成失败与零梯度 | 记录未完成帧、空音频、全截断和零方差；当前明确停止，不预设正式长训的跳批策略。 |
| 公平对照 | 同 B1、数据、协议与资源核算；Reward 改善不是独立 Benchmark 改善。Qwen 词表额外对齐行的解码行为需纳入失败审计。 |
| 多卡与梯度检查点 | 尚未实现/验证；不提供伪 torchrun 支持，不承诺双卡可靠运行。 |

这些是云端模型与研究验收条件，不是隐瞒的本地数学故障。当前 R6-L PASS，停止在 G3 云 GPU 部署入口，等待用户另行授权。
