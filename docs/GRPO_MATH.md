## 一、R6-L 算法边界

本说明先固定 R6-L 的概率模型与优化目标，再据此实现和测试。本轮使用真实 GRPO 计算、Tiny Qwen Omni 与极小样本，不据此推断正式 RL 模型能力。本文及配套复核属于同轮自我审查，不能称为外部独立验收。

组内相对优势与 PPO clipped surrogate 采用 GRPO 的基本思路；本文自行定义多模态联合动作、严格采样分布、精确条件 KL 和恢复边界。原始方法参考 [DeepSeekMath §4.1](https://arxiv.org/html/2402.03300v3#S4.SS1) 与 [PPO](https://arxiv.org/abs/1707.06347)。本实现不使用价值网络，也不声称逐项复现论文的 KL 估计器或训练配方。

### 1 策略与版本

| 名称 | 定义 | 允许更新的时机 |
| --- | --- | --- |
| Current Policy $\pi_\theta$ | 当前模型及固定采样变换定义的分布 | 一批完整 Rollout 收集后做优化器更新 |
| Rollout / Old Policy $\pi_{\mathrm{old}}$ | 收集该批轨迹时 Current 的版本 | 整批采样完成前固定；后续多个更新 epoch 保留原 logprob |
| Reference Policy $\pi_{\mathrm{ref}}$ | 本次 RL 启动时的初始 SFT 策略；Tiny 可使用固定随机初态 | 整次训练固定，无梯度 |

Old 不需要常驻第二份模型。采样时保存动作、Mask、Old Logprob、Policy 版本及配置，更新期间禁止重新计算 Current 后冒充 Old。更新只复用当前批采样轨迹；用完即丢弃，不接受未声明的历史离线 Rollout。多 epoch 是同一行为策略上的 PPO surrogate 更新，不是每个 epoch 都重新变成 on-policy。

所有模型前向使用 `eval()` 关闭 Dropout，Current 的重算仍保留 autograd。Reference 无梯度；固定 Qwen 主权重不等于 `no_grad()` 包住 Current，Projector 和 LoRA 的输入梯度必须保留。

## 二、真实时间轴与联合概率

### 1 状态和 Tensor Shapes

记 $P$ 为完整 prompt 长度，$t=0,\ldots,T-1$ 为生成决策步，$C=8$。状态 $s_t$ 包含 prompt、输入媒体特征和全部过去动作。第 $t$ 步的模型输入止于 $P+t-1$，用于预测新动作 $x_t$、$a_{c,t}$。

| Tensor | Shape | 含义 |
| --- | --- | --- |
| 文本动作与文本 Mask | `[N,T]` | 整数 Token，Bool 有效采样动作 |
| 音频动作与音频 Mask | `[N,8,T]` | 整数 Code/STOP，Bool 有效采样动作 |
| 文本 Logits | `[N,T,V]` | 预测动作的文本分类分数 |
| 音频 Logits | `[N,8,T,A]` | 8 路预测分数，$A$ 包含特殊 ID |
| 联合 Logprob、KL、步 Mask | `[N,T]` | 每个物理决策步的联合量 |
| Reward | `[B,G]` | $B$ 个 prompt，各有 $G\ge2$ 个独立采样 |
| Advantage | `[B,G]` 或展平 `[N]` | $N=B G$，不跨 prompt 标准化 |

Qwen 只接收已经存在的文本与输入媒体。Talker 接收同位置的 Qwen 隐藏态及过去延迟声码，8 个输出头同时计算。当前步新采样的文本 Token 和其它码本不回灌到本步 Logits。因此实际分解为
$$\pi_\theta(x_t,a_t\mid s_t)=\pi^x_\theta(x_t\mid s_t)^{m^x_t}\prod_{c=0}^{7}\pi^a_{\theta,c}(a_{c,t}\mid s_t)^{m^a_{c,t}}.$$
未激活分支是由状态决定的 PAD 点质量，概率为 1。联合对数概率为
$$\ell_{\theta,t}=m^x_t\log\pi^x_\theta(x_t\mid s_t)+\sum_{c=0}^{7}m^a_{c,t}\log\pi^a_{\theta,c}(a_{c,t}\mid s_t).$$
这里按 **物理时间步** 联合，不将同一声学帧的对角线码本误当成同条件下同时生成的八个变量。整个轨迹的概率再对所有生成步相乘，其 logprob 为 $\sum_t\ell_{\theta,t}$。

### 2 Delay、EOS、STOP 与截断

- 文本从 $t=0$ 开始，首次 EOS 是有效动作；EOS 后文本 PAD 是强制值，文本 Mask 为 False。
- 非 thinking 语音分支的码本 $c$ 从 $t=1+c$ 开始。第一枚 STOP 是该路有效动作，后续 PAD 均无动作概率。
- 音频 8 路独立停止；文本 EOS 后仍继续音频。仅当文本已 EOS 且所有已安排的音频分支均停止，或达到预算时结束。
- 预算截断不伪造 EOS/STOP。已采样但尚不能组成完整声学帧的动作仍计入 Policy Loss。解码只输出八路都为真实声学码的完整帧。
- prompt padding 不计动作；音频尾部使用的文本 PAD 属于有效时间位置，attention Mask 仍为 1，否则会改变 Talker 条件。
- `text` 模式固定音频为 PAD。`audio` 模式继续采样文本作为条件，但必须冻结所有改变文本分布的参数，包括 Qwen LoRA 和输入 Projector；Current/Old 的文本因子因此抵消，只优化音频。`joint` 模式同时计入两种动作。
- 无有效动作的轨迹不能构成训练样本，应明确拒绝，不能除以零或悄悄替换为零 Reward。

八路 STOP 不齐可能导致空音频或缺帧。这是可记录的采样结果，可以通过完整性 Reward 处理，不是删除对应概率项的理由。

### 3 可手算的概率例子

某一步有一个文本动作和八个有效音频动作。Old 对这些已选动作给出文本概率 $1/2$、每路音频概率 $1/2$；Current 给出文本概率 $3/4$，音频各路仍为 $1/2$。则
$$p_{\mathrm{old}}=\frac1{512},\qquad p_\theta=\frac3{1024},\qquad r=\frac{p_\theta}{p_{\mathrm{old}}}=\frac32.$$
若错误地把九个 logprob 取均值再指数，得到的是 $r=(3/2)^{1/9}$，不再是该联合动作的概率比。若只音频且八路的概率比均为 2，则正确联合 Ratio 为 $2^8=256$，按路平均会错误地得到 2。

一个两码本教学时间轴可写为下表，实际实现为八路同样的延迟规则。

| 生成步 | 文本 | Codebook 0 | Codebook 1 | 有效动作数 |
| --- | --- | --- | --- | ---: |
| 0 | Token | PAD | PAD | 1 |
| 1 | EOS | Code | PAD | 2 |
| 2 | PAD | STOP | Code | 2 |
| 3 | PAD | PAD | STOP | 1 |

EOS 与 STOP 的概率都参与目标；强制 PAD 不参与。声学帧由 `(c0,t1)` 与 `(c1,t2)` 组合，但两个动作的条件状态不同。

## 三、采样分布与 Reward

### 1 同一分布贯穿采样与重算

对允许动作集合 $S_t$ 与固定温度 $\tau>0$，使用
$$\pi_\theta(v\mid s_t)=\frac{\exp(z_{\theta,v}/\tau)}{\sum_{u\in S_t}\exp(z_{\theta,u}/\tau)},\qquad v\in S_t.$$
文本集合排除输入模态占位符和 PAD，保留 EOS；音频集合只含有效 Mimi 码及 STOP。Old、Current、Reference 在同一历史上使用完全相同的集合和温度。文本和音频可以配置不同正温度。该目标优化的是明确的温度变换后 Policy，不能把它描述成未经变换的原始 softmax 策略。

R6-L 仅支持 `top_k=0`、`top_p=1`，其它值直接报错。动态截断会导致 Current/Old/Reference 支持集不同，本轮不通过忽略截断归一化或悄悄重用原始 logits 来近似处理。贪心温度 0 不用于 Policy Gradient。

### 2 组内相对优势

每个 prompt 独立采样 $G$ 条轨迹，以有限标量总 Reward $R_i$ 计算
$$\bar R=\frac1G\sum_{i=1}^{G}R_i,\qquad \sigma=\sqrt{\frac1G\sum_{i=1}^{G}(R_i-\bar R)^2},\qquad A_i=\frac{R_i-\bar R}{\sigma+\varepsilon}.$$
使用总体标准差，不使用 $G-1$ 的无偏估计。Reward 和 Advantage 均停止梯度。零方差组的 Advantage 为零。非有限分项或总 Reward 明确拒绝整组，不删除个别样本后改变组统计，也不把 NaN 当零。

手算 $R=[0,2,4]$ 得 $\bar R=2$、$\sigma=\sqrt{8/3}$，所以 $A=[-2,0,2]/(\sqrt{8/3}+\varepsilon)$。$R=[7,7,7]$ 得 $A=[0,0,0]$。零方差不代表 KL 必须为零；若 Policy 已偏离 Reference，KL 仍可以提供梯度。

Reward 接口按分项返回数值与错误信息。R6-L 的确定性 Token、声码或完整性奖励用于数学和更新验证；它们不代表正式文本正确性、ASR、语义或音质。G6 的 Reward 公式、任务配比与超参数待 B1 成立后确定。A2A echo 记录仅作诊断，真实问答奖励必须另有可信回答文本与目标语音来源。

## 四、Clipped Objective 与 KL

### 1 每步联合 Ratio

冻结采样时的 $\ell_{\mathrm{old},i,t}$，对同一动作轨迹重算 Current，得到
$$r_{i,t}=\exp(\ell_{\theta,i,t}-\operatorname{stopgrad}(\ell_{\mathrm{old},i,t})).$$
Outcome Advantage 在一条轨迹内广播，PPO 项为
$$s_{i,t}=\min\left(r_{i,t}A_i,\operatorname{clip}(r_{i,t},1-\epsilon,1+\epsilon)A_i\right).$$
这是 **每步联合动作 Ratio** 的 surrogate，不是完整轨迹重要性权重 $\prod_t r_{i,t}$。不能据此声称消除了多 epoch 的状态分布偏差。输入或指数结果非有限时应停止并报告，不靠额外裁剪 log-ratio 隐藏数值失败。

例如 $\epsilon=0.2$。正优势 $A=1$、$r=1.5$ 时 $s=1.2$，处在 clipped 平台且对 Current logprob 的梯度为零。负优势 $A=-1$、$r=0.5$ 时 $s=-0.8$，同样被裁剪；负优势且 $r=1.5$ 时 $s=-1.5$，仍有降低该概率的梯度。

### 2 精确分类 KL

每个有效分量采用 Current 到固定 Reference 的精确 KL
$$D(p\Vert q)=\sum_{v\in S_t}p_v(\log p_v-\operatorname{stopgrad}(\log q_v)).$$
共同状态下的联合分布按头独立，因此
$$K_{i,t}=m^x_{i,t}D(\pi^x_\theta\Vert\pi^x_{\mathrm{ref}})+\sum_c m^a_{i,c,t}D(\pi^a_{\theta,c}\Vert\pi^a_{\mathrm{ref},c}).$$
对 Current 的概率权重和 logprob 两部分都保留梯度。Reference 不参与优化，不将其计算结果写回 Old。计算仅在合法 support 上进行，避免 `0 * (-inf - -inf)`。

手算 $p=[0.75,0.25]$、$q=[0.5,0.5]$ 得
$$D(p\Vert q)=0.75\log1.5+0.25\log0.5\approx0.130812036.$$
令 $p=\operatorname{softmax}(z)$，独立导数为
$$\frac{\partial D}{\partial z_j}=p_j\left(\log\frac{p_j}{q_j}-D\right).$$
反向 KL $D(q\Vert p)$ 的值和梯度均不同。本实现沿用旧 Rollout 访问状态计算条件 KL，属于明确的正则目标，不是对新策略完整轨迹分布 KL 的无偏估计，也不采用混入旧采样分布后未经修正的单动作 KL 估计。

### 3 归一化与梯度累积

令 $M_{i,t}$ 表示所选模式至少存在一个有效分量，$L_i=\sum_tM_{i,t}>0$，$u_{i,t}=-s_{i,t}+\beta K_{i,t}$。

| 模式 | Loss | 长度权重 |
| --- | --- | --- |
| `sequence_mean` | $\frac1N\sum_i\frac1{L_i}\sum_tM_{i,t}u_{i,t}$ | 每条轨迹等权 |
| `token_mean` | $\frac{\sum_{i,t}M_{i,t}u_{i,t}}{\sum_iL_i}$ | 每个有效物理步等权 |

`token_mean` 中 token 指联合决策步，不指八路声码分量数。**概率相加** 与 **Loss 归一化** 是两件不同的事。先计算正确联合 logprob，再按所选规则归一化，不能提前按有效码本数平均。

若两条轨迹的逐步 Loss 分别为 `[1]` 与 `[3,3,3]`，两种结果分别为 2 与 2.5。累积多个微批时，分母须覆盖整个优化器窗口，不能把长度不同微批的均值再次等权平均冒充 `token_mean`。

## 五、Reference、保存与复核要求

Reference 必须固定来源与内容。若实现通过替换可变参数快照复用同一模型，应先保证其冻结部分与初始 Policy 完全相同，并在 Reference 前向后完整恢复 Current，再建立 Current 的 autograd 图。不能在未完成反向的图中原地切换参数；不能仅关闭 LoRA 冒充同时含已训练 Bridge/Talker 的 B1 Reference。

当前配置只接受 `reference.source: policy_initial`。SFT 初始化除参数键、Shape 和 dtype 外，还核对完整 LoRA 配置，避免同 rank、不同 alpha 改变实际初始策略。恢复时同时核对 Reference 快照与冻结底座摘要，防止共享底座使 Reference 随 Resume 漂移。

Checkpoint 复用已有模型、Optimizer、Scaler、Scheduler、RNG 保存机制，另记录 RL 数据游标、配置/资产/源码身份、Reference 身份和已完成 Policy 版本。仅在完整优化器边界保存。恢复后从该边界重新采样下一批，不携带半轮 Rollout 或过期 Old Logprobs。原采样 Token、Mask、Old 版本、Reward 分项及失败记录作为独立 run 日志归档。

若整批 Policy 与 KL 梯度均为零，当前入口不执行 AdamW 更新、不推进策略版本，结果标为 `no_update` 并停止。非完整 Rollout 的诊断快照不能恢复；如果此前已在同一步号保存成功边界，则保留该完整快照，恢复后重新采样未完成批次。`no_update` 不能称为 Tiny 更新通过；正式长训是否跳过零梯度批及其游标策略留待 G6 明确。

数学复核必须用独立期望值，包括上述优势、概率乘积、PPO 分段梯度、KL 解析梯度和长度归一化。关键计算用 float64 中央差分与 `gradcheck` 验证，模型流水线再用真实 Tiny HF Qwen 类检验采样与 teacher-forcing 重算、Cache、冻结及实际参数更新。最终测试结果由 `docs/stages/R6-local.md` 和 `docs/reviews/R6-L_FINAL_REVIEW.md` 据实际执行填写。
