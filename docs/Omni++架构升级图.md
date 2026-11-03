# Omni++ 架构升级备忘录

> **用途**：MiniMind-O 源码学习完成后，作为 Omni++ 的架构演进与消融实验清单。  
> **原则**：保留自研 Transformer 核心；一次只改变少数变量；先得到可复现 Baseline，再根据评测结果决定是否保留改动。  
> **依据**：当前阅读的 `model_minimind.py`、`model_omni.py`，以及已讨论的升级方向。这里是**研发候选方案**，不预设性能一定提升。

## 1. 当前架构与主要局限

```text
Text ──────────────────────────────┐
Image → SigLIP2 → Vision Projector ─┤→ Thinker (Decoder LM) → Text Head
Audio → SenseVoice → Audio Projector┘         │
                                             └→ 中间层 Bridge ─┐
历史 Mimi Audio Codes → TalkerEmbedding ────────────────────────┤
                                                               ↓
                                                      融合 → Talker
                                                               ↓
                                               8 Audio Heads → Mimi Decoder → WAV
```

**当前优势**：Thinker–Talker 模块清楚；已有图像与语音输入、离散音频生成、GQA、可选 MoE 和 KV Cache，适合研究与重构。  
**主要限制**：图像采用固定粒度的 patch 表示；Projector 主要是简单 MLP；Bridge 固定取单层；Talker 用 8 路编码 + 固定 Delay Pattern；长序列、复杂音视频时序与训练/推理效率仍有改进空间。

## 2. 可提升项总览

| 优先级 | 方向 | 当前方案 → 候选改动 | 预计价值 | 成本 |
|---|---|---|---|---|
| **P0** | 训练与评测基线 | 建立 Text/Image/Audio 分项测试、吞吐与消融记录 | 验证后续升级是否有效 | 低～中 |
| **P1** | 视觉分辨率 | 固定图像 tokens → 多裁剪 / 动态分辨率 | OCR、小目标、细节理解 | 中 |
| **P1** | 视觉 Token 压缩 | 直接使用 patch tokens → Pooling / Resampler | 降低视觉上下文成本 | 中 |
| **P1** | 多模态融合 | MLP Projector → Gated Fusion / 可选 Cross-Attention | 更灵活的模态交互 | 中～高 |
| **P1** | Bridge | 固定单层输出 → 多层加权 / Gated Bridge | 改善语音条件信息的来源 | 低～中 |
| **P2** | Talker 对齐 | 固定 8 路 Delay → Chunk-wise / 可学习对齐策略 | 音文同步、流式语音 | 高 |
| **P2** | 模型核心与系统 | Dense/MoE + GQA → QK Norm、路由消融、SDPA/FA、AMP/DDP | 稳定性、训练与推理效率 | 中～高 |
| **P2** | 后训练 | 基础 SFT → 蒸馏 / 偏好优化 / 适用任务上的 GRPO | 任务完成度、回答质量 | 高 |

## 3. 重点改进：如何做、如何验证

### 3.1 视觉：动态分辨率 + Token 压缩（P1）
- **现状**：默认 SigLIP2 的 256×256、patch32 对应约 **64 个视觉 tokens/图像**；源码 `MMVisionProjector` 的 `target_tokens` 参数并未真正执行 token 数量压缩。
- **改法**：先用**多裁剪/局部高分辨率 patch**增强细节，再用 Mean Pooling 或轻量 Resampler 控制送入 Thinker 的视觉 tokens；保留原 64-token 路径作基线。
- **评测**：OCR、小物体问答、图像描述准确率；视觉 token 数、推理延迟、峰值显存。
- **风险**：视觉 token 增长会增加 Attention 计算；单纯压缩可能损失细节。

### 3.2 多模态融合：Gated Projector（P1）
- **现状**：视觉/音频特征通过 Projector 转成 LLM 隐藏维度，再注入占位符位置。
- **改法**：在原 MLP 投影外增加**可学习门控**；进阶实验再尝试 Cross-Attention。先不要同时引入复杂的多融合模块。
- **评测**：图文/语音问答；模态缺失、噪声输入鲁棒性；训练是否稳定。
- **风险**：仅修改结构而缺乏对齐数据与训练，通常不能自动提高能力。

### 3.3 Bridge：固定层 → 多层融合（P1，推荐重点）
- **现状**：默认 Thinker 有 8 层，`bridge_layer=3` 代表**第 4 层**的输出；Talker 读取该中间隐藏状态，而 Thinker 第 5–8 层继续承担文本预测。
- **候选**：
  1. **Baseline**：固定第 4 层 `H4`。
  2. **Last-layer**：改用 `H8`。
  3. **Weighted Bridge**：`H_b = Σ softmax(a_l) · H_l`（选少数层即可）。
  4. **Gated Bridge**：`H_b = g⊙H4 + (1-g)⊙H8`，`g=sigmoid(W[H4;H8])`。
- **评测**：文本任务质量是否退化；语音内容一致性、可懂度与延迟；不同 Bridge 的收敛速度。
- **风险**：跨层特征分布不同，门控不一定比固定层好；需要联合训练和公平消融。

### 3.4 Talker：音文对齐与流式生成（P2）
- **现状**：历史 8 路 Mimi Codes 经 `TalkerEmbedding` 融合为 `[B,T,D]`，Talker 输出 8 组音频 logits；固定 Delay Pattern 错位预测和组帧。
- **改法**：先验证当前 Delay 的输出正确性和流式边界；后续尝试**按语音 chunk 组织的生成**及更灵活的文本-音频步数对齐。
- **评测**：语音可懂度（ASR 反转写/WER）、音质主观评分、首包延迟、连续播放与中断稳定性。
- **风险**：文本 token 与音频帧不天然一一对应；修改调度通常必须同时修改数据预处理、训练标签与解码器接口。

### 3.5 模型核心、训练与后训练（P2）
- **核心**：继续沿用 CS336 自研的 GQA、RoPE、SwiGLU、KV Cache 与 Dense/MoE。将 QK Norm、YaRN、MoE Router/负载均衡作为可开关实验；用 AMP、SDPA/FlashAttention、DDP 等做系统基准。
- **训练**：先完成图文/语音对齐与 SFT；必要时做蒸馏。只对**有可靠可验证奖励**的任务尝试 GRPO，不把它强行套到语音自然度上。
- **评测**：Loss/任务准确率/专家负载、吞吐 tokens/s、峰值显存、训练稳定性；记录相同数据和预算下的对照结果。

## 4. 推荐实施顺序

1. **阶段 A｜跑通 Baseline**：加载权重、文本/图像/语音推理；固定测试集、记录结果与效率。
2. **阶段 B｜做一个主创新点**：优先 **Multi-Layer / Gated Bridge**，一次只改一处，完成消融。
3. **阶段 C｜提升视觉**：多裁剪或动态分辨率 + Token 压缩，比较质量/延迟权衡。
4. **阶段 D｜延伸到 Talker 与后训练**：先验证 Delay Pattern，再研究音文对齐；最后考虑 SFT/蒸馏/任务型 RL。

## 5. 设计底线（避免“堆模块”）

- **保留自研核心**：不为了兼容 Hugging Face 而把整个 Transformer 换成库封装；HF 作为外层接口。
- **权重兼容不是 API 兼容**：改变 Bridge、Tokenizer、Vision Token 数或 Talker 结构后，需逐项检查参数形状、位置编码、数据对齐及权重加载。
- **每项改造必须配消融**：Baseline → 单项改动 → 组合改动；报告效果、额外成本与失败案例。
- **优先做出一个有实证结果的改进**，胜过同时堆叠多个未验证的“前沿模块”。

---

**建议最终简历亮点**：`从零实现 Omni Transformer 核心 + 提出并验证可学习 Bridge / 视觉 Token 效率方案 + 多模态训练与系统消融`。
