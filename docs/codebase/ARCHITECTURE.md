## 一、现有架构

### 1 组织方式

本仓库是按职责拆分的本地 Python 包。`scripts/` 显式装配资产、模型和运行配置，`omni/models/` 按 Thinker、编码器、Talker 拆分。数据准备、训练和波形推理分别在 `omni/data/`、`omni/training/`、`omni/inference/`。目前多模态模型 `OmniPlusOmni` 继承 MiniMind 的 `OmniPlusForCausalLM`，Qwen3-0.6B 通过独立脚本运行官方文本模型。v2.0 的 Qwen3-1.7B Omni 是迁移目标。 [Omni 组合](../../omni/models/model_omni.py) · [Qwen 文本入口](../../scripts/check_qwen3.py)

### 2 数据与调用链

1. `prepare_p4.py` 调用 `prepare_splits()` 从本地 T2A、A2A、I2T Parquet 划分样本，按原始记录与媒体哈希去重。指定 `--cache-features` 时注入本地编码器和 tokenizer，`cache_split()` 保存处理好的样本。 [准备入口](../../scripts/prepare_p4.py) · [划分与缓存](../../omni/data/prepare.py)
2. `OmniDataset` 展开模态占位符，用完整模板和生成前缀识别 assistant 区间。它构造文本 `[T]`、音频 `[C,T]` 和已移位一次的两套标签。`OmniCollator` 右侧填充计算时间轴，并保留模态特征列表的原 batch 索引。 [数据协议](../../omni/data/omni_dataset.py)
3. `OmniPlusOmni.forward()` 在 prefill 用 `Projector` 输出替换文本嵌入中的占位位置。共享 `OmniPlusModel.forward_embeddings()` 推进 Thinker，返回归一化文本特征、指定 Block 的归一化前特征、位置、KV 与辅助损失。 [模态路由](../../omni/models/model_omni.py) · [共享前向](../../omni/models/thinker/model_thinker.py)
4. `lm_head` 输出文本 logits。`TalkerModule` 将所选 Thinker 特征、历史码本和可选 speaker 条件融合，输出 `[B,C,T,A]` 音频 logits。旧 `OmniOutput.past_key_values` 的顺序是 Thinker 在前、Talker 在后。 [联合模型](../../omni/models/model_omni.py) · [Talker](../../omni/models/talker/model_talker.py)
5. 训练入口取缓存并运行 `train_step()`，外部 `sft_loss()` 对已移位标签计算文本 CE、各码本 CE、STOP 加权和可选 MoE aux。恢复会严格核对模型配置、可训练参数、数据哈希和训练设置。 [训练入口](../../scripts/train_sft.py) · [训练步骤与恢复](../../omni/training/sft.py) · [Loss](../../omni/training/losses.py)
6. 推理入口按单话轮消费 `stream_generate()` 的 `(token, frame)`，`stream_waveform()` 对 `[1,C,1]` 完整帧凑块并冲刷尾部，每次流单独持有 Mimi 解码状态。Demo 将 PCM 写入本地 WAV。 [波形流](../../omni/inference/streaming.py) · [Demo](../../scripts/demo_p5.py)

### 3 模块边界

| 模块 | 所有权 | 外部调用方的责任 | 证据 |
| --- | --- | --- | --- |
| `OmniPlusConfig` | MiniMind 文本、音频词表、编码器宽度与 Bridge 默认层 | 模型资产与 tokenizer 配对 | [config.py](../../omni/models/config.py) |
| `SenseVoiceEncoder`、`SiglipEncoder` | 本地加载、冻结、CPU FP32 输出。语音为 `[N,512]`，P2 图像实测 `[64,768]` | 批内索引、占位和投影 | [编码器](../../omni/models/encoders/model_encoders.py) · [P2](../stages/P2.md) |
| `OmniPlusModel`、`OmniPlusForCausalLM` | 自研 Block、缓存与文本 logits。两者分别拥有 `model.*` 主干和 `lm_head.*` 输出头 | 生成与媒体条件由调用方编排 | [Thinker](../../omni/models/thinker/model_thinker.py) |
| `OmniPlusOmni`、`TalkerModule` | 模态注入与分层缓存，`Talker.embed_proj` 投影 Thinker 条件并融合音频与 speaker | 编码器、Mimi Codec 和数据采集在模型外显式创建 | [Omni](../../omni/models/model_omni.py) · [Talker](../../omni/models/talker/model_talker.py) |
| `delay_codes()`、`restore_codes()`、`code_mask()` | 延迟排列、对已知帧数的逆变换与声学值校验 | Dataset 和生成循环各自仍有时轴代码，迁移时须单独验证两侧 | [对齐函数](../../omni/data/audio_alignment.py) |
| `MimiCodec`、`MimiStreamingDecoder` | Mimi `[B,8,F]` 与波形转换，按话轮持有独立 Codec KV | 文本采样、每码本 STOP、离散帧对齐由上游管理 | [Codec](../../omni/inference/codec.py) · [流式解码](../../omni/inference/streaming.py) |

### 4 重用方式与生命周期

| 方式 | 当前证据 | 用途 |
| --- | --- | --- |
| 显式传入依赖 | `OmniDataset(..., audio_encoder=..., vision_encoder=...)`、`MimiCodec(model)` | CLI 决定本地资产，普通测试可注入小模型或替身 |
| 共享实现和稳定权重名 | `OmniPlusModel.forward_embeddings()`、Talker 复用 `OmniPlusTransformerBlock` | 保留历史 MiniMind state dict 结构与前向数值参照 |
| 同步迭代器 | `stream_generate()`、`stream_waveform()` | 消费方控制推进速度，关闭时级联释放上游状态。已审计 Demo 的调用链没有后台队列 |
| 每次运行显式分配路径 | `train_sft.run()`、`check_p5.main()`、`demo_p5.main()` | 训练恢复读取旧 checkpoint，新结果使用独立运行目录 |

### 5 R0 接缝与迁移风险

- R1 建议使用官方 CausalLM 的独立适配层。Transformers 4.57.6 的 `Qwen3Model` 在首次启用缓存时分配 `DynamicCache`。`cache_position[T]` 表示含 padding 的物理缓存位置，默认 `position_ids` 使用该位置。`generate()` 会从 mask 累加得到逻辑 `position_ids[B,T]`，padding 位置填 1。直接前向与生成对照时须显式对齐位置语义。 [Qwen3 v4.57.6](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen3/modeling_qwen3.py#L335-L425) · [GenerationMixin v4.57.6](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/generation/utils.py#L593-L624)
- HF `hidden_states[0]` 是送入首个 Block 的嵌入，中间 `hidden_states[i+1]` 对应零基 Block `i` 的输出。末项由 `check_model_inputs` 换成最终归一化结果。若需要最后一层归一化前输出，R1 需明确捕获方式并验证。 [Qwen3 输出记录](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen3/modeling_qwen3.py#L280-L298) · [输出收集 v4.57.6](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/utils/generic.py#L1017-L1103)
- R2 需给 2048→768 条件变换定义单一所有者，保留旧 `talker.embed_proj` 路径用于历史复现。新路由分开持有 Qwen Cache、Talker KV 元组和 Mimi 会话缓存。R3 用同版 tokenizer 与 Mimi 原码本重建数据，R4 才检验学习质量和全尺寸恢复。 [现有投影](../../omni/models/talker/model_talker.py) · [风险清单](CONCERNS.md)

### 6 证据索引

- [历史结构、实测与证据边界](../stages/P5.md)
- [权重及 tokenizer 位置](../../configs/assets.yaml)
- [输入编码、共享 Thinker、Talker](../../omni/models)
- [数据、训练与推理入口](../../scripts)
- [依赖版本锁](../../uv.lock)
