## 一、设计范围

本文记录 R1–R3 的接口草案。旧 MiniMind 入口、权重键和实验目录继续承担 P0–P5 复现。v2.0 使用独立 Qwen 路径和独立 run 命名空间。正式骨干为官方 Qwen3-1.7B 非 Base，已有 0.6B 资产用于接口预验收。

## 二、R1 的 Thinker 契约

建议在 `omni/models/thinker/backend.py` 定义最小协议与返回对象，在 `qwen_backend.py` 包装已经显式加载的原生 HF 模型。草案签名如下。

```text
embed_tokens(input_ids: Long[B,T]) -> Float[B,T,D]

forward(input_ids: Long[B,T] | None,
        inputs_embeds: Float[B,T,D] | None,
        *, attention_mask: Bool[B,P+T] | Long[B,P+T] | None,
        position_ids: Long[B,T] | Long[1,T] | None,
        cache_position: Long[T] | None,
        past_key_values: BackendCache | None,
        use_cache: bool,
        hidden_layer: int | None = None,
        logits_to_keep: int = 0) -> BackboneOutput

BackboneOutput(logits: Float[B,K,V],
               selected_hidden_state: Float[B,T,D] | None,
               past_key_values: BackendCache | None)
```

- `B` 为 batch，`T` 为本次输入长度，`P` 为 cache 中已有的物理长度，`D/V` 由模型配置给出。输入 ID 和 embedding 二选一。`logits_to_keep=0` 时 `K=T`，正数保留尾部位置。Logits 截取与 Bridge 所需时间轴独立。
- 模型、输入、mask、索引和 cache 位于同一计算设备。ID 与位置为 `torch.long`。嵌入和输出沿用所加载模型的浮点精度。冻结参数与 `eval()` 分别控制梯度和模块状态。R2/R4 训练 Projector 时，仍需让梯度穿过冻结的 Thinker 回到输入 embedding。
- Qwen 使用原生 HF `Cache`。`cache_position` 是物理 cache 位置，`position_ids` 是参与 RoPE 的逻辑位置。左 padding 的 mask 覆盖历史与当前步。R1 对照原生裸 `forward` 和 `generate` 各自的默认位置规则，并测试显式索引、prefill、decode 和多轮新 cache。Cache 可原地更新，复验分支分别创建缓存，关闭 cache 时从空状态开始。
- `hidden_layer` 约定为从零起算的 decoder block，读取 block 后、最终 norm 前的表示。原生 HF 的 `hidden_states[0]` 是输入嵌入，非末层取 `hidden_states[i+1]`。末层输出可能已替换为最终归一化表示，R1 须单测显式取点，或在首版约束选取非末层。默认 Bridge 层与 norm 约定到 R2 确定。
- 官方 Qwen tokenizer、模板和 EOS/PAD 策略随 revision 记录。R1 覆盖中英文、左 padding、模板思考模式和多轮。R2 再审计占位符。如果需要新增行，先明确初始化、冻结、输出头绑定和恢复策略。文本 EOS、思考边界与 Mimi STOP 分属不同协议。

## 三、R2 的组合边界

| 接口草案 | 形状与归属 |
| --- | --- |
| `Projector.forward(features)` | 复用现有 `Projector`。音频 `[N,512]`、视觉 `[64,768]` 投影到 Qwen 隐藏维，按样本维护有效长度、占位和 mask |
| `FixedBridge.forward(selected_hidden_state, attention_mask)` | `[B,T,D_qwen] → [B,T,768]`，持有 v2 唯一一次跨宽度投影。规范化位置和掩码语义在 R2 确定 |
| `Talker.forward(condition, audio_ids, position_ids, attention_mask, past_key_values, use_cache, spk_emb)` | 条件 `[B,T,768]`，音频历史 `[B,8,T]`，返回 `[B,8,T,2112]`、自身 cache 与 aux。首版 4 层、宽 768、随机初始化 |
| `OmniCache(thinker, talker)` | v2 wrapper 分别持有两套状态。Mimi streaming decoder 在外层按话轮维护自身 KV 和卷积历史 |

现有 `TalkerModule.embed_proj` 已承担 `hidden_size → hidden_size → talker_hidden_size`。v2 入口应直接消费 Bridge 的 768 维条件，避免第二次跨宽度投影。R2 先验证参数所有权与梯度，再定最小的独立 Talker 接线方式。旧 `OmniPlusOmni` 的拼接 tuple 与 `talker.embed_proj` 权重键用于原有历史路径。

R2 以真实小样本核对联合时轴。按 Qwen 模板、原始文本与版本匹配的 Mimi codes 重建标签，校对 codebook delay、占位、EOS/STOP 和 padding。当前 Dataset 的标签已经 shift 一次，v2 保留由外部 CE 消费对齐标签的契约。`code_mask()` 仅标记声学码，训练监督仍须纳入 STOP。较长音频继续使用有效联合时轴，batch 补齐位置才进入 padding mask。

## 四、资产清单和配置草案

R1 建议新增 `configs/assets_v2.yaml` 作为声明式索引，现有 `configs/assets.yaml` 继续服务历史路径。`configs/model/qwen3_1_7b.yaml` 引用资产 ID 并指定验收精度和 backend 选项。文件名为后续迁移设计。

```yaml
schema_version: 1
assets:
  qwen3_1_7b:
    repo_id: Qwen/Qwen3-1.7B
    local_path: weights/qwen3-1.7b
    revision: null
    license: null
    loader: transformers.Qwen3ForCausalLM
    tokenizer_asset: qwen3_1_7b
```

- `revision`、`license` 和文件指纹在 R1 核实。加载时由调用方解析本地路径，manifest 记录模型和 tokenizer 的 revision、配置、模板及文件 SHA-256，保存于 `experiments/r1/<run_id>/manifest.json`。路径或指纹冲突时停止验收，复用 0.6B 只进入单独轻量记录。
- R2/R3 补充 encoder/processor、codec、来源与转换信息、采样率和码本参数。旧 SigLIP 裁剪目录、SenseVoice 以及 Mimi FP16 转换资产依据现有路径和实测记录复用。换资产时单独检查输出、兼容性和许可。
- R3 给数据 split、媒体、tokenizer、模板、encoder、codec、dtype 和预处理版本建指纹，按任务分隔缓存。新版 checkpoint 持有同一组指纹、参数冻结清单、优化器、RNG 和阶段进度。首轮冻结 Qwen，R4 正式训练前基于真实单卡 profile 选预算，LoRA 由后续实验决定。

## 五、证据入口

- MiniMind 配置、嵌入前向与组合 Cache。 [模型配置](../../omni/models/config.py) · [Thinker](../../omni/models/thinker/model_thinker.py) · [Omni](../../omni/models/model_omni.py)
- 冻结编码器、Bridge 旧投影、联合标签与损失。 [编码器](../../omni/models/encoders/model_encoders.py) · [Talker](../../omni/models/talker/model_talker.py) · [Dataset](../../omni/data/omni_dataset.py) · [损失](../../omni/training/losses.py)
- 本地 HF 4.57.6 的 `models/qwen3/modeling_qwen3.py`、`cache_utils.py` 与 `generation/utils.py`。运行时核验使用 `.venv` 中已安装版本。
- 原规划与文件级实施归属。 [v2.0 规划](../PROJECT_PLAN.md) · [迁移表](R0-migration-map.md)
