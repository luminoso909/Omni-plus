## 一、映射口径

本表依据 R0 审计时工作区的源码、配置、脚本和测试。现状包含已有 P5 文件，Git HEAD `f56e3ae` 本身只到 P4。表中测试列标注现有覆盖和后续验收归属，实测结果见阶段报告。R1–R7 的文件名和处理方式是迁移落点，均需在对应阶段获授权后实施。历史证据见 [P0](../stages/P0.md)、[P1](../stages/P1.md)、[P2](../stages/P2.md)、[P3](../stages/P3.md)、[P4](../stages/P4.md)、[P5](../stages/P5.md)。

## 二、现有源码与配置

### 1 模型

| 现有文件 | 定义、调用方与现状 | 迁移动作与验收归属 |
| --- | --- | --- |
| [omni/models/config.py](../../omni/models/config.py) | `OmniPlusConfig` 同时承载 MiniMind Thinker、Talker、媒体标记与 Bridge 层号。`OmniPlusForCausalLM`、`OmniPlusOmni`、Dataset、Collator 和旧 MiniMind 验收脚本共用。 | **保留** `omni-plus` 类型和默认值。R1 增 Qwen 独立配置。R2 的融合参数与旧配置分开验收。现有 `test_thinker.py`、`test_omni.py`。 |
| [omni/models/thinker/model_thinker.py](../../omni/models/thinker/model_thinker.py) | `OmniPlusModel` 为 MiniMind 主干，`OmniPlusForCausalLM` 提供词表投影、HF 权重接口和文本生成。`forward_embeddings()` 供现有 Omni 取得层输出与位置。Talker 直接复用 `OmniPlusTransformerBlock`。 | **保留** Dense/MoE 键名、RoPE、数值、Cache 和 HF 回环。R1 在同目录新增 Backend 文件。现有 `test_thinker.py`、`test_omni.py`，真实验收由 `check_minimind.py` 持有。 |
| [omni/models/model_omni.py](../../omni/models/model_omni.py) | `OmniPlusOmni` 继承 `OmniPlusForCausalLM`，prefill 按占位符注入特征。`forward()` 按 Thinker 层数拆分旧 Cache，返回时拼回 tuple。`generate()` 与 `stream_generate()` 共用文本/码本状态机。P2、P3、P4、P5 的模型调用方依赖此入口。 | **保留** 旧入口与 195 键 SFT 严格加载。R2 候选并行实现 `model_omni_v2.py`，组合 Qwen、Bridge 和 Talker 并各自管理 Cache，保留原公开路径。现有 `test_omni.py`、`test_omni_generation.py`、`test_codec.py`。 |
| [omni/models/encoders/model_encoders.py](../../omni/models/encoders/model_encoders.py) | `Projector` 为 LayerNorm 与 MLP，按最后一维映射。`SenseVoiceEncoder` 和 `SiglipEncoder` 分别返回有效 `[L,512]` 和本地验收的 `[64,768]`，由 P2 与 P4 特征缓存脚本调用。 | **保留** 冻结生命周期、单样本序列顺序和实际输出。R2 复用 Projector，实例输出由 768 改为 Qwen 的实测宽度，独立校验混合 batch、占位符位置和 mask。现有 `test_encoders.py`、`test_omni.py`、`test_dataset.py`。 |
| [omni/models/talker/model_talker.py](../../omni/models/talker/model_talker.py) | `TalkerModule` 由旧 Omni 调用。`embed_proj` 目前做 `Hthinker → Hthinker → Htalker` 并在末尾做 RMSNorm，`codec_proj` 处理历史音频。`TalkerEmbedding`、`TalkerHead` 共享 base 与分路 adapter，输出 `[B,8,T,2112]`。 | **保留** 旧 SFT 键和 4 层/768 宽结构。R2 仅为新路径适配 Bridge 条件，明确 `2048 → 768` 的唯一所有者，并独立声明 Talker 的配置与 Cache。现有 `test_omni.py`、`test_omni_generation.py`、`test_codec.py`。 |
| [omni/models/__init__.py](../../omni/models/__init__.py) | 导出旧模型与 `TransformerLM` 兼容入口，将 `omni-plus` 的 HF `AutoModelForCausalLM` 映射到文本类。 | **保留** 注册和导入契约。R1 Qwen 调用官方加载器。`test_thinker.py`、`test_omni.py` 校验旧入口及回环。 |
| [omni/models/thinker/__init__.py](../../omni/models/thinker/__init__.py) | 旧配置、主干与 CausalLM 的便捷导入，`test_thinker.py`、`test_omni.py` 直接使用。 | **保留** 旧导出。R1 优先从新模块显式导入 Backend，需要公开时仅增量导出。 |
| [omni/models/encoders/__init__.py](../../omni/models/encoders/__init__.py) | P2 脚本、编码器测试导入冻结编码器与 Projector。 | **保留** 现有导出，R2 直接复用。 |
| [omni/models/talker/__init__.py](../../omni/models/talker/__init__.py) | 导出 `AudioCodes`、Talker、Embedding 和 Head。 | **保留** 现有导出，新条件接口的验收归 R2。 |

### 2 数据、训练、推理与入口

| 现有文件 | 定义、调用方与现状 | 迁移动作与验收归属 |
| --- | --- | --- |
| [omni/data/omni_dataset.py](../../omni/data/omni_dataset.py) | `OmniDataset` 先展开媒体标记，再检查 generation prompt 的 Token 序列是完整对话的前缀，据此定位最后一轮回答并建立音频延迟监督。`OmniCollator` 对齐 batch，保留模态列表及 `None`。两套标签已移位一次，ignore 为 `-100`。供 P2 和 P4 缓存/训练使用。 | **保留** MiniMind 路径。R2 用真实样本验 Qwen 时间轴和标记，R3 再生成专用缓存。现有 `test_dataset.py`、`test_p4_data.py`。 |
| [omni/data/audio_alignment.py](../../omni/data/audio_alignment.py) | `delay_codes()`、`restore_codes()`、`code_mask()`，由 P3 脚本验证可逆。Dataset 与 `stream_generate()` 各自维护训练写入、推理逐步复原逻辑。 | **保留** 八路和特殊 ID。R2 以同一声学时间轴契约测试这三处，查明重复边界后再决定抽离方式。现有 `test_codec.py`、`test_dataset.py`、`test_omni_generation.py`。 |
| [omni/data/prepare.py](../../omni/data/prepare.py) | `prepare_splits()` 按内容与媒体去重，`cache_split()` 固化序列、监督、特征与配置。由 `prepare_p4.py` 调用，collector 复用 `media_hashes()`。 | **复用** 哈希、去重和特征缓存能力。R3 另建数据 manifest、独立 split 和版本命名空间，分别验证许可、缓存失效及 tokenizer/codec 匹配。现有 `test_p4_data.py`、`test_collect_p4.py`。 |
| [omni/data/__init__.py](../../omni/data/__init__.py) | 导出 Dataset、Collator，多处脚本与测试直接导入。 | **保留** 公开导出。R3 新入口增量接入。 |
| [omni/training/losses.py](../../omni/training/losses.py) | `loss_counts()`、`sft_loss()`。文本/音频分开统计有效监督量，逐码本归一化，音频 stop 可加权。调用方为 `sft.py`。 | **复用** P4 已验证数学定义，R3/R4 单独核对新 logits 与监督的对应关系。现有 `test_sft.py` 手算、全忽略、梯度和累积一致性覆盖。 |
| [omni/training/sft.py](../../omni/training/sft.py) | `configure_trainable()` 依赖 `model.model.layers` 与 MiniMind 参数前缀，`train_step()`/`evaluate()` 使用音频 stop 配置。Checkpoint 保存 model、optimizer、scaler、config、trainable、progress 和 RNG，恢复会核对配置及冻结集合并严格载入。 | **保留** 旧训练与恢复。R3 抽取或新增 v2 参数分组和 checkpoint 格式，R4 验证 LoRA、新模块与上游输入梯度。现有 `test_sft.py`、`test_sft_cli.py`。 |
| [omni/training/__init__.py](../../omni/training/__init__.py) | 包标记，目前调用方直接导入 `losses`、`sft`。 | **保留**，按 R3/R4 实际入口决定是否导出新函数。 |
| [omni/inference/codec.py](../../omni/inference/codec.py) | `MimiCodec` 冻结注入的 HF 模型，整段 decode 接收 `[1,8,F]` 声学 codes。`load_audio()`/`save_audio()` 管理单声道 IO 和重采样，供 P2/P3/P5 使用。 | **保留** 24 kHz、八码本、ID 范围和返回约定。R2 复用并对照同版 Mimi，R7 另建评测入口。现有 `test_codec.py`。 |
| [omni/inference/streaming.py](../../omni/inference/streaming.py) | `MimiStreamingDecoder` 为单话轮持有 Mimi Attention KV 和有限卷积历史。`stream_waveform()` 将 `(token, frame)` 事件转为 `WaveformEvent`，自然结束时解码尾块，关闭时释放上游生成器。P5 Demo/基准直接使用。 | **保留** 波形事件、尾块与取消语义。R2 通过新生成器输出兼容事件，分别重置 Qwen、Talker、Mimi 缓存。现有 `test_streaming.py`、真实 `check_p5.py`。 |
| [omni/inference/__init__.py](../../omni/inference/__init__.py) | 导出 Codec、流式解码、波形 IO，Demo 和测试直接使用。 | **保留** 现有导出。 |
| [omni/__init__.py](../../omni/__init__.py) | 包标记，由 `test_environment.py` 检查导入。 | **保留**。 |
| [omni/evaluation/__init__.py](../../omni/evaluation/__init__.py) | 包标记，现有基准在 `scripts/`。 | **保留**。正式跨任务评测随 R4/R7 产生可用数据后落地。 |

### 3 资产、配置与工程基线

| 现有文件 | 调用方与边界 | 后续动作 |
| --- | --- | --- |
| [configs/assets.yaml](../../configs/assets.yaml) | README 引用的路径清单。现有验收脚本各自接收 CLI 路径或定义默认值，未找到运行时加载此 YAML 的调用。 | **保留** 历史清单。R1 新增 `assets_v2.yaml`，由新入口显式读取。 |
| [configs/model/thinker_tiny.json](../../configs/model/thinker_tiny.json) | `test_thinker.py` 读取的 MiniMind 小模型配置。 | **保留**。R1 的 Qwen tiny fixture 与 1.7B 正式配置各自独立。 |
| [configs/training/p4_data.yaml](../../configs/training/p4_data.yaml) | `prepare_p4.py` 的历史配置，关联 P4 样本源与缓存。 | **保留**。R3 用独立数据配置、split 和缓存路径。 |
| [configs/training/p4_overfit.yaml](../../configs/training/p4_overfit.yaml) | `check_p4.py`、`train_sft.py` 默认配置。 | **保留**。R3/R4 新建全宽模型的试验配置，保存真实配置、来源与恢复元数据。 |
| [pyproject.toml](../../pyproject.toml)、[uv.lock](../../uv.lock)、[CI](../../.github/workflows/ci.yml) | Python 3.12、pytest/Ruff 与 Ubuntu CPU 基础检查，测试用例默认离线。 | **保留** 当前环境及锁。R1 先验证现有 Transformers 的接口，任何必要变更单独记录。1.7B 实测只由显式脚本启动。 |
| [.gitignore](../../.gitignore)、`experiments/p1/`–`experiments/p5/`、`outputs/` | 权重、tokenizer、数据、缓存、WAV 与输出二进制受忽略规则保护。阶段 JSON 等小型证据保留。 | **禁止修改** P0–P5 历史报告与原有实验结果。新结果写 `experiments/r*/<run_id>/`，大文件写独立忽略目录。旧 `../minimind-o` 只读。 |

## 三、旧脚本与测试的回归归属

### 1 现有入口

| 脚本 | 已有用途 | 迁移归属 |
| --- | --- | --- |
| [check_minimind.py](../../scripts/check_minimind.py) | 本地 Dense/MoE 严格载入、层输出、Router、Cache、HF 往返和文本生成。 | P1 复现入口保留，R1 新 Backend 不替代本脚本。 |
| [check_qwen3.py](../../scripts/check_qwen3.py) | 官方 Qwen3-0.6B 文本基线，独立 HF AutoModel/Tokenizer，收集由 `--download` 显式触发。 | P1 入口保留，R1 新脚本对照原生 HF 和 Backend。 |
| [check_p2.py](../../scripts/check_p2.py) | 本地真实 Encoder、旧 Omni SFT 与混合 batch。 | P2 入口保留，R2 另建新路径验收。 |
| [check_p3.py](../../scripts/check_p3.py) | Mimi 整段还原、旧版 logits/Cache 对照、文本生成 WAV。 | P3 入口保留，R2 复用 codec-only 方法。 |
| [check_p4.py](../../scripts/check_p4.py) | 小样本任务 smoke 和训练恢复对照，调用 `train_sft.run()`。 | P4 入口保留，R3/R4 的全宽模型独立实测。 |
| [collect_p4.py](../../scripts/collect_p4.py) | 带传输预算的远端小样本收集，与普通测试分开。 | P4 入口保留，R3 先做许可、字段、大小审计。 |
| [prepare_p4.py](../../scripts/prepare_p4.py) | 旧样本去重划分、可选冻结特征缓存。 | 保持 P4 默认参数，R3 复用函数并另开数据入口。 |
| [train_sft.py](../../scripts/train_sft.py) | 构造 MiniMind Omni，读取 P4 缓存并运行训练/恢复。 | 保留 P4 CLI，R3/R4 新增 v2 入口及显式冻结策略。 |
| [check_p5.py](../../scripts/check_p5.py) | 真实 MiniMind SFT 与 Mimi 的 Cache/生成/PCM 性能测量。 | P5 入口保留，R2 复用 PCM 验法，R7 复用计时口径。 |
| [demo_p5.py](../../scripts/demo_p5.py) | 本地 MiniMind 文本和语音流，PCM 写 WAV。 | P5 入口保留，v2 Demo 由新训练权重与独立验收支撑。 |

### 2 已有单测文件

| 测试文件 | 当前覆盖与保留条件 |
| --- | --- |
| [test_environment.py](../../tests/test_environment.py) | P0 包导入、依赖、CPU Tensor 和可用时的 MPS。 |
| [test_thinker.py](../../tests/test_thinker.py) | P1 配置、Dense/MoE、权重/HF 往返、生成、Cache、padding 与位置行为。 |
| [test_p1_minimind.py](../../tests/test_p1_minimind.py) | P1 权重复制与源绑定、容差、层输出快照、参考源码只读。 |
| [test_p1_qwen.py](../../tests/test_p1_qwen.py) | P1 本地 tiny Qwen 官方路径、词表 padding 和超界保护。R1 Backend 测试单独新增。 |
| [test_encoders.py](../../tests/test_encoders.py) | P2 冻结、有效长度、视觉顺序与本地路径约束。 |
| [test_omni.py](../../tests/test_omni.py) | P2/P5 多模态注入、前反向、不同宽度、Bridge 指定 Block、联合 Cache 和 HF 回环。 |
| [test_codec.py](../../tests/test_codec.py) | P3 八路延迟还原、声学 ID、Mimi 冻结与波形 IO。 |
| [test_dataset.py](../../tests/test_dataset.py) | P2/P4 模态索引、多轮、长音频监督、标记与单次 shift，含极小本地 HF tokenizer。 |
| [test_collect_p4.py](../../tests/test_collect_p4.py) | P4 模拟 HTTP Range、预算、响应体长度和哈希。 |
| [test_p4_data.py](../../tests/test_p4_data.py) | P4 媒体去重、划分稳定性、缓存特征索引和标签。 |
| [test_sft.py](../../tests/test_sft.py) | P4 loss、累积、冻结、AMP 和 checkpoint/RNG 恢复。 |
| [test_sft_cli.py](../../tests/test_sft_cli.py) | P4 CLI 恢复与持续训练一致，拒绝覆盖缓存标签契约。 |
| [test_omni_generation.py](../../tests/test_omni_generation.py) | P5 文本/音频停止、思考结束、参考码本、Cache、左 padding 和预算。 |
| [test_streaming.py](../../tests/test_streaming.py) | P5 PCM 数值与部分尾块、关闭生成器与话轮隔离。 |

## 四、R1 拟新增文件与接口风险

### 1 文件落点

| 拟新增路径 | 职责 | 对照与验收 |
| --- | --- | --- |
| `omni/models/thinker/backend.py` | 最小 `ThinkerBackend` 契约，声明 `input_ids`/`inputs_embeds`、位置、mask、指定层特征、logits 与原生 cache 的所有权。 | 契约由 tiny fixture 覆盖。现有 MiniMind `forward_embeddings()` 和公开 tuple 保持原样。 |
| `omni/models/thinker/qwen_backend.py` | 使用官方 HF 模型的独立薄适配层。显式提供 embedding、前向与 Cache，保持对应 tokenizer/template 由调用方注入或加载。 | 同一官方权重原生前向逐项对照 logits、hidden、prefill/单步/多步 decode。 |
| `configs/model/qwen3_1_7b.yaml` | Qwen 专属本地路径、模板模式、选层与精度的显式配置。模型真实尺寸、层数、头数和特殊 ID 从实际资产读取。 | 拒绝把 MiniMind 配置当作 Qwen 模型结构。层号、模板与运行日志一致。 |
| `configs/assets_v2.yaml` | 新资产清单。记录官方来源、固定 revision、本地路径、许可证与核验状态，由新入口显式解析。 | R1 确认实际文件和哈希，资产缺失时明确失败。旧 `configs/assets.yaml` 保留。 |
| `tests/test_qwen_backend.py` | CPU tiny Qwen fixture，覆盖两种输入、层索引、Cache 生命周期和左 padding。 | 检查 `input_ids` 与等价 embedding、原生数值、Cache On/Off 生成、连续新话轮。普通 pytest 保持离线。 |
| `scripts/check_qwen_backend.py` | 显式对照本地官方 Qwen3-1.7B，固定模型、精度、模板和采样。既有 0.6B 只用于先行接口检查。 | 1.7B 真实中英文生成、权重加载、数值和缓存通过才完成 R1。 |
| `experiments/r1/<run_id>/manifest.json`、`docs/stages/R1.md` | 记录 repo/revision、文件 SHA、真实模型配置、tokenizer/template、设备与精度、误差和结论。 | 真实未执行项目逐项保留状态。新 run 独立保存，普通 CI 只跑 fixture。 |

### 2 兼容约束

1. **文本词表与标记。** MiniMind 默认词表 6400、BOS 1、EOS 2，`pad_token_id` 默认 `None` 而历史脚本常显式取 0。旧图像/音频占位分别为 12/16，思考结束为 `[26,234,234]`。R1 根据官方 Qwen tokenizer/config/template 记录实值，验证词表对齐额外行、左 padding、思考模板和 EOS。R2 再决定媒体标记映射、冲突测试和新增行是否训练，并重新检查完整模板和 generation prompt 的前缀契约。
2. **Cache 与位置。** 旧 `PastKeyValues` 为逐层 `(K,V)` tuple，张量为 `[B,Hkv,T,Dhead]`。MiniMind 的逻辑位置由 mask 累积得到，旧 Omni 在 8 层 Thinker 与 4 层 Talker 之间切分。当前本地 Transformers 的 Qwen3 前向公开 `Cache`/`DynamicCache` 与 `cache_position`，位置语义须用原生对照固定。测试物理缓存位置与逻辑 `position_ids`、prefill/decode mask、左 padding 及独立请求重置。Qwen、Talker 与 Mimi 状态由各自模块持有。
3. **Bridge 取层与投影。** MiniMind 缺省在零基第 3 个 Block 后、最终 norm 前取特征。HF 输出还涉及 embedding 项和最终 norm，R1 用 native fixture 核对索引及取点。R2 独立定义 `2048 → 768` 的单一投影并处理旧 `talker.embed_proj` 的职责，避免改变旧权重键或把 Talker 扩宽。
4. **训练数据与冻结。** Text EOS、声码 stop 2050、pad 2049、speaker 2051 分别属于不同语义。音频声学 ID 仅为 `[0,2047]`。替换 tokenizer 后重新构建联合标签与 mask，校验短文本/长声码与因果历史。进入 Projector 训练后，冻结 Qwen 参数仍需允许其前向图将梯度传给投影输入。
5. **权重与恢复。** 历史 MiniMind 仍按原 `.pth` 与 HF 目录严格加载。R1 直接载入 Qwen 对应的官方 HF 目录。R3/R4 的新恢复配置须绑定基础模型 revision、Tokenizer/Codec、独立模块和可训练组，沿用严格缺项检查与 RNG/进度恢复。

## 五、R2–R7 阶段边界

| 阶段 | 复用或新增的候选落点 | 开始依赖与实测关口 |
| --- | --- | --- |
| R2 模块迁移 | `omni/models/model_omni_v2.py` 并行组合层、独立 Fixed Bridge。复用现有 Encoder、Projector、Talker 结构与 Mimi 流式事件，最小适配 Talker 条件。 | R1 真实数值与 Cache 通过。两条真实 T2A 监督时间轴、I2T/A2T/A2A 路由、冻结与梯度、codec-only PCM、Cache/尾块/取消。新 Talker 的自然语音留给 R4。 |
| R3 数据与资源 | 从 `omni/data/prepare.py`、`omni/training/{sft,losses}.py` 复用工具，另建 v2 prepare/train/eval 入口、数据配置与 manifest。 | R2 标签契约通过。先小样本审计来源、许可、去重、Codec/Tokenizer/Encoder 指纹，生成新 split/cache。经授权做单卡实测前后向与恢复，预算由实测决定。 |
| R4 正式 SFT | 在 R3 v2 入口扩展 T2A、输入模态对齐和联合训练配置。版本化保存 Bridge/Talker/Heads、Projectors、可选 Qwen LoRA 与完整训练状态。 | R3 数据与资源通过，用户确认规模。先冻结 Qwen 训练新语音模块，再训练输入投影及按需 LoRA。实际可懂语音、模态条件有效性、独立评测和可恢复 B1 checkpoint。 |
| R5 Bridge 研究 | 在 R2 Bridge 接口增设候选与消融配置、评测脚本和独立 run。 | 冻结 R4 B1、split、Talker 初始化与预算，至少三个可追溯候选分别训练和评测，记录资源与负结果。 |
| R6 GRPO 研究 | 新增独立 RL 训练、rollout 与 reward 文件，路径由用户确认设计后落地。保留原 SFT/推理路径。 | 稳定 B1 与可验证 reward，用户决定策略/预算。先验证行为分布、多码本有效动作 logprob 与 loss/mask，再做同 SFT 起点的受控对照。 |
| R7 评测与交付 | 在 `omni/evaluation/` 与新 benchmark/demo 入口复用 P5 指标口径，补齐技术报告、数据许可与真实结果。 | R4 有可评测 B1，R5/R6 结果或延期状态明确。固定同设备、batch、长度、精度和采样的测量条件，保留原始 JSON/CSV。 |
