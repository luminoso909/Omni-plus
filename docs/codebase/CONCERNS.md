## 一、迁移风险

### 1 按优先级排序

| 优先级 | 已核对的现状 | 可能的影响 | 后续验收 |
| --- | --- | --- | --- |
| 高 | MiniMind 默认词表 6400、BOS/EOS 1/2、音频/图像占位 16/12。Qwen3-1.7B 官方公开配置为 hidden 2048、28 层、16 Q/8 KV、head dim 128、词表 151936、绑定词嵌入和词表头。本地 0.6B 的 hidden 是 1024 | 混用文本 ID、模板或仅用 0.6B 核对 2048 投影会掩盖接口错误 | R1 固定 1.7B revision 并离线复验实际资产。现有 0.6B 只用于接口先行。 [MiniMind 默认值](../../omni/models/config.py) · [本地 0.6B 资产](../../configs/assets.yaml) · [官方 1.7B 配置](https://huggingface.co/Qwen/Qwen3-1.7B/raw/main/config.json) |
| 高 | `OmniDataset` 分别渲染完整对话和生成前缀，并要求 token 级前缀一致。本地 Qwen3-0.6B 的模板可在非思考生成前缀插入空思考段，`</think>` 也未标为 `special:true`。其 generation 配置的 EOS 为 151645 和 151643，后者同时是 PAD | 旧 `think_end_ids`、字符切分或 `id != pad` 推 mask 会错置正文、停机和音频起点 | R1/R2 从固定资产分别读 tokenizer、template、generation 配置，验证中英、多轮、空思考、正常文本碰撞和左 padding。 [旧模板处理](../../omni/data/omni_dataset.py#L236-L254) · [0.6B 检查](../../scripts/check_qwen3.py) |
| 高 | MiniMind 用逐层 KV 元组，Omni 按层拼接 Thinker/Talker。Qwen 使用可变 `Cache`，注意力收到非空对象时会调用 `update()`，即使外层 `use_cache=False` | 拼接或复用旧话轮状态会破坏 cache 位置、开关对照与取消语义 | R1 分开物理 `cache_position` 与逻辑 RoPE 位置，并在禁用 cache 时明确拒绝或丢弃旧对象。R2 分离三路缓存。 [旧 Omni Cache](../../omni/models/model_omni.py#L89-L141) · [Qwen Attention v4.57.6](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen3/modeling_qwen3.py#L185-L224) |
| 高 | Dataset 已把 inputs 去最后一位、labels 去第一位。Qwen 的原生 CausalLM loss 还会移动未对齐的原始标签 | 重复 shift 会监督错误的文本位置。音频 STOP 也可能被声学值掩码丢掉 | R2 保留外部 `sft_loss()`，确认模型前向的 logits 和 labels 对齐；单测 STOP、PAD 和无监督位置。 [移位](../../omni/data/omni_dataset.py#L304-L315) · [外部 CE](../../omni/training/losses.py) · [HF causal loss v4.57.6](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/loss/loss_utils.py#L45-L62) |
| 高 | `train_sft.py` 当前从 MiniMind 缓存构造 `OmniPlusOmni`。`configure_trainable()` 固定访问 `model.model.layers`，checkpoint 保存全模型并核对 config/可训练参数名 | 直接沿用旧入口会绑定错误的模块、优化器分组和检查点格式 | R3/R4 单独验证 Qwen 冻结组、梯度路径、基础模型 revision、可训练权重和严格恢复。 [训练入口](../../scripts/train_sft.py#L66-L132) · [冻结与恢复](../../omni/training/sft.py) |

### 2 需处理的既有耦合

| 耦合 | 当前来源 | 迁移边界 |
| --- | --- | --- |
| Bridge 与 Talker | 旧 Bridge 为指定 Block 特征，`talker.embed_proj` 内部完成 Thinker 宽度→Talker 宽度，旧键参与 195 键严格加载 | R2 先验证非末层 Block 的归一化前特征，再为 Fixed Bridge 的 2048→768 投影定单一所有者。旧权重继续走原路径。 [Talker 投影](../../omni/models/talker/model_talker.py#L92-L111) · [P3](../stages/P3.md) |
| 模态占位符 | `_expand_marker()` 用输入字符串计数和扩展占位，`_inject_features()` 通过 token ID 定位。在本地 0.6B `added_tokens_decoder` 中有图像占位 `<\|image_pad\|>`，未见 `<\|audio_pad\|>` | R1/R2 验证固定 1.7B 的实际编码和碰撞路径。若新增 token，需核对绑定权重的 embedding/head 行、冻结与回环。 [占位展开](../../omni/data/omni_dataset.py#L136-L157) · [特征注入](../../omni/models/model_omni.py#L48-L71) · [本地资产定位](../../configs/assets.yaml) |
| 音文联合时轴 | Dataset 手工构造目标 delay，生成端按对角线还原，`audio_alignment.py` 另有独立工具。`code_mask()` 只识别声学 ID，Dataset 的 CE 标签含 STOP。音频延长区的文本可填 PAD ID，attention mask 仍为 1 | R2 使用两条真实样本核对长短文本和声码、尾部和停止位置，记录监督 mask 与上下文 mask。 `code_mask()` 的声学语义不能代替 loss mask。 [对齐](../../omni/data/audio_alignment.py) · [数据时轴](../../omni/data/omni_dataset.py#L247-L350) · [生成循环](../../omni/models/model_omni.py#L277-L348) |
| 数据与缓存 | `cache_split()` 同时保存预编码特征、文本 IDs、音频 labels 和模型 config。P4 训练核对接口字段及缓存文件哈希，缓存记录中没有独立 tokenizer/template 或 codec revision | R3 可核对后复用未改变编码器版本的原始特征与兼容 Mimi 原码，重新生成 Qwen 的派生 IDs/labels，登记输入版本。 [缓存格式](../../omni/data/prepare.py#L160-L204) · [训练校验](../../scripts/train_sft.py#L46-L97) |

### 3 资产与安全

| 风险 | 已有约束 | 待补证据 |
| --- | --- | --- |
| 外部 checkpoint 与参考源码 | 历史真实加载使用本地路径和 `weights_only=True`，P3 参考对照显式编译旧仓源码，涉及执行本地参考代码 | [TODO] R1 固定并校验官方 Safetensors revision，使用官方内置模型类并显式关闭远程自定义代码。P3 对照仍仅使用已知可信的只读参考路径。 [P3 脚本](../../scripts/check_p3.py) · [资产表](../../configs/assets.yaml) |
| 数据许可、音频可识别性与跨 split 泄露 | P4 划分按记录和媒体哈希去重，子集报告限制本地使用。公开摘要与原始缓存分开 | [TODO] R3 逐来源核验许可、媒体分发边界、speaker 和独立评测集。现有哈希去重的结论适用于已有子集。 [划分实现](../../omni/data/prepare.py#L56-L157) · [P4](../stages/P4.md) |

### 4 性能与规模

| 风险 | 当前证据 | 验收时记录 |
| --- | --- | --- |
| Qwen 词表投影占用 | 原生 Qwen 支持 `logits_to_keep`，旧 SFT 对完整文本 logits 转 FP32。按 151936 词表，`B=1,T=2048` 的单个 BF16 logits 张量理论约 594 MiB | R1 解码测末位 logits，R3 记录完整 SFT logits、CE 和实际峰值。上述数字为尺寸估算。 [Qwen 配置](https://huggingface.co/Qwen/Qwen3-1.7B/raw/main/config.json) · [HF forward v4.57.6](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen3/modeling_qwen3.py#L480-L505) · [现有 loss](../../omni/training/losses.py) |
| 读取中间层与冻结后的反传 | HF 启用隐藏层收集时保留各层，4.57.6 会临时替换 Block `forward`，在正常返回后恢复。旧 `train_step()` 调用 `model.train()`，冻结参数与是否保存输入梯度是不同控制 | R1 对照使用单话轮串行路径。R2/R3 核对所需层、计算图和训练/冻结状态。训练 Projector 时，梯度需经过冻结的 Qwen 返回输入嵌入。 [HF 输出收集 v4.57.6](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/utils/generic.py#L1002-L1103) · [训练步骤](../../omni/training/sft.py#L50-L101) |
| 样本与 checkpoint 随规模放大 | 旧数据准备和训练将已处理样本列表读入内存，checkpoint 每次保存完整模型。旧 P5 Mimi 流在同一话轮保留全部 Transformer KV | R3 用全宽度和真实时轴测峰值、磁盘与吞吐，R4 使用单独的恢复契约。旧 P4/P5 结果用于历史复现。 [数据准备](../../omni/data/prepare.py#L173-L204) · [checkpoint](../../omni/training/sft.py#L128-L190) · [P5](../stages/P5.md) |

### 5 变动集中与回归边界

| 区域 | 90 天 Git 历史与当前代码 | 回归依据 |
| --- | --- | --- |
| Thinker 导出和旧模型名 | `omni/models/thinker/__init__.py` 有 4 次历史变动，旧 `model_lm.py` 有 2 次。当前 `model_thinker.py` 约 590 行，含 Block、LM 封装和文本生成 | 保留公开类名、导入与权重键，使用 `test_thinker.py` 和 P1 显式严格加载对照 |
| Omni 组合与 Talker | `model_omni.py`、`model_talker.py` 各有 2 次历史变动，工作树另含 P5 共享前向与流式边界 | 以 `test_omni.py`、`test_omni_generation.py`、`test_streaming.py` 与历史 195 键对照保护旧路径 |
| 配置与依赖 | `pyproject.toml` 有 5 次、`uv.lock` 4 次、`configs/assets.yaml` 2 次。P5 Mimi 流直接依赖已验证版本的子模块 | R1 保留 4.57.6 回归并记录版本，若调整依赖再核对 codec-only 数值和 P5 流式合同 |

扫描器会把被 Git 忽略的 tokenizer 内容算作 TODO/FIXME 字样。以 `rg -n 'TODO|FIXME|HACK' omni` 复核，生产包内未发现此类标记。这与上面的静态风险是两类证据。

### 6 留待阶段决策

1. [ASK USER] R1 启动时，优先复用用户提供的 1.7B 目录，还是由用户授权后获取固定 revision？
2. [ASK USER] R2 是否需要末层归一化前 Bridge 特征，还是先固定一个可直接核对的中间层？
3. [ASK USER] R3 确定首个真实单卡 profile 时，使用哪个设备和数据范围？正式预算根据该次实测决定。

### 7 证据索引

- [依赖锁](../../uv.lock) · [资产路径](../../configs/assets.yaml) · [配置默认值](../../omni/models/config.py)
- [现有 Dataset](../../omni/data/omni_dataset.py) · [Delay 与还原](../../omni/data/audio_alignment.py) · [训练和恢复](../../omni/training/sft.py)
- [Qwen3 v4.57.6 官方实现](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen3/modeling_qwen3.py) · [Qwen3-1.7B 官方配置](https://huggingface.co/Qwen/Qwen3-1.7B/raw/main/config.json)
- Git 历史复核命令 `git log --since=90.days --format= --name-only -- omni pyproject.toml uv.lock configs/assets.yaml`。统计不包含工作树未提交变动。
