## 一、外部集成

| 系统 | 类型 | 用途及重要性 | 认证入口 | 依据 |
| --- | --- | --- | --- | --- |
| Hugging Face Hub | 模型文件与元数据 API | Qwen3-0.6B 资产的固定 revision 和文件校验，供独立文本基线 | 脚本未传入显式 token | `scripts/check_qwen3.py` |
| Hugging Face 数据集 HTTP | 公开文件 Range 读取 | 官方 P4 小子集收集，后续 prepare/cache 读取本地 Parquet | 请求不带项目认证头 | `scripts/collect_p4.py` |
| Apache Parquet 格式文件 | GitHub raw 文件读取 | `collect_p4.py` 解析 Parquet 页的格式定义及许可 | 公开 URL | `scripts/collect_p4.py` |
| MiniMind-O / FunASR / HF 本地模型 | 外部目录和模型组件 | 历史 Thinker、SenseVoice/SigLIP2、Mimi 与 Dense SFT 验收所需资产 | 本地文件权限 | `configs/assets.yaml`、`omni/models/encoders/model_encoders.py`、`scripts/check_p3.py` |
| GitHub Actions / PyTorch CPU 索引 | CI 与依赖安装 | Ubuntu 基础回归。CI workflow 权限为 `contents: read` | workflow 权限和公共包索引 | `.github/workflows/ci.yml` |

当前运行入口的数据存取基于文件系统。依赖声明和实现中没有看到数据库、消息队列、API 网关或 APM 客户端配置。

## 二、数据存放

| 存储 | 角色及访问层 | 兼容约束 |
| --- | --- | --- |
| `weights/minimind/`、`tokenizers/minimind/` | `check_minimind.py` 读取 Dense/MoE 及本地 tokenizer | 严格权重键、绑定权重和词表对应 |
| `weights/qwen3-0.6b/` | `check_qwen3.py` 的 HF 模型与词表目录 | 固定 revision `c1899de289a04d12100db370d81485cdf75e47ca` |
| `../minimind-o/model/`、`out/` | 冻结编码器、Mimi、历史 Omni SFT 权重 | 路径只读引用，模型版本与历史报告对应 |
| `cache/p4/sources-v2/`、`.cache/p4/prepared/`、`cache/p4/` | PyArrow 子集、训练/验证记录、已缓存样本 | 数据身份、词表和标签约定与恢复检查相符 |
| `outputs/p4/`、`outputs/p5/` | 训练 checkpoint 与本地波形 | checkpoint 与模型配置、冻结方式、优化器状态对应 |
| `experiments/` | 运行摘要、权重/数据校验值、CSV/曲线 | 按来源和 run 识别，R0 保护历史记录 |

R0 读取确认 MiniMind Dense/MoE、Qwen3-0.6B、P4 两份特征缓存和 step-600 checkpoint 文件存在。Qwen3-1.7B 的资产目录 `weights/qwen3-1.7b/` 尚不存在，其接入在 v2 计划中安排于 R1。

## 三、凭据与授权

- `.env.example` 仅列离线选项。`.gitignore` 忽略 `.env` 及其变体。对 `omni/`、`scripts/`、`tests/` 和 workflow 搜索了常见 token 变量名及私钥标记，未发现匹配项。
- 在线数据入口由用户显式启动。Qwen 脚本通过 `--download` 进入下载，P4 通过 `collect_p4.py` 收集有限子集。HF 模型和 tokenizer 的离线检查使用本地目录及 `local_files_only=True`。
- [TODO] 私有资产场景的账户、组织权限和凭据轮换策略在当前仓库中未定义。

## 四、失败处理

- P4 Range 请求连接/读取超时为 15/60 秒，元数据和格式文件请求超时为 30 秒。代码先检查 HTTP 206 与 `Content-Range`，再读受限长度的 body。预算或范围不符时抛错。
- Qwen 资产元数据查询及 ETag 超时为 20 秒，限制下载文件集合，并核对大小和已提供的 LFS 哈希。脚本层未额外实现重试或熔断。
- 本地验收使用明确的设备与路径。缓存、checkpoint 或权重和预期配置不符时按对应入口报错。

## 五、可观察性

- Qwen 保存固定 revision、文件大小和 SHA-256。P4 保存 Range 起点、长度、哈希、保留的源行和累计传输量，另存本地 Parquet 的摘要。
- 训练和推理使用终端进度、JSON、CSV 和性能报告。P5 记录首 Token、首块音频、块间隔、RTF、RSS 与重复试验设置。
- [TODO] 仓库未定义跨运行集中监控或追踪后端，现有审查依赖本地报告与终端错误。

## 六、证据

- [外部资产路径](../../configs/assets.yaml)、[可选环境变量](../../.env.example)、[忽略规则](../../.gitignore)
- [Qwen 资产获取与离线检查](../../scripts/check_qwen3.py)、[P4 受限收集](../../scripts/collect_p4.py)、[P4 资产记录](../../experiments/p4/subset-assets-v2.json)
- [冻结编码器](../../omni/models/encoders/model_encoders.py)、[训练恢复](../../omni/training/sft.py)、[P5 基准记录](../../experiments/p5/cpu-fp32-20261105.json)
