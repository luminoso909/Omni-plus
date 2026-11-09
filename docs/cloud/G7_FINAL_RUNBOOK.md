## 一、用途与前置条件

本手册基于 `6c497ca` 的真实 CLI，经本轮开发完成度审计补充。**G7 未执行；以下云端命令均待授权和实测。** 本地 R2–R6-L PASS 证明接口、数学及 Tiny 闭环，不证明 B1 或研究收益。 [开发审计与缺口编号](../reports/OMNIPLUS_DEVELOPMENT_AUDIT.md)

**B1 首次 Benchmark 必须在 G4D 结束时执行。G7 是固定协议下的最终统一复核，不是第一次评测。** 当前统一 Benchmark 仍有待实现入口，下文明确分列；不要把旧规划中的 `eval_b1.yaml` 或 `--checkpoint` 当成现有命令。

进入 G7 前检查：

- [ ] G3 有真实硬件、CUDA/精度、前反向/恢复、资源和预算记录；G4 有验收后的 B1 SFT Checkpoint、Adapter、原始配置、首次 Benchmark 和可懂语音/模态有效性证据。
- [ ] G5/G6 各自有同 B1 的独立研究结果，或明确列为失败/延期/未执行；不能把未做的实验列为获胜方法。候选与 Reward 不在本手册预设优劣。
- [ ] 固定 Text、I2T、A2T、T2A、真实 QA A2A 的独立测试集、许可、媒体身份、语言/长度分层和评测协议；validation 用于选模型，test 不再用于调参。
- [ ] B1、祖先组件及所有候选的数据契约 v3、Checkpoint/源码/资产 SHA 与训练来源可追溯；B1/后续训练媒体不得进入 test。未登记转码关系另做审计，不能仅依赖字节哈希。
- [ ] 审计 A1/C1 的只读评测加载和全任务生成评测、C2 的统一性能计时已完成必要开发；缺项时只能执行现有诊断，不能签收完整 G7。
- [ ] 获得 G7 有限评测/优化预算与发布范围授权。性能实验不自动触发训练或改变模型结构。

## 二、固定代码、环境与产物目录

以下命令在获批 Linux CUDA 主机的新 checkout 执行。`G7_COMMIT` 必须是已交付的实际提交；若尚未 Push，由负责人另行授权 Push 或用 Git bundle 私下搬运。本轮不执行这些操作。

```bash
set -euo pipefail
: "${G7_REPO_URL:?填写已核实的仓库地址}"
: "${G7_COMMIT:?填写实际完整提交SHA}"
: "${G7_CHECKOUT:?填写尚不存在的云端checkout绝对路径}"
test ! -e "$G7_CHECKOUT"
git clone "$G7_REPO_URL" "$G7_CHECKOUT"
cd "$G7_CHECKOUT"
git switch --detach "$G7_COMMIT"
test "$(git rev-parse HEAD)" = "$G7_COMMIT"

python3.12 -m venv .venv
source .venv/bin/activate
: "${G7_TORCH_VERSION:?填写G3已验证版本}"
: "${G7_TORCHAUDIO_VERSION:?填写配套版本}"
: "${G7_TORCH_INDEX:?填写G3已验证的官方CUDA wheel索引}"
python -m pip install "torch==$G7_TORCH_VERSION" "torchaudio==$G7_TORCHAUDIO_VERSION" --index-url "$G7_TORCH_INDEX"
python -m pip install -e '.[dev,media,encoders,plots,data]'
python -m pip check
python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
python scripts/run_experiment.py --help
python scripts/bridge_ablation.py collect --help
nvidia-smi

G7_ROOT=outputs/g7/final-001
test ! -e "$G7_ROOT"
mkdir -p "$G7_ROOT/environment" "$G7_ROOT/logs"
git rev-parse HEAD > "$G7_ROOT/environment/commit.txt"
git status --short > "$G7_ROOT/environment/git-status.txt"
python -m pip freeze > "$G7_ROOT/environment/pip-freeze.txt"
nvidia-smi > "$G7_ROOT/environment/nvidia-smi.txt"
```

目录名每次更换；不覆盖旧结果。CUDA 依赖按 G3 锁定的相容环境安装，不能照搬 Mac wheel。实际导入来源检查要求环境安装到当前 checkout；不借用另一目录的 editable install。

`outputs/` 应位于有足够空间的持久盘，Checkpoint 备份另置。运行全过程冻结源码。正式完整回归可在此时执行，但本次审计只运行报告所列的轻量子集：

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m pytest -q > "$G7_ROOT/logs/pytest.log" 2>&1
ruff check omni tests scripts > "$G7_ROOT/logs/ruff.log" 2>&1
ruff format --check omni tests scripts > "$G7_ROOT/logs/format.log" 2>&1
python -m pip check > "$G7_ROOT/logs/dependencies.log" 2>&1
git diff --check
```

### 1 G3–G6 文档导航与旧命令纠正

G3 的 Git 跟踪路径为 `docs/G3_CLOUD_RUNBOOK.md`，本机用户工作区已将其移至 `docs/cloud/G3_CLOUD_RUNBOOK.md`，尚未提交。本轮没有接管这个迁移。其早期“train/resume/evaluate/profile 尚未实现”的描述已过时；`g3_h100.yaml` schema 1 仍最多 10 步。新 schema 2 的资源准备和实际命令以 [G4](G4_SFT_RUNBOOK.md) 为准；100 步需事先将配置预算设为 100 并授权。

[G5](G5_BRIDGE_RUNBOOK.md) 提供矩阵/谱系/Bridge 比较，[G6](G6_GRPO_RUNBOOK.md) 提供三种 RL 模式和基线比较。所有入口使用 `--run-dir` 和 `--resume`，没有 `--run-id`、`--checkpoint`、`--device` 或 `export` 子命令。`profile/grpo-profile` 都进行参数更新，**不在 G7 当作只读推理测速命令调用**。

## 三、取得权重、数据与来源清单

### 1 私有交付包

优先取得经过 G4/G5/G6 验收的有限交付包，包含：

| 目录/材料 | 必须保留 |
| --- | --- |
| Qwen/媒体资产 | 完整官方 Qwen3-1.7B 非 Base 目录，revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`；与 R2 相容的 SenseVoice/SigLIP/Mimi、Processor、Tokenizer/模板、许可证及发布方 SHA |
| B1 与候选 | 完整 Checkpoint、Adapter、原始 resolved YAML、训练 Manifest/metrics、祖先 Checkpoint 与 lineage/cache manifest；不能只拿一个 `.pt` 文件 |
| 数据包 | 审计 JSONL、官方 split 或独立分组规则、原媒体 ID/全部别名、文件 SHA、来源/revision/许可、目标转写与 QA 录音溯源 |
| 评测包 | B1 首次 Benchmark 协议/逐样本原始结果、锁定 test ID、ASR/评分器版本、随机种子与生成参数 |

对已有包用以下真实系统命令复制并校验；示例要求发布方提供包内相对路径的 `SHA256SUMS`：

```bash
: "${G7_BUNDLE_DIR:?填写已批准私有交付包目录}"
test ! -e "$G7_ROOT/imported"
mkdir -p "$G7_ROOT/imported"
rsync -a --checksum "$G7_BUNDLE_DIR/" "$G7_ROOT/imported/"
(cd "$G7_ROOT/imported" && sha256sum -c SHA256SUMS)
```

权重缺失时，先取得下载授权，按 `configs/assets_v2.yaml` 固定版本和摘要；相容媒体裁剪/转换权重不可用同名官方模型静默替换。G6 手册已有实际 `hf download` 用法，本手册不重复提供默认全量数据下载。完整正式数据必须来自已批准且审计的供应流程；通用 JSONL 入口不是所有上游压缩包的解析器。

### 2 缓存和身份

新的评测环境/Processor 变化需新缓存，保留旧清单；不要覆盖、重签或手改原契约。`prepare` 使用 SFT 配置，不接受 `grpo` 顶层键。对于尚未训练的新配置，可使用：

```bash
: "${G7_PREPARE_CONFIG:?填写已审计SFT数据配置路径}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
python scripts/run_experiment.py prepare --config "$G7_PREPARE_CONFIG" --dry-run
python scripts/run_experiment.py prepare --config "$G7_PREPARE_CONFIG"
```

这只准备数据，不能让一个历史 Checkpoint 自动接受新数据身份。当前完整恢复/评测严格绑定原配置、源码和缓存；跨源码/路径迁移须保留原环境复现，或使用**待实现并验收的 A1 只读评测入口**。不能删校验强行把历史模型塞进新协议。

## 四、YAML 与现有可执行评测

### 1 已有 YAML 字段

下列为当前 parser 支持的字段片段，需在 **G4/B1 或 G5 新训练配置冻结前** 合并；不是现存 Checkpoint 的可随意改写项，也不是一份独立可运行配置。资产和数据配置沿用 G4/G5 手册的完整格式。

```yaml
evaluation:
  max_new_tokens: 128
  asr_dir: weights/media-reference/model/SenseVoiceSmall
  benchmark:
    split: test
    t2a_samples: 8
```

`test`/`8` 是示例，需要真实独立 test 样本及用户锁定协议；不足时报错，不能重复样本凑数。训练周期内验证仍使用 validation，不能用此 test 指标选择 best。GRPO 使用 `grpo.evaluation_split/evaluation_prompts`，其字段也属于严格训练身份。新的 G7 独立配置 schema **待 A1/C1 实现**，不能把规划字段交给当前 parser 冒充已支持。

### 2 SFT / Bridge：已有诊断和 T2A Benchmark

`B1_CFG/B1_CKPT` 必须指向该 Checkpoint 的原始、可核验配置和完整状态；当前源码身份也须相同。先用 dry-run 检查路径/预算，但当前 SFT dry-run 不完整检查 Checkpoint 内容或实际 test split，真正 evaluate/demo 成功才是对应诊断证据。

```bash
: "${B1_CFG:?填写B1原始配置路径}"
: "${B1_CKPT:?填写B1完整Checkpoint路径}"
python scripts/run_experiment.py evaluate --config "$B1_CFG" --resume "$B1_CKPT" --dry-run > "$G7_ROOT/logs/b1-preflight.json"
python scripts/run_experiment.py evaluate --config "$B1_CFG" --resume "$B1_CKPT" --run-dir "$G7_ROOT/b1-loss" > "$G7_ROOT/logs/b1-loss.log" 2>&1
python scripts/run_experiment.py demo --config "$B1_CFG" --resume "$B1_CKPT" --run-dir "$G7_ROOT/b1-audio" > "$G7_ROOT/logs/b1-audio.log" 2>&1
```

`evaluate` 输出 `evaluation.json`：teacher-forced 文本/音频 Loss、每路 CE/STOP、条件合适时的模态调换。`demo` 另生成预测 WAV；若原配置含 `benchmark`，按相同 split/样本数生成 T2A 批次，写 `audio/samples.csv` 和每样本目录，逐样本完整记录位于 `evaluation.json` 的 `benchmark.samples`。**它不运行 Text/VQA/A2T/QA A2A 的统一生成 Benchmark，也不保证实时 PCM Demo。**

G5 候选按各自原配置和终点重复该命令。已有矩阵的汇总入口：

```bash
: "${G5_MATRIX:?填写原始matrix.json路径}"
python scripts/bridge_ablation.py collect --matrix "$G5_MATRIX" --output-dir "$G7_ROOT/g5-summary"
```

若结果已搬移/训练恢复在新目录，按 `collect --help` 及 G5 手册提供 `--artifacts` 路径映射；记录训练、评测目录与失败日志。此工具严格核对原矩阵、终点、来源/硬件/协议，**只汇总 G5 Bridge 矩阵，不是 B1+GRPO 的通用报告器**。旧矩阵不能手改 SHA 升格为新实验。

### 3 GRPO：已有 B1 与终点 Reward 对照

每种模式分别使用自己的原始 RL 配置；固定同一 B1、样本、评估 seed 和采样协议。当前实现可运行：

```bash
: "${RL_CFG:?填写Text或Audio或Joint的原始配置路径}"
: "${RL_CKPT:?填写对应完整RL Checkpoint路径}"
python scripts/run_experiment.py grpo-evaluate --config "$RL_CFG" --baseline --dry-run > "$G7_ROOT/logs/rl-baseline-preflight.json"
python scripts/run_experiment.py grpo-evaluate --config "$RL_CFG" --baseline --run-dir "$G7_ROOT/rl-b1" > "$G7_ROOT/logs/rl-b1.log" 2>&1
python scripts/run_experiment.py grpo-evaluate --config "$RL_CFG" --resume "$RL_CKPT" --run-dir "$G7_ROOT/rl-final" > "$G7_ROOT/logs/rl-final.log" 2>&1
```

更换模式必须更换结果目录和日志名，不能覆盖上一模式。输出 `evaluation.json`、`rollouts.jsonl`，包括 Reward 分项、文本/声码/Mask、EOS/STOP/截断；不更新策略，Baseline 不制造伪训练 Checkpoint。当前 CLI 不自动保存每条解码 WAV，模型通用评测和 WAV 导出仍待补。

Reward 改善不能代替独立质量 Benchmark。不同模式权重/标度不同，Reward 不直接跨模式横比。`text_match` 是 Token 严格匹配，不是开放语义评分；`asr_match=max(0,1-CER)` 不是音质/自然度指标。

## 五、完整 Text / Vision / Audio Benchmark

本节是 **待实现入口的验收协议**，不提供虚构的 `benchmark.py`、`eval_b1.yaml` 或未知 CLI。实现时复用现有模型/Formatter/Codec/评测函数，新增薄只读入口；不得建立另一训练器。C1 的最低集必须在 G4D 之前具备，G7 只复核和扩展。

| 任务 | 固定输入与标签 | 必测指标 | 当前已有 / 待实现 |
| --- | --- | --- | --- |
| Text | 独立可验证文本问题；B0 仅用于文本保留对照 | EM/任务 Accuracy、失败/空回答率、长度 | GRPO Token 匹配已有；通用 prompt-only 生成、规范化/答案提取器和统一汇总待实现。 |
| I2T | 固定原图、问题/Caption；禁止答案进入 prompt | VQA Accuracy 或与任务相符的 Caption 指标、换图/无图对照 | 输入投影和特征调换已有；生成回答及任务评分待实现，不能将 Caption CE 称 Accuracy。 |
| A2T | 独立录音/可信 transcript，16kHz 输入 | CER/WER、空输出/失败率、语言分层 | 数据/模型接口和编辑距离已有；Omni 自身生成转写的批量评测待实现。不要改用 SenseVoice ASR 输出冒充 Omni 输出。 |
| T2A | 固定待朗读文本/可信参考，不输入目标声码 | ASR CER/WER、失败/截断、时长、试听/可选音质代理 | 当前 `demo` 支持小批量预测 WAV/ASR；独立统一协议、人评/音质汇总待补。 |
| QA A2A | 输入语音与独立可信回答文本/目标录音 | 文本正确性、语音内容一致性、CER/WER、完整性及音质 | QA 格式闸门已有；真实问答数据、自由生成批评测待补。echo 诊断单独报告，不能混入 QA 成绩。 |

统一要求：

1. 模型输入只取 prompt 与合法输入媒体，不使用目标文本/音频作为条件。生成参考、评分参考与训练标签分离；沿用已测因果时轴和 Delay。
2. 每样本保存模型 ID/Checkpoint SHA、data ID/group/split SHA、prompt/template 版本、预测文本/可分发音频引用、参考、停止原因与指标原始计数。失败不删除。
3. CER/WER 报告 micro 编辑数/参考长度，同时按语言/长度分层。明确大小写、标点、空格和中文分词规则；当前函数只做空白处理，不能无说明用于标准化跨论文比较。
4. Accuracy 定义先锁定：有多答案任务按指定任务规则，不擅自用 Token exact match 替代。开放生成由可复核评分规则/人评处理，评分器版本和调用成本单列。
5. 音质区分可懂度、内容一致、自然度、噪声和截断。UTMOS 等仅在许可、依赖与版本确定且实际运行后报告；没有人评就写未测，不填写推测 MOS。
6. 空音频、解码失败、超预算、无目标/无评分资源均保留失败原因。指标缺失用 `null + reason`；报告总样本、成功/失败/ASR 覆盖数，不能只挑成功例平均。

## 六、公平比较、消融与统计

最小比较表如下，数值须从原始记录导出，当前不填结果：

| 方法 | 起点 | 训练变量 | 必须同列的成本 | 当前状态 |
| --- | --- | --- | --- | --- |
| B1 Fixed | G4D 验收结果 | 无额外研究训练 | SFT 累计预算、参数、硬件 | 未训练 |
| Fixed 续训控制组 | 同一 B1 | 与 Bridge 候选相同预算 | 额外 updates/数据访问/GPU·h | 未执行 |
| Last/Weighted/Gated | 同一 B1 | Bridge 与获批更新范围 | 新增参数、重训练预算、推理成本 | 未执行；最终候选待定 |
| Text/Audio/Joint GRPO | 同一 B1，各自独立分支 | Reward/策略范围 | Rollout、Reward、更新与额外数据成本 | 未执行；正式配方待定 |

固定 Qwen、Tokenizer、Talker/Mimi、训练来源、封存 test、模板、thinking 模式、生成上限/停止规则、推理协议、硬件与 dtype。把架构差异、额外训练和系统开关分成不同表，不把参数量/预算/随机性收益全部归为 Adaptive Bridge。仅推理替换 Bridge 未重训不构成有效研究对照。

GRPO 的训练采样协议与最终公共推理协议分别登记；当前 RL 行为分布不支持 Top-k/Top-p 截断。最终生成若需不同解码方式，应由统一评测入口在所有候选上同样执行并测试，不能混用不同分布结果。

报告配对样本差值、实际种子数和不确定性。多 Caption/同媒体任务按媒体组统计，避免把同一媒体视图当独立重复。种子不足/预算减少时如实标注，不生成假置信区间。正式候选、次数和预算由 B1 后决定；无提升或退化也是有效结果。

## 七、系统性能与按需优化

**当前现成指标不是完整系统 Benchmark。** `benchmark_audio` 的计时只覆盖模型事件流，不含 Mimi/ASR/文件 IO，warmup=0；`evaluate_run` 未提供独立推理 autocast 参数，训练 YAML 的 BF16 不能证明评测也运行 BF16。P5 的 TTFT/TTFA/RTF 脚本是历史 MiniMind 路径，且目前为用户未提交文件，不能称为已经交付的 Qwen 统一测速器。

| 指标 | G7 必须锁定的口径 | 入口状态 |
| --- | --- | --- |
| TTFT | 请求到首个有效非特殊文本 token；输入准备是否计入明确分列 | Qwen 统一计时待实现。 |
| TTFA | 请求到首段可播放的非空 PCM，必须含首次 Mimi 解码；不是首个 code/token | PCM API 已有，统一计时待实现。 |
| RTF | 实测生成耗时 / 实际输出 PCM 秒数，分模型与端到端；空 PCM 不计算 | 待统一导出。 |
| P95 chunk gap | 连续非空 PCM 到达间隔，记录 chunk_frames/尾块；样本不足明确标记 | 待实现。 |
| 吞吐 | text tokens/s、audio frames/s、requests/s；固定 batch/并发、有效长度、warmup/重复数 | 单条流接口已有；不要宣称已有批量服务。 |
| 显存/内存 | CUDA allocated/reserved 峰值、驻留模型、输入长度；CPU RSS 独立测量 | CUDA peak 部分已有，完整协议/CPU RSS 待补。 |

可执行的 GPU 外部观察命令（不会启动模型训练）：

```bash
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used --format=csv -l 1
```

此采样会持续运行，人工结束；只是设备遥测，不能替代 PyTorch 峰值或逐事件 latency。获批实验时将输出存入独立日志。

系统实验顺序：先冻结正确性基准 → 重复测量瓶颈 → 单变量优化 → 相同样本/设备复测。AMP FP32/BF16、KV Cache on/off、Mimi chunk_frames 分开对照，记录 logits/生成/波形容差和质量变化。模型 API 有 Cache 开关，但当前 `generate_audio` 固定 `use_cache=True`，开关矩阵 CLI **待实现**。

Triton 当前没有已交付 kernel 或测速入口；仅在 Profile 证明热点后再开发，验证前向/梯度/精度/端到端收益。未做可标“未执行/无必要”，不阻塞核心项目结论。DDP/FSDP/ZeRO、双卡 Rollout 与训练分离、gradient checkpointing 均不能视作当前可靠功能。G7 不借性能实验重新训练模型或自动扩大 GPU 预算。

## 八、表格、曲线、架构图、Demo 与报告

以下是**目标交付目录规范，非当前脚本自动生成的承诺**：

```text
outputs/g7/<unique-run>/
  environment/                  commit、依赖锁、CUDA/硬件与执行命令
  imported/                     私有模型/数据包及发布方摘要
  logs/                         stdout、stderr、失败与系统遥测
  b1-loss/ b1-audio/             现有 SFT 入口输出
  rl-b1/ rl-final/               单个 RL 模式的同协议 Reward 对照
  g5-summary/                   原 Bridge 矩阵 summary.json/csv
  benchmark/                    待实现：全任务逐样本、汇总与失败清单
  system/                       待实现：逐事件时间戳、AMP/Cache/优化对照
  charts/                       待实现：质量/成本、收敛、延迟分布
  demo/                         待完成：授权音频/图像案例与操作说明
  report/                       最终技术报告、模型/数据卡、限制与索引
```

统一汇总字段至少包含 `run_id/method/seed/checkpoint_sha/source_sha/data_split_sha/task/metric/value/unit/sample_count/failure_count/hardware/dtype/protocol/timing_scope/status`。每个统计值能追到原始 JSON/CSV 和样本记录。当前 R5 的 JSON/CSV、R4 `loss.svg` 可复用，但跨 B1/G5/G6 的读取、统计/作图和失败合并工具 **待实现**。

最终材料：

- 分任务质量表，Fixed/Adaptive 与 SFT/GRPO 的独立对照表，更新参数/训练预算/推理成本表，含负结果与未测项目。
- 训练/验证曲线、质量与成本散点、延迟分布/PCM 间隔；横轴标实际 optimizer updates、token/frame 或 GPU·h，恢复段去重。没有数据不画趋势线。
- 架构图标清冻结/训练模块、2048→768 投影归属、4×768 Talker、8-codebook、16k/24k 音频路径与两个模型 Cache；可用仓库 Markdown Mermaid/SVG，不把规划图标成实测性能图。
- 可复现 Text、I2T、A2T/QA A2A 与 T2A Demo。现有 `demo` 只提供 T2A 离线生成；Qwen 交互/实时 PCM 展示 **待实现**。提供未挑选的成功/失败案例和许可允许的短音频，不能播放目标 WAV 冒充预测。
- 最终技术报告：问题与自主贡献、上游归属、训练数据/模型来源、算法与协议、实验/资源、失败与局限、复现入口。README 区分“实现/本地通过/云端通过/有研究证据”。

## 九、回传、归档与发布

云端在所有产物写入结束后生成摘要，避免边写边复制。下面在仓库根执行，只归档本次唯一目录；接收端需要相同相对布局：

```bash
find "$G7_ROOT" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "$G7_ROOT/SHA256SUMS"
```

Mac 本地另开终端，使用发布时记录的相同相对 `G7_REL`。此例回传完整私有产物；容量/许可不允许时先定义单独的小型公开包，再单独生成其摘要，不能排除大文件后声称完整包校验通过。

```bash
: "${G7_SERVER:?填写SSH主机}"
: "${G7_REMOTE_REPO:?填写云端仓库绝对路径}"
: "${G7_RETURN_DIR:?填写尚不存在的本地私有归档绝对路径}"
G7_REL=outputs/g7/final-001
test ! -e "$G7_RETURN_DIR"
mkdir -p "$G7_RETURN_DIR/$G7_REL"
rsync -av --partial "$G7_SERVER:$G7_REMOTE_REPO/$G7_REL/" "$G7_RETURN_DIR/$G7_REL/"
(cd "$G7_RETURN_DIR" && shasum -a 256 -c "$G7_REL/SHA256SUMS")
```

另存备份，抽查新进程加载/推理后再释放云资源；恢复训练测试仍属于相应训练阶段的授权，G7 不自动更新权重。发布流程由用户最终批准：

1. 核对项目自身许可证与所有权重/媒体/数据许可；原始语音、私有路径、访问令牌和不可再分发资产不进入公开包。
2. 将小型汇总、图表、最终报告、README 和真实配置复制到待审发布路径；为模型/Adapter 提供独立版本与 SHA、基础模型依赖、获取方式。大型 Checkpoint 留私有持久存储或另行获准的模型库。
3. `git status --short`、逐文件 Diff 和白名单暂存，确认历史 P0–P5、未提交用户文件及大资产未混入。当前 G3 移动和 P5 未跟踪材料由用户独立处理。
4. 本地提交后由用户授权 Push；GitHub Release 的 tag 指向实际已公开提交，上传经审查的小包/链接及摘要。当前无自动发布 CLI，不编造 `export/publish` 子命令，不在本轮执行 Push/Release。
5. 用干净 checkout 按公开 README 独立复现许可允许的最小测试和 Demo；下载步骤必须显式，不默认拉全量数据或训练。

## 十、最终验收与失败披露

- [ ] B1 首次 Benchmark 已在 G4D 完成并归档；G7 使用锁定协议复测，没有用 test 选择超参数/Checkpoint。
- [ ] Text/Vision/Audio 全任务结果、原始样本与失败覆盖齐全；QA A2A 证据不足时标未完成，不用 echo 替代。
- [ ] B1/Bridge/GRPO 对照具有共同来源、可追溯数据、预算与硬件；研究未做或无收益如实保留。Reward 不冒充独立质量。
- [ ] TTFT/TTFA/RTF/吞吐/峰值和真实 PCM 边界一致；系统优化逐项标已测、未做或失败，未实现 Triton/多卡不列为成果。
- [ ] Checkpoint/Adapter/数据/源码/环境/配置/评测版本可核验；完整恢复与只读加载区别明确，没有删除身份检查绕过历史隔离问题。
- [ ] 图表、架构图、许可允许的 Demo、最终技术报告、README、摘要和私有备份齐全；公开路径不依赖本机未提交文件。
- [ ] 独立干净 checkout 的测试、轻量复现和人工 Demo 检查有实际记录；自动测试通过不能替代人听音质。

失败实验保留 `failed/interrupted/not_run/deferred` 状态、原因、已消耗预算、最后有效 Checkpoint、原始日志和样本；缺失指标记 `null`。OOM、无有效动作、空音频、ASR 失败、配置/谱系拒绝均不可删样本后填“成功”。明确哪些候选未跑，哪些指标只测 validation/单 seed，哪些优化无收益。

完整项目通过依赖上述核心条件。若部分研究获准延期，可交付如实标注的阶段成果，但不能宣称全部 G5/G6/G7 研究完成。本轮仅补齐手册，停在审计交付处，等待下一步授权。
