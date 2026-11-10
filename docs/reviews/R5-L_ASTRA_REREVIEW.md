## 一、复审结论

**PASS WITH REQUIRED FIXES**

2026-11-08 独立复审提交 `6cd49888413cf5778586764b2fe8d70134883483`。逐项核对原审查报告、修复 Diff、正式规划与评测入口，并在该提交的干净归档运行回归。P2-01 已闭环。两个 P1 的常规路径通过，但仍有可复现的边界缺口，正式 G5 比较须先补齐。

| 审查项 | 本轮结果 | 证据 |
| --- | --- | --- |
| P1-01 跨阶段数据隔离 | 部分闭环 | 已保留的媒体、文本身份和祖先记录会参与隔离。选样丢失的已审计别名仍能进入 G5 Test。 |
| P1-02 执行源码指纹 | 部分闭环 | 同一 checkout 中改动生成、Collator 或 Config 会拒绝恢复和评测。CLI 从另一 checkout 导入源码时仍可能记录当前 checkout 的哈希。 |
| P2-01 成本汇总 | 通过 | JSON 与 CSV 的训练、推理字段对应原始记录，恢复区间和 CPU 空值均正确。 |

## 二、P1-01 数据隔离复验

`omni/data/bridge_splits.py:12–58` 用 Checkpoint 记录的 SHA-256 校验 B1 和祖先组件的缓存清单。`omni/training/r3_local.py:448–464` 与 `scripts/bridge_ablation.py:80–81` 使实际热启动和规划共用此检查。现有测试验证了直接媒体重合、跨任务或换 ID 后的相同字节、纯文本身份、已记录的传递连接、封存 Test、完整性校验和独立扩充。独立运行谱系与训练定向测试为 **22 passed**。

### 1 剩余 P1：去重选样丢失原审计组的别名证据

**位置** `omni/data/sft_sources.py:88,104–110,145`、`omni/data/r3_prepare.py:140–144`、`omni/data/bridge_splits.py:78–100`。

`read_sources` 在完整 JSONL 上按身份关联记录。组 ID 是全部成员身份的摘要，`inspect_sources` 随后只把选中记录放入缓存契约。每条记录的 `media_ids` 仅包含本行身份。跨阶段检查能比较两组的完整摘要是否相等，但当成员集合变化时，无法从旧摘要取回曾记录的成员身份。

本轮构造两条同为 B1 `train` 的合法 T2A 记录，两份 WAV 分别为 8 kHz 和 16 kHz，媒体 ID 都是 `source-id`。实际 `read_sources` 将两条记录归为同一组。预算选样保留 8 kHz 记录，缓存未保存 16 kHz 记录的 SHA。G5 `test` 使用同一份 16 kHz 文件，媒体 ID 改为 `different-id`。它与旧别名的字节摘要相同，和 B1 选中行的 `media_ids` 无交集，两边组 ID 也不同。

在与 Checkpoint SHA 自洽的正式谱系下，实际结果为 `formal_read_bridge_base=ACCEPTED` 和 `formal_plan=ACCEPTED planned_ablation ['g-test-renamed-alias']`。探针使用临时 WAV 和模型、配置、编码 fixture，审计 JSONL 解析、选样、谱系校验和矩阵规划逻辑均运行提交中的代码。该用例验证的是原清单明确记录的同组关系。

**最小修复及复验** 将选中组的完整身份集合绑定到源和目标缓存契约，或用 SHA 锁定的原始审计清单重建该集合。无法提供完整身份的历史缓存应要求重新生成正式比较所需证据。增加真实 JSONL 选样回归，确保未选中别名换 ID 后，规划和热启动都在生成结果或复制权重前拒绝。保留同 split 和无重叠扩充正例。

复现脚本与日志位于 `/private/tmp/omni-r5-recheck-split.PfbZRg/`，文件为 `probe_split_alias.py` 和 `probe_split_alias.log`。

## 三、P1-02 源码身份复验

`omni/training/run_source.py:19–44` 按相对路径计算 `omni` Python 文件及两个实验 CLI 的 SHA-256。`omni/training/r3_local.py:583–605` 把源码和运行时纳入 Checkpoint 身份。`scripts/bridge_ablation.py:106–123,195–212` 核对矩阵、训练和评测记录。原报告涉及的 Legacy 生成、Collator、Config 在同根修改时均会触发严格拒绝。源码、身份字段及运行时不一致的汇总回归也通过。源码与汇总定向测试为 **18 passed**。

### 1 剩余 P1：记录的文件根可能和 Python 实际导入根不同

**位置** `scripts/run_experiment.py:7–19`、`scripts/bridge_ablation.py:10–23`、`omni/training/run_source.py:19–44`。两个 CLI 使用脚本路径确定 `ROOT`，而 Python 根据安装环境和导入路径解析 `omni`。源码指纹扫描 `ROOT/omni`，没有核对已导入包的位置。

从干净归档执行实际 CLI，复用原工作区已有的 editable `.venv` 并移除 `PYTHONPATH`。Python 导入原工作区的 Legacy Omni 与 `r3_local`，而 `plan` 正常退出并将归档的文件摘要写入矩阵。独立探针在两种导入环境下解包实际 `stream_generate` 函数，确认函数源码不同，但完整 `run_identity` 的摘要相同。

| 值 | SHA-256 前缀 |
| --- | --- |
| 矩阵记录的 `model_omni.py`，取自干净归档 | `d1459a58e1ca` |
| 复用环境实际导入的 `model_omni.py`，取自原工作区 | `ebb1c1d95385` |
| 两种导入环境生成的完整 `run_identity` | `7c6be6ad6c9d` |

进一步用归档代码构造一个 Tiny Fixed 随机初始化的 step 0 Checkpoint，再用上述错配环境运行实际 `evaluate`。命令退出 0，写出 `evaluation.json`，`checkpoint_step=0`、总 Loss `22.418333053588867`，仍记录归档摘要。此过程使用现成 Tiny 缓存，评测验证了严格身份检查确实接受错配环境。评测走 Loss 前向，函数源码探针单独验证了 Legacy 生成实现的差异。

**最小修复及复验** 两个 CLI 在读取资产或写入产物前校验实际导入的 `omni` 包根与 `ROOT/omni` 相同，或显式保证导入来自被签名的根。补真实子进程回归，覆盖复用另一 checkout 的 editable 环境时拒绝，以及同根环境中的正常规划、恢复和评测。

CLI 导入日志、矩阵和评测记录位于 `/private/tmp/omni-r5-source-review.BFjuDa/`。独立身份探针与两组 JSON 位于 `/private/tmp/omni-r5-rereview/`。

## 四、P2-01 成本字段复验

`scripts/bridge_ablation.py:151–171,295–305` 将训练 Manifest 的循环耗时、每步耗时、峰值和训练段区间分别映射到 `training_*`，将模型流时延和推理峰值映射到 `inference_*`。恢复用例验证 `(1, 2]` 只计 1 次更新。CUDA fixture 验证训练峰值 allocated/reserved 分别为 1000/2000 字节，推理峰值为 500 字节，JSON 与 CSV 映射一致。

使用干净归档的 `collect` 复读历史 Tiny 矩阵，逐项与原始 Manifest 和 Benchmark 对照。四个运行均导出有效记录，CPU 峰值 JSON 为 `null`、CSV 留空。以 Fixed 为例，2 次更新的训练循环耗时 `0.1380588750 s`，单样本模型流时延 `0.0355569590 s`。旧矩阵维持 `comparable=false` 的诊断标签。训练循环计时含验证及 Checkpoint IO，推理计时仅覆盖模型流。消融与评测定向测试为 **23 passed**。

## 五、回归与工作区核验

验证环境为 Mac arm64、Python 3.12.14、PyTorch 2.15.0、Transformers 4.57.6。复审证据来自 CPU fixture、真实 Tiny CLI 和历史 CPU 产物。CUDA 字段的测试使用 fixture。

| 实际检查 | 结果 |
| --- | --- |
| `git archive 6cd4988` 后运行全仓 `python -m pytest -q -p no:cacheprovider` | **283 passed, 1 skipped in 36.69s**，沙箱 MPS 不可用 |
| `ruff check --no-cache omni tests scripts` 与 `ruff format --check --no-cache omni tests scripts` | 通过，干净归档 73 个文件 |
| `python -m pip check`、`git diff --check` | 通过 |
| 历史 Tiny 实际 CLI `collect` | 四行通过，成本与原始记录一致，`comparable=false` |

归档回归使用 `PYTHONPATH` 指向归档及只读 `GIT_DIR`，设定 `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2`。日志为 `/private/tmp/omni-r5-rereview/pytest.log`。

开工前的 203 个文件状态均通过字节核验，P0–P5 SHA-256 与历史验收证据相符。仓库新增文件仅为本复审报告。复审停在 R5-L，等待用户决定两个剩余 P1 的处理。
