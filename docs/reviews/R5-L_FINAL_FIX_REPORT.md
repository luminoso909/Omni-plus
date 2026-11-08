## 一、最终结论

**PASS**。2026-11-08，基于 `6cd49888413cf5778586764b2fe8d70134883483`，完成最新复审中两个剩余 P1 的最终修复与验收。此前审查和修复记录保留原文，本表记录当前问题状态。

| 问题 | 状态 | 闭环证据 |
| --- | --- | --- |
| P1-01 B1→G5 数据隔离 | 已关闭 | 旧提交接受遗漏别名，修后完整身份匹配时拒绝跨 split，身份缺失时也拒绝。源、目标及祖先均覆盖，合法同 split 和独立扩充通过。 |
| P1-02 执行源码指纹 | 已关闭 | 旧提交接受另一 checkout 的实际导入，修后两 CLI 和运行身份 API 拒绝错根。同根归档的规划、训练、恢复与评测通过。 |
| P2-01 成本汇总 | 保持通过 | 成本实现未改，JSON/CSV 字段、恢复区间、CPU 空值及四条历史记录复验通过。 |

R5-L 的本地代码、数学和训练工程已验收，可在用户授权后进入 R6-L。正式 B1、G3–G7、CUDA 训练与 Bridge 性能收益仍未验收。

## 二、根因与最小修复

**数据身份**。原流程完整分组后只保存预算选中的行，未选中别名只留不可逆的组摘要。审计契约升级至 v3，`identity_groups` 记录选中组在完整源清单中的身份全集，包括未选行和跨任务别名。检查全集排序、去重、摘要与 `group_id` 相等，以及行身份属于该全集，再执行已有跨阶段分组与角色检查。B1、目标和递归祖先均使用相同规则。

正式数据缺少全集时拒绝认证。旧缓存和 Checkpoint 不回填、不重签，严格恢复继续使用原提交。正式 B1 尚未成立，应以新目录准备完整身份缓存并从开始绑定。已有 Tiny 旧数据可保留诊断用途。

**源码身份**。原指纹扫描脚本所在目录，但 Python 可从 editable 安装指向的另一 checkout 导入。新增 `check_source_root()` 核对已加载 `omni` 模块的 `__file__`、`__spec__.origin` 以及各包 `__path__`，两 CLI 在读配置前调用，`run_identity()` 也调用。静态指纹函数仍支持离线比较归档。源码身份升至 v3，利用既有 `collect` 检查拒绝旧 v2 正式矩阵，防止旧漏洞产生的矩阵继续获得可比性认证。矩阵容器仍是 v2，成本字段与算法未变。

## 三、反例与正向实测

| 检查 | 修复前 `6cd4988` | 最终修复后 |
| --- | --- | --- |
| B1 的 A/B 两份 WAV 共用原始 ID，选样丢 B，G5 Test 改 ID 使用 B 的相同字节 | `read_bridge_base` 与 `plan` 均 ACCEPTED | 缺全集时拒绝，完整全集下两个入口均报 train→test 身份冲突，未创建输出目录 |
| 合法同 split、独立新媒体 | 已有基础覆盖 | 实际 `inspect_sources`、正式 `plan` 和热启动通过，共享权重逐 Tensor 恢复 |
| 目标侧遗漏别名、未纳入任务预算的别名 | 缺少覆盖 | 真实 JSONL 重新审计后拒绝跨 split |
| 复用主工作区 editable 环境运行另一归档的 CLI | `plan` exit 0，实际 `model_omni.py` SHA `ebb1c1d95385…`，记录 `d1459a58e1ca…` | `plan`、`evaluate` exit 1，报告实际与预期根，均无输出目录 |
| 正确来源归档与相同源码 | 已有基础覆盖 | 规划→smoke 1 步→新进程 resume 至 2 步→evaluate 全部 exit 0，身份版本均为 3 |
| 源码不变或发生变化 | 原同根修改测试已有 | 同根正常恢复，Legacy 生成、Collator、Config 改动继续拒绝旧 Checkpoint；错根子模块、缺少 origin、外部包搜索路径和旧矩阵认证均拒绝 |

数据反例使用合法合成 PCM WAV、真实 JSONL 解析与预算选样。正式入口测试中的资产、模型和编码为 fixture，没有把它们记为真实 1.7B 训练。正确来源的实际 CLI 另使用已有许可 Tiny 缓存，冻结参数摘要相同，Bridge/Talker/音频 Embedding/Heads 梯度非零，恢复段为 `(1, 2]`，留出 Loss 为 `21.694904327392578`。这些数值仅证明工程闭环。

本机复现材料保留在 `/private/tmp/omni-r5-final-data-review/` 的 `before.log`、`after_original.log`、`after_complete.log`，以及 `/private/tmp/omni-r5-final-source-yedmjown/` 的 `before-source-mismatch.json`、`probe.json` 和 `v3/after-results.json`。后者记录逐条实际 CLI 命令和退出码。永久回归用例随本次提交保存。

## 四、全仓验收

Mac arm64，Python 3.12.14、PyTorch 2.15.0、Transformers 4.57.6，CPU FP32。本轮普通 pytest、实际 CLI 与静态检查均在本地执行。

```bash
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
export MPLCONFIGDIR=/private/tmp/omni-r5-final/mpl
.venv/bin/python -m pytest -q -p no:cacheprovider --basetemp=/private/tmp/omni-r5-final/pytest
.venv/bin/ruff check --no-cache omni tests scripts
.venv/bin/ruff format --check --no-cache omni tests scripts
.venv/bin/python -m pip check
git diff --check
```

| 验收 | 结果 |
| --- | --- |
| 当前工作区全仓 | **319 passed, 1 skipped in 50.81s** |
| `git archive 6cd4988` 加本次 10 个源码/测试文件 | **312 passed, 1 skipped in 47.68s**，不依赖用户未提交代码 |
| 数据隔离、真实审计管线和消融定向 | **46 passed** |
| 源码身份、R5 训练和消融定向 | **37 passed** |
| Ruff / format | 全通过，当前工作区 78 个 Python 文件格式通过 |
| 依赖 / diff | 通过，无损坏依赖 |
| 历史 `bridge_ablation.py collect` | 四条记录 passed，训练/推理成本与原始 Manifest/Benchmark 一致，CPU JSON 为 null、CSV 留空，`comparable=false` |

两个跳过项均为同一个 MPS 环境测试，在当前沙箱不可用。新旧 Bridge、SFT、冻结梯度、Checkpoint/Resume、Cache 和 Streaming 全部包含在回归内。干净归档使用 `PYTHONPATH=/private/tmp/omni-r5-final/commit` 和原 `.git` 的只读 `GIT_DIR`，两个运行日志为 `/private/tmp/omni-r5-final/{pytest,clean-pytest}.log`。远程 CI 与 CUDA 未运行。

## 五、文件与保护范围

| 文件 | 修改 |
| --- | --- |
| `omni/data/sft_sources.py`、`omni/data/bridge_splits.py` | 保存完整组身份，拒绝不完整正式证据 |
| `omni/training/run_source.py`、`omni/training/r3_local.py` | 实际导入根校验，源码身份 v3 |
| `scripts/run_experiment.py`、`scripts/bridge_ablation.py` | CLI 启动校验，成本逻辑不变 |
| `tests/test_r5_bridge_splits.py`、`tests/test_r5_run_source.py` | 扩充祖先与缺证据检查，原指纹拒绝测试改为真实归档子进程 |
| `tests/test_r5_data_identity.py`、`tests/test_r5_source_origin.py` | 新增真实管线反例、合法正例、跨 checkout 和旧认证拒绝 |
| `AGENTS.md`、`README.md`、主规划、待办、本地云端手册、G5 Runbook | 同步最终状态、运行根和历史证据迁移要求 |
| `docs/stages/R5-local.md`、本报告 | 授权追加 R5 最终验收，记录问题闭环 |

开工记录了 204 个既有文件的字节摘要或不存在状态。提交仅包含本轮白名单。P0–P5、R0–R4-L 与已记录 SHA-256 一致，R5 阶段报告仅追加本轮授权的末节。原审查、复审和前次修复记录保留，所有用户原有未提交改动排除在本次提交之外。旧实验、权重和 Checkpoint 未覆盖，686 个资产/实验文件的大小和修改时间核对不变。

## 六、剩余边界

审计隔离覆盖已登记的原始 ID、字节摘要、文本身份和传递别名。未登记且字节不同的转码关系仍需数据审计补证，不能把内容哈希称为语义去重。运行期间保持源码文件固定，依赖版本继续由既有运行身份约束。正式 B1、可信 QA A2A、CUDA/BF16 训练及消融效果仍依赖后续云端验收。当前没有阻断 R6-L 本地开发的剩余 P1，等待用户授权后续阶段。
