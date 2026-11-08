## 一、修复范围

2026-11-08，基于 R5-L 交接提交 `a51b560a731d67efa136ed7fd79082044218a6f8`，完成独立审查报告的 P1-01、P1-02 和 P2-01。当前状态为本地修复已验证，待用户复审。正式 G5 仍待 B1 与云端验证。

| 审查项 | 修复与证据 |
| --- | --- |
| P1-01 跨阶段隔离 | 规划和热启动共用来源校验。正式 G5 用 B1 与继承组件 Checkpoint 中的摘要核验各自原始缓存清单，按媒体 ID、内容摘要、文本身份及传递分组检查 train/validation/test。测试覆盖跨任务同媒体、改 ID 同内容、继承组件冲突、封存 Test、无重叠扩充和缺失或被篡改的来源证据。热启动拒绝时模型张量逐项相同，规划拒绝时结果目录尚未创建。 |
| P1-02 执行源码身份 | 源码身份 v2 绑定整个 `omni` Python 包及两个实验 CLI。矩阵 v2、Checkpoint、训练 Manifest 和评测记录共用该身份，汇总同时比较 PyTorch 和 Transformers 版本。临时修改 Legacy 生成、Collator 或 Config 后，恢复和评测在构造模型前拒绝。汇总测试覆盖源码不同、身份字段缺失和运行时不同。 |
| P2-01 成本汇总 | JSON 和 CSV 分别导出 `training_*` 和 `inference_*`。训练记录包含循环耗时、每步耗时、allocated/reserved 峰值及恢复段 `(start_step, end_step]` 和更新数。推理记录包含模型流时延与 allocated 峰值。CPU 显存在 JSON 为 `null`，CSV 留空。 |

正式 G5 的 `model.initialization.bridge_base_lineage` 指向以 Checkpoint SHA-256 为键的 JSON。B1 条目登记原始缓存清单，继承组件条目登记 Checkpoint 及其原始缓存清单。相对路径以谱系文件所在目录为准。身份覆盖审计数据中实际保存的媒体和文本标识。历史 Tiny 矩阵可以生成诊断汇总，`comparable=false`。历史 Checkpoint 在原提交执行严格恢复。 [G5 操作手册](../cloud/G5_BRIDGE_RUNBOOK.md)

## 二、文件清单

| 路径 | 内容 |
| --- | --- |
| `omni/data/bridge_splits.py` | 校验来源谱系并按已知身份合并分组 |
| `omni/training/run_source.py` | 计算与 checkout 路径无关的源码摘要 |
| `omni/training/r3_local.py` | 共用热启动检查，记录和验证新身份 |
| `scripts/bridge_ablation.py` | 在规划时校验来源，汇总身份及训练和推理成本 |
| `tests/test_r5_bridge_splits.py`、`tests/test_r5_run_source.py` | 谱系、数据角色及源码变更的拒绝回归 |
| `tests/test_r5_training.py`、`tests/test_r5_ablation.py` | 实际规划与热启动入口、历史兼容、结果身份和成本字段回归 |
| `docs/cloud/G5_BRIDGE_RUNBOOK.md` | 正式 G5 配置、谱系清单和汇总字段 |

## 三、测试与保护核验

Mac arm64，Python 3.12.14、PyTorch 2.15.0、Transformers 4.57.6。本轮使用 CPU 和本地 fixture，CUDA 成本字段由构造记录验证。旧 Tiny 矩阵另外做只读复验，四条历史记录均能导出成本且汇总为诊断。

```bash
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
export MPLCONFIGDIR=/private/tmp/omni-r5-fix/mpl
.venv/bin/python -m pytest -q -p no:cacheprovider --basetemp=/private/tmp/omni-r5-fix/pytest-final
.venv/bin/ruff check --no-cache omni tests scripts
.venv/bin/ruff format --check --no-cache omni tests scripts
.venv/bin/python -m pip check
git diff --check
```

| 检查 | 实际结果 |
| --- | --- |
| 当前工作区全仓 pytest | **290 passed, 1 skipped**，37.57 秒 |
| HEAD 归档叠加修复源码及测试 | **283 passed, 1 skipped**，37.56 秒。只读 Git 元数据用于测试的 `git rev-parse HEAD` |
| Ruff、格式、依赖、diff | 全部通过，76 个 Python 文件无需格式调整 |
| 历史 Tiny `bridge_ablation.py collect` | 4 行通过，训练与推理字段分别导出，`comparable=false` |

两次全仓复验的跳过项均为沙箱 MPS 不可用。首次全仓检查发现新集成测试预期英文 `split`，实际异常为中文“留出集冲突”。更正匹配后定向测试和两次最终全仓检查通过。

逐项比对开工前指纹，除 5 个本次编辑的既有文件外，其余 193 个既有文件的内容或不存在状态均一致。P0–P5 与 R0–R4-L 的 SHA-256 和 R5 验收证据相符，R5-L 历史报告、独立审查报告与用户未提交文件也通过字节校验。原始 Tiny 矩阵只读，复验输出写入 `/private/tmp/omni-r5-fix/legacy-summary/`。
