## 一、Astra Ultra 审查范围

审查实现提交 `046e9470fc23b843ef93aa6161a949cc014ea79a`，父提交 `f3b593f687d2c6986a41b2acdd5bd8b9e7a4728d`。本交接文件单独提交，审查代码以实现 SHA 为准。

范围为四种 Bridge、Qwen 取层、SFT 接线、同一 Fixed 权重来源迁移、严格 Resume，以及独立矩阵和 Benchmark 导出。G5 正式训练、CUDA 与独立音质验收仍待执行。

## 二、关键代码

| 路径 | 审查点 |
| --- | --- |
| `omni/models/bridge.py:11` | 校验类型、零基选层和数量，默认 Fixed 路径 |
| `omni/models/bridge.py:35` | 公共投影、RMSNorm 和 mask，Weighted/Gated 融合 |
| `omni/models/thinker/qwen_backend.py:22` | 请求层 hook 的生命周期、顺序、当前序列窗和梯度 |
| `omni/models/model_omni_v2.py:83` | 条件送入既有 Talker，媒体特征、双 Cache 和掩码 |
| `omni/training/r3_local.py:414` | Fixed 热启动、资产/模型身份、共享张量的严格匹配 |
| `omni/training/r3_local.py:560` | 源码、配置、数据和初始化身份 |
| `omni/training/r3_local.py:724` | 原 SFT 循环、参数组、梯度日志和完整恢复 |
| `omni/training/r3_local.py:964`、`omni/evaluation/bridge_benchmark.py:70` | 留出集、逐样本生成、时间/显存与失败记录 |
| `scripts/bridge_ablation.py:29`、`scripts/bridge_ablation.py:177` | 候选计划、独立目录、身份与终点 Checkpoint 汇总校验 |

## 三、数学与验证锚点

以 $H_i\in\mathbb{R}^{B\times T\times D}$ 表示原始 post-block 状态。Fixed 用原设定的层，Last 用原生 Block 27。Weighted 用 $\alpha=\operatorname{softmax}(a)$ 与 $F=\sum_i\alpha_iH_i$。Gated 用 $g=\sigma(w^\top[H_1;H_2]+b)$ 与 $F=gH_1+(1-g)H_2$。零初始化产生均匀权重或 0.5 门控。所有候选只执行一次 $\operatorname{RMSNorm}(FW^\top)$，用 `[B,T]` mask 清零补齐，输出 `[B,T,768]`。原生 Bridge 参数分别为 1,573,632、1,573,632、1,573,634 和 1,577,729，其中 Weighted 取 2 层。

- 层号对照应区分末 block 的原始值与 HF tuple 的最终 Norm 输出。CPU FP32 的真实 1.7B 对照使用 19 tokens，Block 13、末 block、Norm 后值及 logits 的最大误差均为 0。
- 数学单测以独立公式比较 Weighted 和 Gated 的输出、融合参数与输入梯度。FP32/BF16 shape、mask、参数量及冻结 Qwen 路径有覆盖。
- 共享 Fixed 热启动从实际 R4 Tiny Checkpoint 导入 134 个相同 Tensor。四个候选各跑 2 步 T2A，另跑 Weighted 的两步 Joint。连续与跨进程恢复完整对象一致，含 395 个 Tensor 叶节点。跨类型恢复和缺失 B1 的原生矩阵按预期失败。

## 四、需要重点复核的风险

1. HF 升级可能改变 block 输出与最终 Norm 语义。当前已验证 Transformers 4.57.6，版本变更后应复跑选层、梯度和 Cache 对照。
2. 多层保留增加激活和带宽，Gated 还构造 `[B,T,2D]` 输入。H100 BF16 的真实峰值、吞吐及混合精度行为待 G3/G5 验证。
3. B1 热启动要求 Checkpoint 与来源 manifest 的 SHA 一致。正式 B1 若选择非终点 `best`，需单独冻结匹配该 Checkpoint 的发布清单。热启动继承全部共享权重并重建 optimizer。只有同候选的严格 Resume 才继承 optimizer/调度/游标。
4. `collect` 的可比较标志表示身份和协议校验通过。每组仅 1 个留出样本，各生成 16 帧、预算截断，SenseVoice 返回空转写，CER/WER 均为 1.0。该诊断无法给出候选排序。模型流时延排除 Mimi、ASR 和写盘，CPU 无显存测量值。
5. 原生选层和模态输出通过本地机制验收。四组公平效果仍需基于冻结 B1、相同训练预算、独立样本与人工试听。G5 的选层、运行种子数和评价指标由正式方案确定。

## 五、复验命令与证据

```bash
source .venv/bin/activate
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
python -m pytest -q tests/test_r5_bridge.py tests/test_r5_training.py tests/test_r5_ablation.py
MPLCONFIGDIR=/private/tmp/omni-r5-mpl python -m pytest -q
ruff check omni tests scripts
ruff format --check omni tests scripts
python -m pip check
git diff --check
```

48 个新增用例纳入全仓回归。主机完整回归 **270 passed**，干净归档 **262 passed, 1 skipped**，跳过项为沙箱 MPS。Ruff、72 文件格式和依赖检查通过。

- [R5-L 阶段报告与 Tiny 复验命令](docs/stages/R5-local.md)
- [G5 操作手册](docs/cloud/G5_BRIDGE_RUNBOOK.md)
- [实测命令、参数和指标 JSON](experiments/r5-local/20261108-cpu-fp32/acceptance.json)
- [矩阵摘要 CSV](experiments/r5-local/20261108-cpu-fp32/summary.csv)
- [独立归档与历史指纹](experiments/r5-local/20261108-cpu-fp32/verification.json)
