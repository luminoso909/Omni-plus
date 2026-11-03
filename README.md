## 一、Omni-plus

基于 MiniMind-O 架构思路，自主复现并开展算法改进的全模态模型学习与研究项目。

**当前仅完成 P0 工程初始化**：Python 包骨架、环境检查、pytest、Ruff 与 CPU CI 配置。尚未实现模型、训练或推理算法，未下载模型权重和数据集，未验证权重兼容性。

### 1 项目目标与架构规划

以下均为后续计划：自主实现 Thinker、Talker、Bridge、Projector 和多模态数据处理；实现 SFT、GRPO、KV Cache 和流式语音推理；兼容 MiniMind Dense/MoE 权重；通过独立 Backbone Adapter 支持 Qwen3；研究 Adaptive Bridge、多模态 GRPO，并进行消融与评测。

```text
文本 / 图像 / 语音
  → Tokenizer / Encoders / Projectors
  → Thinker（MiniMind Dense / MoE；后续独立 Qwen3 Adapter）
  → 文本输出 + Bridge → Talker → 流式语音
```

核心算法由用户学习并实现，Codex 辅助工程、测试、调试与文档。

### 2 环境安装与验收

优先使用 macOS Apple Silicon 原生 arm64 Python 3.12。当前检出目录名为 `Omni++`，包分发名为 `Omni-plus`，导入名为 `omni`。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python --version
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.backends.mps.is_available())"
python -c "import omni; print('Omni-plus OK')"
pytest -q
ruff check .
ruff format --check .
python -m pip check
git status
```

MPS 是否可用由主机环境决定，测试会检查可用性；可用时实际执行 MPS Tensor 运算，不可用时跳过该项。CPU Tensor 测试始终运行。无 CUDA 专用依赖；Ubuntu CI 先从 PyTorch 官方 CPU 索引安装 torch，再安装项目。CI 配置已创建，远程执行结果需以 GitHub Actions 为准。

依赖由 `pyproject.toml` 管理，范围限制主版本以减少后续 API 漂移；Transformers 暂限定 4.x，P0 未验证任何模型 API。实际安装版本与验收证据见 [P0 报告](docs/stages/P0.md)。`.env.example` 仅为可选设置示例，不会自动加载。

### 3 项目目录

```text
configs/{model,training,inference}/  配置占位
omni/models/{thinker,encoders,talker}/  模型包占位
omni/{data,training,inference,evaluation}/  功能包占位
scripts/                           后续命令入口
tests/                             环境基础测试
experiments/                       小规模实验汇总
docs/{learning,stages,reports}/     学习、阶段与研究记录
.github/workflows/ci.yml           Ubuntu CPU 基础检查
```

### 4 当前状态与下一阶段

P0 只交付工程骨架；本地验收及 GitHub 状态见 [P0 报告](docs/stages/P0.md)。P1 建议先确定 MiniMind Thinker 的模块命名、权重键与张量接口，再由用户开始 RMSNorm 等基础模块；需用户另行启动 P1。

### 5 原项目致谢与许可

感谢 Jingyao Gong 的 [MiniMind-O](https://github.com/jingyaogong/minimind-o)。2026-11-03 核实其 [LICENSE](https://github.com/jingyaogong/minimind-o/blob/master/LICENSE) 为 Apache-2.0。P0 未复制其代码或权重；此归属说明不代表原作者背书。

如未来分发含上游材料的派生内容，需随附 Apache-2.0 许可文本、保留相关版权和归属说明、标明修改，并遵循上游 NOTICE（如有）的要求。各模型权重、数据集与外部编码器须分别核实授权。本项目自身许可证尚未指定，未擅自创建 LICENSE。
