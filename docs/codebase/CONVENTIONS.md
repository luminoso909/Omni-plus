## 一、命名与张量接口

| 对象 | 约定 | 例子 |
| --- | --- | --- |
| 文件、变量、函数 | `snake_case`，内部辅助方法使用 `_` 前缀 | `model_thinker.py`、`past_key_values`、`_inject_features()` |
| 类和类型 | `PascalCase`，对外接口提供类型与形状标注 | `OmniPlusOmni`、`AudioCodes`、`ThinkerOutput` |
| 常量与环境变量 | 大写加下划线 | `REPO_ID`、`REVISION`、`HF_HUB_OFFLINE` |

文本序列 `[B,T]`、音频码本 `[B,C,T]`、模态特征 `[N,D]`。模态特征列表保持原始 batch 顺序，缺失值用 `None`。Cache 前向的 mask 覆盖历史和当前输入，`OmniDataset` 输出的文本与音频标签已 shift 一次，`sft_loss()` 直接读取对齐后的标签。

## 二、格式与静态检查

Ruff 配置位于 `pyproject.toml`，Python 目标 `py312`，行长 100，启用 `E`、`F`、`I`、`UP`、`B`。Thinker、Omni、Talker、Encoder 四个模型文件对 jaxtyping 形状字符串定向忽略 `F722`。

```bash
ruff check omni tests scripts
ruff format --check omni tests scripts
```

## 三、导入与模块导出

- 按标准库、第三方库、项目模块分组。包内模型模块使用相对导入，数据与脚本常通过 `omni.*` 引用公共接口。
- `omni.models` 显式声明 `__all__`。HF `AutoModelForCausalLM` 注册到 `OmniPlusForCausalLM`，多模态模型通过 `OmniPlusOmni` 显式构造。
- 可选媒体库在相关入口导入，例如 SenseVoice 构造时导入 `funasr`，波形文件读取时导入 `soundfile`，减少基础文本路径的依赖。

## 四、错误与日志

- 模型和数据层用 `ValueError` 提示配置、形状、标签和对齐错误。缺失本地资产、输出路径已存在等情形由调用入口报告，CLI 参数错误可由 `argparse` 处理。
- 检查脚本使用 `assert` 或 `torch.testing.assert_close` 核对数值。加载历史权重与训练 checkpoint 使用 `strict=True`，恢复时比较配置、冻结参数和采样相关数据身份。
- 脚本向终端打印进度，并保存 JSON、CSV 或曲线。训练按 `run_id` 组织，checkpoint 带模型/优化器/scaler/RNG/进度，报告保留路径、配置及哈希。
- 版本库保留小型汇总，原始媒体、权重、缓存和 `.env` 由 `.gitignore` 排除。新增日志应以项目的资产和凭据保护约定为准。

## 五、测试习惯

- 测试放在 `tests/test_*.py`，使用 pytest fixture、`tmp_path`、`monkeypatch`，张量结果使用 `torch.testing.assert_close`。
- 普通测试使用合成数据、小模型及替身。真实权重、媒体和设备实测由显式脚本组织，并在阶段报告记录条件与结果。
- 当前 pytest 配置未设置覆盖率阈值。已有阶段报告记录通过数、运行条件和专项数值结果。

## 六、证据

- [工程约定](../../AGENTS.md)、[Ruff 与 pytest 配置](../../pyproject.toml)
- [联合模型形状与注入](../../omni/models/model_omni.py)、[数据与标签](../../omni/data/omni_dataset.py)、[损失](../../omni/training/losses.py)
- [恢复实现](../../omni/training/sft.py)、[训练报告输出](../../scripts/train_sft.py)、[编码器测试](../../tests/test_encoders.py)
