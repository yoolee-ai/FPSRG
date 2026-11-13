# Non-thinking discovery and refinement / 非思考方向发现与精炼

| File / 文件 | Function / 功能 |
|---|---|
| `historical_engine.py`, `engine.py` | Frozen-model forward, last-position contrast, reference scale, digit scores / 冻结前向、末位置对比、尺度与数字分数 |
| `ka_engine.py` | Signed questionnaire/behavior objective, 240-step optimization and checkpoints / 正负问卷与行为目标、240步优化与检查点 |
| `full_sequence.py` | Explicit nonpadding input and output injection / 非填充输入及输出全程注入 |
| `selection.py` | Development-only two-family layer ranking / 开发集双类指标选层 |
| `model_loader.py`, `models.py` | BF16 base plus frozen FP32 LoRA / BF16底座与冻结FP32 LoRA |
| `protocol.py`, `catalog.py` | Item compilation and selected five-axis profile / 题目编译与五维配置 |
| `stable_mass.py` | Unconditional valid-label probability / 合法标签的非条件概率 |

## Configure / 配置

Copy `config.example.json` and set `FPSRG_RESEARCH_CONFIG` to its location.
`${FPSRG_HOME}` resolves to the repository root unless explicitly set. Other
relative paths resolve against the configuration file. Select `axes=OCEAN` for the five-axis
profile or `axes=RTVMS` for the additional five dimensions. These use separate
prepared data directories and shared model identity.

复制 `config.example.json`，设置 `FPSRG_RESEARCH_CONFIG` 指向该文件。
`${FPSRG_HOME}` 默认解析到仓库根目录，也可显式设置；其他相对路径以配置文件为基准。
`OCEAN` 为五维配置，`RTVMS` 为新增五维；分别准备对应数据目录。

```python
# Run in research/off_engine after configuring weights and frozen data.
# 配置权重、冻结数据后，在 research/off_engine 中运行。
from ka_engine import KnowledgeActionEngine
from study import candidates
engine = KnowledgeActionEngine(gpu=0)
engine.preflight()
engine.fit_candidate(candidates()[0])
```

The engine uses the GPU exposed as `cuda:0`; select it with `CUDA_VISIBLE_DEVICES`.
The caller schedules remaining candidates. No SSH, process eviction or automatic
GPU reservation is included. The prepared run requires `data/catalog.json`,
`data/behavior.json`, the matching adapter and its commit. The loader verifies
adapter hashes while omitting the optimizer state from inference-only packages.

引擎使用 `cuda:0`，由 `CUDA_VISIBLE_DEVICES` 指定实际卡。调用方调度其他候选，
不包含 SSH、停止其他进程或自动占卡。运行目录需要 `data/catalog.json`、
`data/behavior.json`、对应适配器及 commit；仅推理交付可不附优化器状态。

## Objective / 目标

The retained loop uses questionnaire choice loss (0.5), complete self-description
loss (0.1), signed complete behavior loss (0.5), neutral KL every fourth step (0.1),
and unit-vector L2 penalty (0.01). Baseline-relative log-odds targets cycle through
−1, −0.5, +0.5, +1. Full-sequence injection leaves padding unchanged.

原循环保留问卷选择损失0.5、完整自陈损失0.1、正负完整行为损失0.5、每四步中性KL 0.1，
以及单位向量L2惩罚0.01。目标以基线log-odds平移定义，系数轮换−1、−0.5、+0.5、+1。
全序列注入不改变padding。
