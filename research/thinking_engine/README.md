# Native-thinking refinement / 原生思考向量精炼

| File / 文件 | Contents / 内容 |
|---|---|
| `train_math.py` | Pair validation, exact token boundaries, 0.7 thought / 0.3 final weighting / 成对校验、token边界、思考0.7与回答0.3权重 |
| `vector_ops.py` | Full-input injection, chunked full-vocabulary CE, neutral KL, unit-ball projection / 全输入注入、分块全词表CE、中性KL、单位球投影 |
| `refinement.py` | Original contrast extraction and 240-step candidate fitting / 原始对比方向提取与240步候选精炼 |
| `scoring.py` | Exact final digit or A/B parsing, valid-only means and denominators / 最终数字或A/B解析、有效均值与完整分母 |
| `selection.py` | Original validity-gated development layer selection / 原有效率门控下的开发选层 |
| `model_loader.py` | Checkpointing helper / 梯度检查点辅助 |

`${FPSRG_HOME}` defaults to the repository root. The model path is
`${FPSRG_HOME}/models/huihui-six`; the default output is
`${FPSRG_HOME}/outputs/thinking-refinement`. Set `FPSRG_RESEARCH_RUN` to override
the output (relative values resolve against the repository root). Supply the frozen full1500
epoch2 Huihui model, tokenizer, complete paired trajectories and neutral anchors
to `refinement.fit_candidate`. The function retains checkpoint, optimizer and RNG
resume checks. This directory does not contain a model download or fleet launcher.

`${FPSRG_HOME}` 默认指仓库根目录；模型为`models/huihui-six`，默认输出为
`outputs/thinking-refinement`。用`FPSRG_RESEARCH_RUN`覆盖输出目录，相对值仍以仓库根为基准。
将冻结的full1500第二轮灰灰模型、tokenizer、
完整配对轨迹及中性锚点传给 `refinement.fit_candidate`。函数保留检查点、优化器与随机状态
的恢复检查；本目录不包含模型下载或集群启动器。

| Protocol / 协议 | Value / 值 |
|---|---|
| Axes / 维度 | A, C, N, R, M, T |
| Candidate layers / 候选层 | 8,16,24,32,40,48,56,60; zero-based / 零编号 |
| Update / 更新 | Frozen model, vector only; 240 steps / 底座冻结，只优化向量，240步 |
| Initialization / 初始化 | Mean thought activation contrast, 0.1 reference scale / 思考均值激活差，参考尺度0.1 |
| Grid / 格点 | {−1,0,+1}^6; 729 configurations |
| Scoring / 计分 | Complete final answer; 2 mappings × 2 repetitions / 完整最终答案，两映射各两次 |

```python
from scoring import score
case = {'axis': 'A', 'kind': 'Q', 'targets': {'high': '5'}}
score(case, {'complete': True, 'final': '5'})  # 100.0
```
