# Joint measurement and coverage / 联合测量与覆盖

| File / 文件 | Contents / 内容 |
|---|---|
| `measurement.py` | Frozen 640-probe ten-axis compiler and scoring / 十维640探测题编译与计分 |
| `joint_engine.py` | Complete-sequence boundary state reuse / 完整序列边界状态复用 |
| `persistent_engine.py` | Retained shallow caches / 浅层缓存持续复用 |
| `accelerated_engine.py` | Packed batching and MLP compaction / 合批与MLP压紧 |
| `batch_prefix_ops.py`, `fused_mlp_ops.py` | Exact causal-prefix MLP operators / 因果前缀MLP算子 |
| `coverage.py` | Same-node constrained coverage and retrieval error / 同节点约束覆盖与检索误差 |

```python
import numpy as np
from coverage import measure
scores = np.array([[50.]*5, [70.]*5])
targets = np.array([[50.]*5, [75.]*5])
gate = np.array([True, True])
result, witnesses = measure(scores, targets, gate, limit=60)
```

`measure` retains the paper criterion: every dimension differs by at most 20,
and total L1 error is at most `limit`, satisfied by the same node. Supply 60 for
five dimensions or 120 for ten. It reports raw and gated candidates separately.
Inputs must contain at least one gated node. Array indices in the witness output
are row indices; map them to the supplied artifact's explicit node IDs.

`measure` 保留论文口径：逐维误差不超过20，总L1不超过`limit`，且由同一个节点满足。
五维使用60，十维使用120；分别输出未筛选与筛选目录。至少有一个合格节点。
见证记录中的索引是数组行号，使用时按资产提供的显式node ID映射。

GPU measurement uses the OFF backend with both `grid` and `off_engine` on the
Python import path. The caller supplies jobs, vectors, layers and frozen numeric
configuration, runs its `validate()` admission, then measures chunks. The original
640-probe shape and strict numeric checks are retained. For other instruments,
create a distinct protocol rather than changing a published catalog in place.

GPU测量需同时把 `grid` 和 `off_engine` 放入Python导入路径；调用方提供题目、向量、层与
冻结数值配置，先运行`validate()`再分块测量。这里保留640题形状及原数值检查。
其他评价体系应使用独立协议，不覆盖已有目录。
