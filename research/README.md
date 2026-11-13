# Research kernels / 论文实验内核

| Directory / 目录 | Contents / 内容 | Runtime / 运行条件 |
|---|---|---|
| `post_training` | First-person LoRA and full-SFT objectives / 第一人称LoRA与全参数后训练目标 | Encoded synthetic training data; external training runner |
| `off_engine` | Non-thinking vector discovery and refinement / 非思考方向发现与精炼 | Qwen3.5-27B + persononly125 LoRA; frozen training data |
| `thinking_engine` | Huihui six-axis thought/final losses and refinement / 灰灰六维思考与回答精炼 | Frozen full1500 epoch2 model; paired native traces |
| `grid` | First-token scoring, boundary reuse, coverage / 首token计分、边界复用、覆盖 | NumPy for scoring; GPU backend for model forwards |
| `tests` | CPU contracts / CPU协议测试 | Python 3.11+, NumPy |

```bash
python -m unittest discover -s research/tests -v
```

The algorithms come from the frozen experiments listed in `SOURCE_MANIFEST.json`.
Paths and cluster scheduling were separated from scientific computation. Modules
retain their original flat imports: use each engine in its own Python process.
Model weights, training demonstrations and the cluster orchestration are external.

算法来自 `SOURCE_MANIFEST.json` 记录的冻结实验。计算部分与路径、集群调度分开；
各引擎保留原有平面导入结构，请在独立 Python 进程中使用。权重、训练示范与集群调度
不在此目录。复现需要的输入见 `DATA_REQUIREMENTS.md`。

| Artifact / 资产 | Model binding / 模型绑定 | Measurement / 测量 |
|---|---|---|
| Ten-axis OFF / 十维非思考 | Qwen3.5-27B + independent FP32 persononly125 LoRA | 59,049 nodes; conditional first-token probabilities |
| Six-axis Huihui / 六维灰灰 | Full-parameter 1,500-example, epoch-2 model; no LoRA | 729 nodes; complete native thought and final answers |

Use the supplied catalog only with its matching weights, vectors and injection
scope. A changed prompt, model or numeric mode needs a separately identified run.

目录必须和绑定的权重、向量、注入范围配套。修改提示词、模型或数值模式时，另建实验标识。
