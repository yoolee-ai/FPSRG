# Provenance / 代码来源

The release extracts the deployed `fpsrg_ten_off_single_gpu` implementation
(2026-10-30) and the `huihui_full1500e2_customer6_grid16` / PersonLab emotion
implementation (2026-11-03/04). Publication changes replace installation-specific
paths and controllers with an external manifest and a single-GPU HTTP entry.

本模块抽离自2026-10-30的十维非思考单卡部署，以及2026-11-03/04的灰灰六维
格点与PersonLab情绪实现。发布时将机器路径和集群调度替换为外部清单与单卡入口。

| Component / 组成 | Retained mechanism / 保留机制 |
| --- | --- |
| `precision.py` | Original FP32 LoRA A→B branches, projection mapping, and HF rounding boundaries / 原FP32 LoRA分支、投影映射和舍入边界 |
| `model.py` | Post-scheduler-reorder routing; residual reconstruction before addition; FP32 layer sums; one BF16 cast / 重排后路由、完整残差注入、逐层FP32相加后一次BF16转换 |
| `server.py` | `LLM.llm_engine.add_request` + `step`, continuous batching, cache salts / 连续批处理和人格隔离缓存 |
| Emotion / 情绪 | Original 729 base rows + joy and anger offsets; strength2 / 原729行加两组情绪偏移，固定2倍 |

CUDA Graphs remain disabled. The adapters target vLLM 0.19.0's private runner
interfaces; updating vLLM requires new routing, numerical and cache tests.
No weights, private conversations, credentials or server process state are
included in this module. CPU tests were run after extraction; GPU regression
has not been repeated for this package.

CUDA Graph保持关闭。模块依赖vLLM 0.19.0的内部runner接口，升级须重测路由、
数值与缓存。模块不包含权重、私人聊天、凭据或服务进程状态；抽离后已做CPU测试，
尚未重复GPU回归。
