# 已发布资源

[English](README.md)

| 资源 | 维度 | 节点数 | 文件 |
|---|---|---:|---|
| [qwen35-ten-off](qwen35-ten-off/README.zh-CN.md) | O C E A N R T V M S | 实测59,049；可用33,942 | 向量、全量分数、门控、服务目录 |
| [huihui-six-thinking](huihui-six-thinking/README.zh-CN.md) | A C N R M T | 实测729；可用657 | 向量、完整回答分数、压缩库、情绪 |

`manifest.json` 指定模型、维度顺序、生成设置和资源哈希。`catalog.json` 包含通过筛选的节点、实测分数、系数与库行映射。`node_id` 是实验编号，`persona_id` 是运行时库下标。

克隆后运行 `python scripts/verify_assets.py`。分数和向量数值保留原值，路径统一为项目布局。
