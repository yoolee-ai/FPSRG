# First-person post-training / 第一人称后训练

| File / 文件 | Protocol / 协议 |
|---|---|
| `lora_objective.py` | 1,000 examples, one full pass; full native thought/final/EOS CE, 125 length-grouped batches / 1000条一次覆盖，原生思考、最终回答、EOS完整CE，125个长度分组批次 |
| `full_sft_objective.py` | 1,500 examples × 2 epochs, 16 examples per global batch, 188 updates / 1500条两轮，全局批量16，188步 |

These are the original data-validation, scheduling and chunked vocabulary-loss
kernels. The caller supplies tokenized records with `input_ids`, `labels`,
`sequence_tokens` and target boundaries, plus the model/optimizer/distributed
training lifecycle. Prompt labels are −100. Full-SFT tail placeholders have zero
weight. No training data, optimizer state or GPU-launch script is bundled here.

这里保留原数据校验、排程与分块词表损失。调用方提供含`input_ids`、`labels`、
`sequence_tokens`及目标边界的编码记录，以及模型、优化器和分布式训练流程。
提示位置标签为−100；全参数训练尾批占位样本权重为零。不附训练数据、优化器状态或占卡脚本。
