# Third-party components / 第三方组件

| Component / 组件 | Source / 来源 | License / 许可 |
|---|---|---|
| vLLM | https://github.com/vllm-project/vllm | Apache-2.0 |
| Transformers | https://github.com/huggingface/transformers | Apache-2.0 |
| PyTorch | https://github.com/pytorch/pytorch | BSD-style; see upstream LICENSE |
| Qwen3.5-27B | https://huggingface.co/Qwen/Qwen3.5-27B | See the selected weight revision's model card and license |
| Huihui upstream weights | Model identity and file hashes in `models/` | Apache-2.0 in the archived upstream model card |

Dependencies are installed separately rather than vendored. Model weights and the LoRA are distributed separately; retain their original notices when copying them. Research modules record the source implementation for each extracted algorithm.

依赖通过包管理器安装。模型与 LoRA 单独存放，复制时保留上游许可文件。研究模块分别记录算法来源。
