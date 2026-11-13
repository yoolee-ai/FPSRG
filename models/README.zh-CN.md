# 模型权重

[English](README.md)

本目录提供各格点对应模型版本的文件校验清单，权重与代码分开存储。

如需我们训练的灰灰权重或本项目使用的其他模型权重，请邮件联系 **[prm21@mails.tsinghua.edu.cn](mailto:prm21@mails.tsinghua.edu.cn)**。权重后续也会通过本 GitHub 仓库发布。

| 目录 | 内容 | 约占空间 |
|---|---|---:|
| `qwen35-base/` | Qwen3.5-27B BF16 底座 | 52 GiB |
| `persononly-lora/` | FP32 第一人称 LoRA 与提交凭据 | 446 MiB |
| `huihui-six/` | 灰灰 full1500 第二轮完整模型 | 51 GiB |

## 从公司存储安装

```bash
python scripts/install_weights.py --name huihui-six --source /data/model-store/huihui-six
# 十维非思考服务使用以下两份：
python scripts/install_weights.py --name qwen35-base --source /data/model-store/qwen35-base
python scripts/install_weights.py --name persononly-lora --source /data/model-store/persononly-lora
```

脚本逐文件校验大小和 SHA-256；已有相同文件自动跳过；不同文件不会覆盖。加 `--verify-only` 只检查、不复制。

资源配置使用 `${FPSRG_HOME}/models/...`，也可以把模型存储挂载到这些目录。Git 仓库当前提供校验清单与安装工具，配套权重请通过上述邮箱获取。

## 版本

- 十维 LoRA 提交：`cb2f77e43b96830130d9ffe2ec93a630aae4ad8892d46a96668317c0603ea0fe`。
- 六维完整模型提交：`31046f6da875da4054f844712683c39197fd68f95fd5cbff94e70362f49f9f2c`。
- 六维训练：1500 条第一人称数据、全参数两轮；`qwen3_5` 架构，64 层，隐藏宽度 5120。

模型卡名称为 Huihui Qwen3.8-27B-abliterated，运行架构为 `qwen3_5`。部署按文件校验清单匹配。

分发六维模型时随附 `Huihui-LICENSE.txt`。上游模型卡标识为 `huihui-ai/Huihui-Qwen3.8-27B-abliterated`，版本前缀 `739e3c5`；第二轮导出权重使用上面的提交哈希识别。

[GitHub 文件限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github) · [Hugging Face 私有仓库](https://huggingface.co/docs/hub/repositories-settings)
