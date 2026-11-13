# Models

[中文](README.zh-CN.md)

This directory contains file hashes for the exact model snapshots used by the released catalogs. Weights are stored separately from the code.

For the trained Huihui weights or other model weights used in this project, email **[prm21@mails.tsinghua.edu.cn](mailto:prm21@mails.tsinghua.edu.cn)**. The weights will also be released through this GitHub repository later.

| Directory | Contents | Approximate size |
|---|---|---:|
| `qwen35-base/` | Qwen3.5-27B BF16 base | 52 GiB |
| `persononly-lora/` | FP32 first-person LoRA and commit receipt | 446 MiB |
| `huihui-six/` | Huihui full1500 epoch-2 model | 51 GiB |

## Install from a mounted model store

```bash
python scripts/install_weights.py --name huihui-six --source /data/model-store/huihui-six
# For the ten-dimensional non-thinking service:
python scripts/install_weights.py --name qwen35-base --source /data/model-store/qwen35-base
python scripts/install_weights.py --name persononly-lora --source /data/model-store/persononly-lora
```

The installer verifies file sizes and SHA-256, resumes matching files, and refuses to replace a different existing file. To check a mounted directory without copying, add `--verify-only`.

Each artifact manifest uses `${FPSRG_HOME}/models/...`; a local mount at this location works as well. The Git repository currently contains the verification manifests and installation tools; request the matching weights using the email above.

## Versions

- Ten-axis LoRA commit: `cb2f77e43b96830130d9ffe2ec93a630aae4ad8892d46a96668317c0603ea0fe`.
- Six-axis full model commit: `31046f6da875da4054f844712683c39197fd68f95fd5cbff94e70362f49f9f2c`.
- Six-axis training: 1,500 first-person records, two full-parameter epochs; `qwen3_5` architecture, 64 layers, hidden size 5,120.

Model-card label: Huihui Qwen3.8-27B-abliterated. Runtime architecture: `qwen3_5`. Exact files, rather than the display label, determine compatibility.

Keep `Huihui-LICENSE.txt` with any copy of the six-axis model. Its upstream model-card identifier is `huihui-ai/Huihui-Qwen3.8-27B-abliterated`, revision prefix `739e3c5`; the full epoch-2 export is identified by the commit above.

[GitHub file limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github) · [Private Hugging Face repositories](https://huggingface.co/docs/hub/repositories-settings)
