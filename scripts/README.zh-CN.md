# 工具脚本

[English](README.md)

| 命令 | 功能 |
|---|---|
| `python scripts/prepare_assets.py` | 由六维基础库与两根情绪向量重建运行向量库 |
| `python scripts/verify_assets.py` | 自动准备并校验资源哈希、节点数量与分数范围 |
| `python scripts/install_weights.py --help` | 从本地模型存储安装或验证权重 |

在项目根目录运行，使用 Python 3.11 和 NumPy。以上命令不启动 GPU 推理，不调用外部服务。
