# Utilities

[中文](README.zh-CN.md)

| Command | Function |
|---|---|
| `python scripts/prepare_assets.py` | Reconstruct the emotion bank from the compact six-axis bank and two emotion vectors |
| `python scripts/verify_assets.py` | Prepare if needed; validate asset hashes, node counts and score ranges |
| `python scripts/install_weights.py --help` | Install or verify weights from a local model store |

Run from the repository root with Python 3.11 and NumPy. None of these commands starts GPU inference or contacts an external service.
