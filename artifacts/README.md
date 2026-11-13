# Released artifacts

[中文](README.zh-CN.md)

| Artifact | Axes | Nodes | Files |
|---|---|---:|---|
| [qwen35-ten-off](qwen35-ten-off/README.md) | O C E A N R T V M S | 59,049 measured; 33,942 served | Vectors, full measurements, gates, serving catalog |
| [huihui-six-thinking](huihui-six-thinking/README.md) | A C N R M T | 729 measured; 657 served | Vectors, complete-response scores, compact bank, emotions |

`manifest.json` selects the model, ordered axes, generation settings and asset hashes. `catalog.json` contains eligible nodes, measured scores, coefficients and explicit bank row IDs. `node_id` is the scientific identifier; `persona_id` is the runtime bank index.

Run `python scripts/verify_assets.py` after checkout. Measurements and vectors keep their original numeric values; paths use the repository layout.
