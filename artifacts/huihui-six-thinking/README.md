# Huihui six-dimensional thinking catalog

[中文](README.zh-CN.md)

| Item | Value |
|---|---|
| Axes | A agreeableness; C conscientiousness; N neuroticism; R risk taking; M material values; T future consequences |
| Coefficients | `{-1,0,1}` |
| Zero-based layers | A24, C24, N32, R40, M24, T24 |
| Measurement | 24 questionnaire + 8 behavior items per axis, 2 mappings × 2 repeats |
| Filtering | At least126/128 valid outputs per axis; literal user-word hit rate≤10%; seven serving exclusions |
| Counts | 729 measured; 663 format-qualified; 657 serving nodes |
| Emotion | none / joy / anger; fixed strength2; joy layer16, anger layer24 |

`vectors.npz` contains the six personality directions; `profiles.json` maps coefficients to bank rows. `base_bank.npz` and `emotion_vectors.npz` are compact components. `python scripts/prepare_assets.py` recreates the exact 2,187-row runtime `bank.npz`, which is ignored by Git. Component and output checksums are in `bank_components.json`.

Emotion offsets are 0,729,1458. Apply them to `persona_id`, not `node_id`. Catalog scores describe the original personality configuration, before the emotion overlay.
