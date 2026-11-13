# Qwen ten-dimensional non-thinking catalog

[中文](README.zh-CN.md)

| Item | Value |
|---|---|
| Axes | O openness; C conscientiousness; E extraversion; A agreeableness; N neuroticism; R risk taking; T future consequences; V prosocial allocation; M material values; S sensory sensitivity |
| Coefficients | `{-1,0,1}` on each axis |
| Zero-based layers | `40,32,32,16,32,32,40,32,40,32` |
| Score | Mean of questionnaire and behavioral scores, on 0–100 |
| Injection | All valid input and generated tokens |
| Filtering | Legal answer mass ≥.98 per axis/interface; questionnaire mapping gap ≤25; behavioral order gap ≤.25; raw A/B bias ≤.90 |

`measurements.npz` preserves all 59,049 nodes; `node_gate_metrics.npz` stores their filter evidence. `catalog.json` exposes 33,942 eligible nodes. `vectors.npz` contains ten FP32 directions in the stated order.

The S axis code means sensory sensitivity; the catalog's `scores` field is the questionnaire/behavior composite. Use `questionnaire_scores` and `behavior_scores` for separate analysis.
