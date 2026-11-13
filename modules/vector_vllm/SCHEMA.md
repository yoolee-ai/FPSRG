# Artifact schema / 格点文件协议

All paths are relative to the manifest, or use `${FPSRG_HOME}`. `sha256` values
refer to the exact published bytes. 配置路径相对清单解析，或使用`${FPSRG_HOME}`；
SHA256校验发布文件的确切字节。

```json
{
  "schema_version": 1,
  "artifact_id": "huihui-six-thinking",
  "backend": "huihui6_full",
  "model": {"path": "${FPSRG_HOME}/models/huihui-six", "fingerprint": "<commit.json SHA256>"},
  "adapter": null,
  "axes": ["A", "C", "N", "R", "M", "T"],
  "layer_by_trait": {"A": 24, "C": 24, "N": 32, "R": 40, "M": 24, "T": 24},
  "assets": {
    "catalog": {"path": "catalog.json", "sha256": "<SHA256>"},
    "bank": {"path": "bank.npz", "sha256": "<SHA256>"}
  },
  "generation": {"thinking": true, "max_model_len": 32768, "max_tokens": 4096},
  "emotion": {"base_rows": 729, "modes": ["none", "joy", "anger"], "strength": 2.0}
}
```

- `catalog.json`: `{"nodes":[{"node_id":42,"persona_id":71,"scores":{"A":...},"coefficients":{"A":...}}]}`. Every axis appears in scores and coefficients. Only gated candidates are included. 节点分数和系数必须包含所有维度，只纳入门控通过的候选。
- `persona_id` is an explicit bank row, never inferred from `node_id`. 它是库行的显式映射，不能由节点编号猜测。
- Huihui `bank.npz`: FP32 `bank[persona, layer, 5120]`, sorted unique int64 `layers`. Row0 is exactly zero. Emotion extends 729 rows to2187, without changing original rows. 灰灰库为FP32，层号从0计数，情绪叠加不改变原729行。
- Ten-axis backend: `backend="ten_off"`, axes `OCEANRTVMS`, `vectors.npz` holds FP32 `vectors[10,5120]`; adapter has `path` and `commit_sha256`. It uses the published `{-1,0,1}` coefficients and layer recipe. 十维使用原始未合并FP32适配器、十根向量及固定层配置。
- Model identity is separate from location. Full Huihui validates the epoch-2 commit; ten-axis validates the LoRA commit and every adapter tensor. 模型标识不依赖部署路径；完整权重和适配器校验方式不同。

`GET /catalog` exposes the sanitized nodes, generation defaults and emotion
configuration, not model filesystem locations. 接口返回节点和默认生成配置，不暴露权重路径。
