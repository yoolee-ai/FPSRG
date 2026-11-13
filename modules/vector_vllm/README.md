# FPSRG vector vLLM

[中文](README.zh-CN.md)

One model per GPU, with a separate vector configuration and prompt for every
request. Supports the released ten-axis non-thinking catalog and six-axis
Huihui thinking catalog, including joy/anger overlays.

## Install and start

Linux, Python 3.11, an available 80 GB NVIDIA GPU and a CUDA 12.8-compatible
driver. Download model files following the repository's model instructions.

```bash
export FPSRG_HOME=/data/fpsrg
python scripts/prepare_assets.py
bash modules/vector_vllm/install_runtime.sh
bash modules/vector_vllm/serve.sh \
  --manifest artifacts/huihui-six-thinking/manifest.json --gpu 0 --port 8000
```

Use `artifacts/qwen35-ten-off/manifest.json` for the ten-axis non-thinking model.
The launcher refuses an occupied GPU. `GET /health` becomes `ready=true` after
loading. For a non-loopback address, set `FPSRG_API_TOKEN` and pass `--bind`.
`FPSRG_RUNTIME_DIR` overrides the module's local `.runtime` directory.

| Setting | Released value |
| --- | --- |
| Python / PyTorch | 3.11 / 2.10.0 (CUDA 12.8) |
| vLLM / Transformers | 0.19.0 / 5.15.1 |
| Tensor parallelism / CUDA Graphs | TP1 / disabled (eager) |
| Chunked prefill / prefix cache | 2048 tokens / persona-isolated |
| Default active requests | 16; `--max-active` accepts 1–64 |
| Context | Artifact-defined, up to 32,768 tokens |

The installer places Transformers 5.15.1 in a separate overlay because vLLM's
package metadata declares Transformers `<5`. `serve.sh` activates that overlay.
FP32 LoRA branches remain unmerged in the ten-axis backend. Huihui loads the
full epoch-2 model with no adapter. Both use the deployed HF-compatible rounding
boundaries and one BF16 conversion after each layer's FP32 vector sum.

## Requests

`GET /catalog` returns eligible node IDs, measured scores and axis names. Select
one `node_id`, or supply every dimension in `targets` for nearest-L1 matching.
Scores outside the measured range select the nearest node; this module does
not silently add extreme-personality prompts. Each request may use a different
system message, conversation, seed, temperature and output budget.

```json
{
  "node_id": 42,
  "messages": [{"role": "user", "content": "What matters most to you in this decision?"}],
  "emotion": "joy",
  "max_tokens": 4096,
  "temperature": 0.7
}
```

Send to `POST /v1/chat/completions`. Read `choices[0].message.content` and
`reasoning_content`, with the selected configuration in `grid`. Node 42 is an
example; choose an eligible ID from the loaded catalog. The thinking setting is
fixed by the artifact. Streaming is not exposed by this minimal API.

Send mixed-personality batches to `POST /v1/chat/completions/batch`:

```json
{
  "defaults": {"max_tokens": 2048},
  "groups": [
    {"name": "A", "node_id": 42, "requests": [
      {"messages": [{"role": "user", "content": "Prompt one"}]},
      {"messages": [{"role": "user", "content": "Prompt two"}]},
      {"messages": [{"role": "user", "content": "Prompt three"}]}
    ]},
    {"name": "B", "node_id": 43, "n": 5,
     "messages": [{"role": "user", "content": "A different prompt"}]}
  ]
}
```

The API accepts at most 64 queued requests. `n` creates independent sampled
requests. Use per-item `requests` to give all five B requests different prompts.
The engine continuously batches prefill and decoding; a 64-request limit does
not imply that 64 full-length 32k conversations fit in memory.

## Emotion and cache isolation

Six-axis artifacts can map `none`, `joy`, and `anger` to separate runtime bank
rows. Joy and anger use the released strength 2.0. Their added effect has not
been recalibrated into the personality scores. The response records the actual
runtime row and emotion. Ten-axis artifacts reject emotion requests.

Cache salts include the model, adapter, vector-bank identity, protocol and
runtime persona row. Different persona/emotion configurations never reuse
each other's post-intervention KV cache. Routing is constructed **after** vLLM
reorders requests and applies to all scheduled prefill/decode tokens.

## Files and CPU tests

```text
src/fpsrg_vector_vllm/artifact.py   Manifest, SHA checks and catalog matching
src/fpsrg_vector_vllm/model.py      Two backends and layer-local vector banks
src/fpsrg_vector_vllm/precision.py  FP32 LoRA and HF-compatible rounding
src/fpsrg_vector_vllm/routing.py    Pure CPU token-routing reference
src/fpsrg_vector_vllm/server.py     Single-GPU API and continuous batching
tests/test_cpu.py                 Routing, cache, mixed-batch and integrity tests
```

```bash
python -m pip install -e modules/vector_vllm
python -m unittest discover -s modules/vector_vllm/tests -v
```

The publication extraction passed CPU tests. This extracted package has not
been revalidated on a GPU; the original deployed implementations are listed in
[PROVENANCE.md](PROVENANCE.md). [SCHEMA.md](SCHEMA.md) specifies artifact files.
