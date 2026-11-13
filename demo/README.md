# PersonLab: six-axis personality and emotion chat

[中文](README.zh-CN.md)

## Start

From the repository root:

```bash
python -m pip install -r demo/requirements.txt
cp demo/.env.example .env
# Configure FPSRG_HOME, FPSRG_API_URL and FPSRG_API_TOKEN in .env.
python -m uvicorn demo.app:app --host 127.0.0.1 --port 8080
```

Start the six-axis Huihui [vector_vllm](../modules/vector_vllm/) service first, with the matching full weights, ACNRMT catalog and emotion-expanded bank. Open `http://127.0.0.1:8080`. Use SSH forwarding for remote access; add HTTPS and authentication at the gateway for company deployment.

## Create people

The **手动中档示例** button loads a manual, all-medium example without an external model call. Supply manual levels for every individual:

```json
{
  "people": [{
    "id": "example", "name": "Example person",
    "levels": {"A":"medium", "C":"high", "N":"low", "R":"high", "M":"medium", "T":"high"},
    "dialogue": [{"role":"customer", "text":"I want to check the actual product terms first."}]
  }]
}
```

To infer levels, omit `levels`, add `"allow_external_extraction": true` at the root, and configure `DEEPSEEK_API_KEY`. Six calls send the full input to the configured DeepSeek endpoint, one for each axis. All eight bundled people and their conversations are synthetic.

## Matching and interaction

1. Read the backend's filtered catalog. Minimize level mismatches first, then L1 distance to the 25/50/75 anchors.
2. Preserve measured catalog scores. `FPSRG_SEMANTIC_SUPPORT` adds the original semantic description when a requested high/low level is not reached.
3. **Shared context** starts every person with the same product conversation and keeps independent final-answer histories. **No context** sends only the current question.
4. Joy and anger apply the fixed `2×` emotion-bank overlay to subsequent turns, without an emotion prompt. Turning the overlay off restores the original node and keeps history.
5. The release adapter waits for a complete backend response before displaying reasoning and answer. Incomplete outputs are displayed but excluded from subsequent context. Failed calls are not retried automatically.

## Configuration and storage

| Setting | Value |
|---|---|
| API | `GET /catalog`, `POST /v1/chat/completions` |
| Personality | Six-axis ACNRMT; this page is not a ten-axis frontend |
| Model/catalog | One matching artifact; recreate people after changing it |
| Emotion | `none / joy / anger`; fixed strength 2; overlay scores are not recalibrated |
| Semantic support | Enabled by default; disable with `FPSRG_SEMANTIC_SUPPORT=false`; scores remain original measurements |
| Isolation | Browser session × individual × context mode; past reasoning is never sent back |
| Local records | `$FPSRG_HOME/runtime/personlab/sessions/`; new inputs, requests, responses, histories; keep out of Git |
| Credentials | Server environment only; never sent to the browser or written to audit records |

Inference payload:

```json
{
  "node_id": 42,
  "messages": [{"role":"user","content":"What would you choose?"}],
  "emotion": "joy",
  "temperature": 0.7,
  "max_tokens": 4096
}
```

The adapter checks `grid.node_id`, `metadata.artifact_id`, `metadata.emotion`, and `metadata.emotion_strength` in the response. The upstream must implement this vector extension; an ordinary OpenAI server does not apply the vectors.

## CPU checks

```bash
python -m unittest demo.test_app -v
node demo/test_frontend.mjs
```

Tests use an in-memory HTTP transport; no model loading or external API calls. The original page layout, person switching, emotion buttons and context isolation are retained. This release replaces the old filesystem worker backend with an API adapter and excludes old gateways, process-control scripts and real user records.
