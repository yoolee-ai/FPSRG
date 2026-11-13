# FPSRG 向量 vLLM 模块

[English](README.md)

一张卡只加载一份模型，每个请求独立指定人格节点和提示词。支持发布的十维
非思考格点，以及六维灰灰思考格点和喜悦／愤怒叠加。

## 安装与启动

需要 Linux、Python 3.11、一张空闲的80 GB NVIDIA卡和兼容CUDA 12.8的驱动。
模型文件按项目根目录的模型说明下载。

```bash
export FPSRG_HOME=/data/fpsrg
python scripts/prepare_assets.py
bash modules/vector_vllm/install_runtime.sh
bash modules/vector_vllm/serve.sh \
  --manifest artifacts/huihui-six-thinking/manifest.json --gpu 0 --port 8000
```

十维非思考使用 `artifacts/qwen35-ten-off/manifest.json`。启动器遇到占卡进程
会退出，不会结束其他任务；`GET /health` 的 `ready=true` 表示加载完成。
非回环监听须设置 `FPSRG_API_TOKEN` 并指定 `--bind`。可用 `FPSRG_RUNTIME_DIR`
指定依赖安装位置，默认写入本模块的 `.runtime`。

| 设置 | 发布配置 |
| --- | --- |
| Python／PyTorch | 3.11／2.10.0（CUDA 12.8） |
| vLLM／Transformers | 0.19.0／5.15.1 |
| 张量并行／CUDA Graph | TP1／关闭，使用 eager |
| 分块预填充／前缀缓存 | 2048 tokens／按人格隔离 |
| 默认同时处理请求数 | 16，`--max-active` 可设1–64 |
| 上下文 | 由格点清单指定，最高32,768 tokens |

vLLM安装元数据要求 Transformers `<5`，因此安装器将5.15.1放在独立overlay，
由 `serve.sh` 前置加载。十维使用独立FP32 LoRA分支，不合并；灰灰加载完整
第二轮权重，不加载LoRA。两者保留部署过的HF舍入边界，每层向量在FP32中相加，
然后只转换一次BF16。

## 调用

`GET /catalog` 返回合格节点、实测分数和维度名。请求可指定 `node_id`，也可
提供完整的 `targets` 分数，由服务按L1距离匹配。超出实测范围时选最近节点，
模块不会自动加极端人格提示词。每条请求可使用不同的system、历史、随机种子、
温度和输出上限。

```json
{
  "node_id": 42,
  "messages": [{"role": "user", "content": "你现在最在意什么？"}],
  "emotion": "joy",
  "max_tokens": 4096,
  "temperature": 0.7
}
```

发送到 `POST /v1/chat/completions`。回答与思考分别在
`choices[0].message.content` 和 `reasoning_content`，实际节点在 `grid`。
42只是示例，请从当前目录选择合格ID。思考模式随格点固定；本最小API不提供
流式输出。

不同人格和提示词可以放在同批：发送到 `POST /v1/chat/completions/batch`。

```json
{
  "defaults": {"max_tokens": 2048},
  "groups": [
    {"name": "A", "node_id": 42, "requests": [
      {"messages": [{"role": "user", "content": "问题一"}]},
      {"messages": [{"role": "user", "content": "问题二"}]},
      {"messages": [{"role": "user", "content": "问题三"}]}
    ]},
    {"name": "B", "node_id": 43, "n": 5,
     "messages": [{"role": "user", "content": "另一段提示词"}]}
  ]
}
```

`n` 表示独立采样次数；若B的五条提示词也不同，同样使用 `requests` 数组。
最多64条排队请求，实际使用连续批处理；64条上限不表示64个32k上下文能够
同时装入显存。

## 情绪与缓存隔离

六维格点支持 `none`、`joy`、`anger`，喜悦和愤怒固定2倍，路由到各自的向量库
行。叠加后的性格分数未重新标定，响应会记录实际情绪和库行；十维不接受情绪
叠加。缓存标识绑定模型、LoRA、向量库、协议和实际人格行，不跨人格或情绪复用
注入后的KV缓存。逐token路由在vLLM完成请求重排后构造，同时覆盖输入和解码。

## 文件与CPU测试

| 文件 | 内容 |
| --- | --- |
| `artifact.py` | 清单、SHA校验、格点匹配 |
| `model.py` | 两种后端和逐层向量库 |
| `precision.py` | FP32 LoRA及HF舍入适配 |
| `routing.py` | CPU逐token路由参考 |
| `server.py` | 单卡API、连续批处理 |
| `tests/test_cpu.py` | 路由、缓存、混批、完整性测试 |

```bash
python -m pip install -e modules/vector_vllm
python -m unittest discover -s modules/vector_vllm/tests -v
```

本次抽离通过CPU测试，未重新占用GPU验证。已部署实现的来源见
[PROVENANCE.md](PROVENANCE.md)，文件协议见[SCHEMA.md](SCHEMA.md)。
