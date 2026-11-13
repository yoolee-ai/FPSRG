# PersonLab：六维人格与情绪对话

[English](README.md)

## 启动

在仓库根目录执行：

```bash
python -m pip install -r demo/requirements.txt
cp demo/.env.example .env
# 在 .env 设置 FPSRG_HOME、FPSRG_API_URL 和 FPSRG_API_TOKEN。
python -m uvicorn demo.app:app --host 127.0.0.1 --port 8080
```

先启动 [vector_vllm](../modules/vector_vllm/) 的六维 Huihui 服务，加载与目录配套的完整权重、ACNRMT 向量和情绪扩展 bank。浏览器打开 `http://127.0.0.1:8080`。需要远程访问时采用 SSH 转发；公司部署应在外层提供 HTTPS 和身份认证。

## 创建人物

点击“手动中档示例”即可创建不调用外部模型的示例。每位人物也可直接提供六维 `levels`：

```json
{
  "people": [{
    "id": "example", "name": "示例人物",
    "levels": {"A":"medium", "C":"high", "N":"low", "R":"high", "M":"medium", "T":"high"},
    "dialogue": [{"role":"customer", "text":"我想先了解商品的实际条件。"}]
  }]
}
```

不提供 `levels` 时，在 JSON 顶层加入 `"allow_external_extraction": true`，并配置 `DEEPSEEK_API_KEY`。六路提取会把完整输入发送至配置的 DeepSeek 端点，分别返回 A/C/N/R/M/T 的高、中、低倾向及证据。内置八位人物与对话均为合成示例。

## 匹配与对话

1. 读取后端已过滤的六维目录，先最小化高/中/低档位不一致数量，再最小化到 25/50/75 目标的 L1 距离。
2. 保留目录原始分数；目标高/低档位不可达时，按 `FPSRG_SEMANTIC_SUPPORT` 添加对应语义提示。
3. “统一前置对话”使用相同商品开场，人物各自累积最终回答；“无上下文”每次只发送当前问题。
4. 喜悦/愤怒按钮从下一轮起持续生效，按服务端 bank 路由固定 `2×` 情绪向量，不添加情绪提示词。关闭情绪恢复原节点，不清空历史。
5. 页面等待后端完整返回，随后显示模型提供的思考与回答。未正常结束的回答保留供查看，不进入后续上下文。请求失败不自动重发。

## 配置与记录

| 项目 | 设置 |
|---|---|
| API | `GET /catalog`、`POST /v1/chat/completions`；字段见下方 |
| 人格 | 六维 ACNRMT；此页面不用于十维目录 |
| 模型与目录 | 同一 artifact；切换 artifact 后需新建人物 |
| 情绪 | `none / joy / anger`；非空模式固定强度 2；叠加后未重新量表标定 |
| 语义补充 | 默认开启；`FPSRG_SEMANTIC_SUPPORT=false` 可关闭；补充后分数不重标 |
| 历史隔离 | 浏览器会话 × 人物 × 模式；历史思考不回传 |
| 数据目录 | `$FPSRG_HOME/runtime/personlab/sessions/`；包含本次新输入、请求、响应与历史，勿提交 Git |
| 凭据 | 后端环境变量；不发送至浏览器，不写入审计记录 |

后端请求：

```json
{
  "node_id": 42,
  "messages": [{"role":"user","content":"你会怎么选择？"}],
  "emotion": "joy",
  "temperature": 0.7,
  "max_tokens": 4096
}
```

适配层核对响应 `grid.node_id`、`metadata.artifact_id`、`metadata.emotion`、`metadata.emotion_strength`。仅兼容这一扩展协议的服务能够实际应用向量；普通 OpenAI 接口不能替代向量后端。

## CPU 检查

```bash
python -m unittest demo.test_app -v
node demo/test_frontend.mjs
```

这些检查使用内存 HTTP 模拟，不加载模型、不调用外部 API。原始页面布局、人物切换、情绪按钮与上下文隔离逻辑保留；发布后端改为独立 API 适配层，不包含旧服务器网关、进程管理或真实人物记录。
