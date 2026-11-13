# FPSRG

[English](README.md)

![一个模型，多样的人](model_cards/huihui-27b/assets/one-model-many-people.png)

**构建多样的参与者，让他们共享一份模型。**

构建异质模型配置、测量多维联合响应，并在同一份模型上提供逐请求向量干预。

创建不同偏好的消费者、模型训练伙伴和交互角色；在同一张卡上混批处理不同人格、提示词与历史。[灰灰模型首页](model_cards/huihui-27b/README.zh-CN.md)展示原始对话、情绪控制及应用流程。

## 项目内容

| 目录 | 内容 |
|---|---|
| [research](research/README.zh-CN.md) | 向量发现、精炼、评分与格点分析 |
| [modules/vector_vllm](modules/vector_vllm/README.zh-CN.md) | 独立 vLLM 向量模块，同批次支持不同人格和提示词 |
| [artifacts](artifacts/README.zh-CN.md) | 向量、实测格点与筛选规则 |
| [demo](demo/README.zh-CN.md) | 人格与喜悦／愤怒控制的 PersonLab 页面 |
| [models](models/README.zh-CN.md) | 权重校验清单与存储约定 |
| [model_cards](model_cards/README.zh-CN.md) | 双语模型首页、彩图与原始回答案例 |
| [scripts](scripts/README.zh-CN.md) | 资源校验、权重安装 |

## 两套配置

| 资源 | 模型 | 维度 | 格点数／服务候选数 | 标定方式 |
|---|---|---|---:|---|
| `qwen35-ten-off` | Qwen3.5-27B＋第一人称 LoRA | O C E A N R T V M S | 59,049／33,942 | 非思考，首 token 概率 |
| `huihui-six-thinking` | 灰灰 27B，full1500 第二轮 | A C N R M T | 729／657 | 思考，完整回答 |

六维资源同时带有 demo 使用的 2 倍喜悦、愤怒向量；目录分数对应情绪叠加前的人格配置。

## 开始使用

```bash
git clone https://github.com/yoolee-ai/FPSRG.git
cd FPSRG
export FPSRG_HOME="$PWD"
bash modules/vector_vllm/install_runtime.sh
# 后续 Python 命令使用模块安装的运行环境。
source modules/vector_vllm/.runtime/venv/bin/activate
python scripts/verify_assets.py
```

按[权重说明](models/README.zh-CN.md)安装对应权重，再按[单卡启动与 API 文档](modules/vector_vllm/README.zh-CN.md)启动推理。[Demo 文档](demo/README.zh-CN.md)说明如何连接 API 打开页面。

每条请求分别携带人格和对话。同卡混批时，向量路由、缓存身份与消息上下文按请求隔离。当前后端采用单卡副本、eager 执行、连续批处理和分块预填充。

## 目录约定

```text
FPSRG/
  artifacts/{qwen35-ten-off,huihui-six-thinking}/
  models/{qwen35-base,persononly-lora,huihui-six}/  # 单独安装
  model_cards/huihui-27b/
  modules/vector_vllm/
  research/
  demo/
  scripts/
```

`FPSRG_HOME` 指向项目根目录。权重目录可由公司模型存储挂载；密钥使用环境变量；运行记录不进入 Git。

## 论文

**FPSRG: Heterogeneous User Simulation — First-Person Post-Training and Multidimensional Semantic Response Grids**  
Ruimin Pei、Haibo Liu、Fan Zhang、Xunhua Guo。

本仓库为公司内部交付版本，包含实现及指定实验资源。上游组件见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
