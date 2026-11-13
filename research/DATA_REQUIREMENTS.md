# Inputs and provenance / 输入与来源

| Component / 模块 | Required input / 需要的输入 | Distribution / 分发 |
|---|---|---|
| OFF refinement / 非思考精炼 | Five-axis `catalog.json` (30 train, 6 dev, 24 test per axis); `behavior.json` (16/4/8 paired scenes per axis) | Prepared research items; no user conversation logs / 准备好的研究题目，不含用户对话 |
| OFF grid / 非思考格点 | 24 questionnaire × 2 mappings + 8 behavior × 2 orders per dimension | Compiled jobs; 640 probes for ten axes / 十维640条编译输入 |
| ON refinement / 思考精炼 | Valid paired high/low native token trajectories, exact target boundaries, neutral anchors | Generated training demonstrations delivered separately / 生成的训练示范独立交付 |
| ON calibration / 思考标定 | Six-axis formal questionnaire/behavior jobs, two mappings and repeats | Complete output parsing, not first-token substitution / 完整输出解析，不替换为首token |
| Coverage / 覆盖 | Node score matrix, IDs, eligibility mask, fixed targets | Catalog artifacts; no text transcripts required / 目录资产，无需对话原文 |

## Source and license checklist / 来源与许可

- Qwen3.5 and Huihui base model: retain the exact upstream model card, revision and
  Apache-2.0 license when redistributing derivative weights. The full1500 exported
  checkpoint needs these upstream notices added to its separate model package.
- IPIP questionnaire source items are public-domain material; retain source URLs
  and item/scoring keys. Authored Chinese translations and new research scenarios
  must retain their authorship/provenance and the company's chosen data license.
- The additional R/T/V/M/Se items are authored research items informed by classic
  definitions, not certified translations of validated scales.
- PyTorch, Transformers, PEFT, safetensors, flash-linear-attention and vLLM are
  external dependencies. Install from their official distributions; do not bundle
  a private environment or strip their license notices.

- 重新分发Qwen3.5及灰灰派生权重时，附上准确的上游model card、版本和Apache-2.0许可。
  full1500导出的权重目录需在独立模型包中补齐这些上游文件。
- IPIP源题属于公有领域，保留来源链接和计分键。中文编写／翻译与新增场景保留作者来源，
  由公司明确数据许可。
- R/T/V/M/Se新增题是参考经典定义编写的研究题，不宣称为经过验证的量表翻译。
- 运行库通过官方发行安装，不复制私人Python环境，不删除上游许可。

This export includes code, not full training trajectories, personal conversations,
API credentials, optimizer dumps or cluster run logs.

本目录交付代码，不包含完整训练轨迹、个人对话、API凭据、优化器转储或集群运行日志。
