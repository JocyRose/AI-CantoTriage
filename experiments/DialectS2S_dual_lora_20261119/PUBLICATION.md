# DialectS2S 测试对照（2026-11-19）

**Built with glm-4**

两组模型使用相同的 28 条测试样本和 14 科限定指令。训练、验证和测试分别为 42、14 和 28 条，语音样本均由 TTS 合成。两组测试输入均为主诉语音和文字指令。

| 指标（每组 28 条） | 原模型 | 双模块 LoRA |
| --- | ---: | ---: |
| Agent 核对的文字参考科室标签命中 | 14/28（50.00%） | 15/28（53.57%） |
| 严格格式及 14 科原名通过 | 22/28（78.57%） | 28/28（100%） |

部分标签命中回答的理由与主诉不符。56 份语音均完成独立 ASR，自动评分未能提取可计分的推荐格式，语音及联合准确率待真人试听。以上结果适用于本次全 TTS 小样本测试。

- [逐条文字、WAV 和 ASR 对照](evaluation/test_comparison.md)
- [逐条评分 JSON](evaluation/test_comparison.json)
- [对照表 CSV](evaluation/test_comparison.csv)
- [汇总与测试参数](evaluation/score_summary.json)
- [原模型和双 LoRA 下载方法](../../models/README.md)
