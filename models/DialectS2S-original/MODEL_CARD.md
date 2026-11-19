# DialectS2S 原始备份

**Built with glm-4**。

这是 2026-11-19 对照测试实际加载的独立原始备份，不含本项目 LoRA 更新。checkpoint 约 22.05 GB，配套 GLM decoder 约 0.527 GB，文件未修改。

完整文件按[清单](model_manifest.json)打包为同仓库 Release 分卷。使用[下载恢复工具](../download_models.py)下载全部分卷、校验及恢复，详见[模型说明](../README.md)。

来源为 `CASIA-LM/DialectS2S@938dc520d99941b9cf8634b8161823adf1a5694e`，上游模型卡标为 MIT；decoder 来源为 `THUDM/glm-4-voice-decoder@81ed541f0350d0b81728681afb628acee62772c2`，其[独立许可证](GLM-4-VOICE-LICENSE.txt)完整保留。

本轮原模型文字参考科室标签命中 14/28，为 Agent 核对，非人工临床复核。原生语音已生成并完成独立 ASR，尚未真人试听；正式语音及联合准确率未评定。
