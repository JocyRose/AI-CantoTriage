# 本轮发布的两个模型

更新时间：2026-11-19。**Built with glm-4**。

| 模型 | 实际文件 | 用法 |
| --- | --- | --- |
| DialectS2S 原始备份 | checkpoint 约 22.05 GB；波形 decoder 约 0.527 GB | 加载原始 checkpoint，输出语音 token 后用 decoder 转为 WAV |
| glm-4-DialectS2S-CantoTriage-dual-LoRA | 共用原始 checkpoint，另加约 10.9 MB 的两组适配器 | 在同一原模型上同时加载 `llm_lora` 和 `tts_lora`，再用相同 decoder |

完整权重和适配器 ZIP 位于[本次 Release](https://github.com/JocyRose/AI-CantoTriage/releases/tag/dialects2s-dual-lora-20261119)。原模型与 decoder 打成未压缩 tar，切分为每卷最多 512 MiB；需下载全部分卷。适配器也直接保存在本仓库中。微调模型没有合并全量权重，不能仅凭适配器独立推理。

## 下载并校验

在仓库根目录执行，Python 3.12 或更高版本，无需额外 Python 包：

```powershell
python models/download_models.py --output './CantoTriageModels'
```

默认同时恢复两个变体，共用底座只下载一次。`--variant base` 仅处理原模型与 decoder；`--variant lora` 仅处理适配器（推理仍需要底座）；`--download-only` 仅下载校验、不解包；`--archives-dir` 指定分卷保存位置。

脚本逐卷校验 SHA-256，再确认完整 tar 的校验值，然后按清单恢复并逐文件校验。不同内容的已有文件或 `.partial` 会报错并保留。中断的单卷需先检查并移走或删除其 `.partial` 后重试，已校验的完整分卷无需重下。
解包中断后，也需检查并移走或删除输出目录中对应的 `.partial`；无需重新下载完整分卷。

下载文件约 22.6 GB，解包另需约 22.6 GB；请预留至少 46 GB，另外留出模型运行空间。恢复后目录：

```text
./CantoTriageModels/
  DialectS2S/
  glm-4-voice-decoder/
  glm-4-DialectS2S-CantoTriage-dual-LoRA/
    adapters/
      llm_lora/
      tts_lora/
      metadata.json
```

[model_manifest.json](DialectS2S-original/model_manifest.json)记录 39 个原始文件、全部分卷及两组适配器的大小和 SHA-256。原模型文件来自 2026-11-18 独立备份，也是 2026-11-19 测试实际加载的 checkpoint。

本次已用发布脚本从本地全部 43 个分卷和适配器 ZIP 完整恢复，39 个原模型及 decoder 文件、5 个适配器文件逐一通过 SHA-256；[恢复验证记录](DialectS2S-original/RESTORE_VERIFICATION.json)保留具体检查范围。这次验证未重新运行推理，也不代表已在另一台电脑部署。

## 加载方式

DialectS2S 是自定义 `OmniSpeechModel`，不能作为独立普通 Qwen 模型加载。官方源码为 [CASIA-LM/OpenS2S](https://github.com/CASIA-LM/OpenS2S)，固定提交 `230f8d560d456abc4ca80d433996b2f5f9114a3a`。下载脚本只恢复模型文件，不安装环境、不下载源码、不启动模型。

原模型按官方 OpenS2S 的方式加载。双 LoRA 变体在同一底座上用 PEFT 对文字语言模型和语音语言模型分别加载 llm_lora 和 tts_lora。各自配置和 SafeTensors 权重保存在适配器目录中。加载前需按官方说明配置 OpenS2S 源码、依赖及 GLM decoder。

模型生成的语音 token 通过随附 GLM decoder 解码，不能把文字另交 TTS 后冒充原生语音。本轮两组的实际 WAV 已在[实验目录](../experiments/DialectS2S_dual_lora_20261119/evaluation/)中。

## 来源与许可

- 原模型：`CASIA-LM/DialectS2S@938dc520d99941b9cf8634b8161823adf1a5694e`，上游模型卡标为 MIT，原卡随模型保留。
- Decoder：`THUDM/glm-4-voice-decoder@81ed541f0350d0b81728681afb628acee62772c2`，保留独立的 [glm-4-voice 许可证](DialectS2S-original/GLM-4-VOICE-LICENSE.txt)，不重新许可为 MIT。
- 微调模型公开名称为 **glm-4-DialectS2S-CantoTriage-dual-LoRA**。原 checkpoint 和 decoder 未修改，适配器未合并。

本轮是 42 条全 TTS 样本的流程试验。结果分母、核对方式和未评定指标见仓库首页及[逐条对照](../experiments/DialectS2S_dual_lora_20261119/evaluation/test_comparison.md)，未证明临床适用性。
