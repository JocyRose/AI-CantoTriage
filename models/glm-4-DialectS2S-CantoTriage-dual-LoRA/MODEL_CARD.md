# glm-4-DialectS2S-CantoTriage-dual-LoRA

**Built with glm-4**

这是 DialectS2S 粤语导诊实验的双模块 LoRA 发布目录。显示名是发布标识，不表示把基础模型的文字骨干改称 GLM：文字骨干配置与两组适配器权重保持原样。

本目录提供 `adapters/llm_lora/`（文字模块）、`adapters/tts_lora/`（语音语言模块）及 `adapters/metadata.json`，共 5 个文件。两个 adapter 必须配合相同的原始 DialectS2S base 使用；音频恢复还依赖独立 GLM decoder。两种实验配置是共享 base 的原始模型与 base 加两个 adapter，并非两份独立合并的全量模型。

完整 base 与 decoder 通过[同仓库 Release](https://github.com/JocyRose/AI-CantoTriage/releases/tag/dialects2s-dual-lora-20261119)分卷发布；下载及校验恢复方法见[模型说明](../README.md)。两组适配器同时保留在本目录和 Release ZIP 中。

适配器元数据中的本地路径和私有备份清单字段已移除。逐条测试结果和音频见[实验说明](../../experiments/DialectS2S_dual_lora_20261119/PUBLICATION.md)。

基础模型继续适用其原有许可及上游组件条款：`licenses/base/README.md` 原字节保留本地基础模型的 `license: mit` 声明，这个声明不扩展为 GLM decoder 的许可证。本地基础模型没有单独的 LICENSE 文件，未另行编造许可证文本。GLM decoder 继续适用其独立许可证，原始全文位于 `licenses/glm-4-voice-decoder/LICENSE`，包括其中的署名及使用条件。完整 base/decoder 发布须保留各自已有许可、版权和上游声明；本模型卡不替换或重新授权这些材料。

每组 28 条测试中，文字 Agent 参考科室标签命中为 base **14/28**、LoRA **15/28**；严格格式通过为 base **22/28**、LoRA **28/28**。这不是真人或临床复核。已有标签命中但理由与患者主诉不符的错误，不能将这些计数称为导诊效果提升。

**56 份测试 WAV 和独立 ASR 均已完成；真人试听 0 份。两组正式 audio_acc=null、joint_acc=null。** ASR 是自动估计，不能作为语音正确性、粤语质量、临床有效性或安全性的证明。原始自动评分器的待复核项和 null 均保留。
