# AI-CantoTriage

粤语单轮结构化医疗导诊研究项目。项目使用 Qwen3-1.7B 对患者主诉进行一次性导诊，并输出结构化结果。当前工作属于小规模可行性实验，不能作为临床依据，也不等同于正式实验结论。

更新时间：2026-11-01 17:53（UTC+08:00）  
更新者：傅昱

## 当前内容

本仓库现在包含：

- `数据集/LCMDC_约44万条/训练数据_428条/`：已整理并核验的 428 条数据，不包含原始数十万条数据。
- `导诊模型/Qwen3-1.7B-windows-mac-compatible/`：LoRA 融合后的 Qwen3-1.7B，Transformers 格式，备注为 **windows, mac兼容版**。
- `scripts/reassemble_qwen3_windows_mac.py`：Windows/macOS 均可使用的标准 Python 分片重组与 SHA-256 校验脚本。
- `AI-CantoTriage-MLX-LoRA.zip`：原有 MLX LoRA 实验包，已保留，未被新模型覆盖。

仓库使用 Git LFS 管理模型分片。请先安装并初始化 Git LFS，再下载模型文件。

## 数据集

数据来自 LCMDC，当前发布的 428 条版本划分如下：

| 划分 | 数量 |
| --- | ---: |
| 训练集 | 281 |
| 验证集 | 48 |
| 测试集 | 99 |
| 合计 | 428 |

数据集保留 `总`、`输入`、`输出` 三类 JSONL 文件及原有目录结构。数据集来源与授权信息：LCMDC，CC BY 4.0，DOI `10.5281/zenodo.13771008`。

## Windows/macOS 模型

模型目录：`导诊模型/Qwen3-1.7B-windows-mac-compatible/`

这是已经融合 LoRA 权重的完整 Transformers 模型，不是需要另外加载 PEFT adapter 的目录。模型基座为 Qwen/Qwen3-1.7B，基座 revision 为 `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`。融合信息和权重哈希记录在模型目录的 `lora_merge.json` 与 `model_parts_manifest.json` 中。

由于第一权重分片超过常见的单文件上传限制，仓库保存为三个连续分片；第二权重分片保持原名。克隆仓库后，在仓库根目录执行：

```sh
git lfs install
python scripts/reassemble_qwen3_windows_mac.py
```

脚本会逐个检查分片大小和 SHA-256，并生成：

```text
导诊模型/Qwen3-1.7B-windows-mac-compatible/model-00001-of-00002.safetensors
```

生成文件的 SHA-256 应为：

```text
a359ab3fb5c101c6e8db4f146e5bdbf8389f4dac046861b87d52b405a40e9c20
```

生成的完整第一分片是本地运行文件，不作为额外仓库文件提交。重组脚本只使用 Python 标准库，Windows 和 macOS 均可运行。模型推理所需的 PyTorch、Transformers 等依赖应根据所使用系统安装对应版本。

示例加载方式：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = "导诊模型/Qwen3-1.7B-windows-mac-compatible"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype="auto",
    device_map="auto",
)
```

使用导诊任务时，应沿用项目评测脚本中的系统提示词、输入格式、结构化 JSON 解析规则和 `enable_thinking=False` 设置，才能与已有结果进行可比复现。

## LoRA 试微调设置

试微调只用于可行性验证：使用 281 条训练数据完成一轮训练，验证集 48 条，测试集 99 条。主要设置为：

- LoRA rank：8
- LoRA scale：16
- dropout：0
- 注入后 16 层
- learning rate：`1e-4`
- batch size：1
- seed：42
- response-only loss

## 测试集结果

三次结果都在同一批 99 条测试集上进行自动结构化评测。指标含义如下：

- **科室完全命中率**：预测的科室集合与标准科室集合完全一致的比例。
- **任一命中率**：预测结果中至少有一个科室与标准科室相符的比例。
- **department micro-F1**：先把每个科室视为一个二分类标签，汇总全部科室的 TP、FP、FN 后计算 F1；它同时考虑漏报和误报，比只看完全命中更能反映多标签科室推荐的整体表现。

| 实验 | 科室完全命中 | 任一命中 | department micro-F1 |
| --- | ---: | ---: | ---: |
| 微调前历史基线 | 6/99（6.06%） | 8/99（8.08%） | 7.62% |
| Mac MLX LoRA | 76/99（76.77%） | 87/99（87.88%） | 83.41% |
| Windows Transformers 复现 | 76/99（76.77%） | 87/99（87.88%） | 83.41% |

Windows 结果是本地融合模型在 Windows CPU Transformers 后端上的新生成结果；Mac 结果来自原 MLX LoRA 实验。两种后端的数值和采样实现可能不同，因此相同汇总分数不应解释为逐条输出完全一致。原始评测记录和限制说明保存在本地项目的 `试微调/准确率对比.json`。

## 研究限制

- 数据量小且类别分布不均衡，验证集和测试集没有覆盖全部 15 个科室。
- 自动指标只评价结构化科室/行动字段，不评价临床安全性、医学事实完整性或粤语表达质量。
- 基线使用的是已有历史输出，不是同一推理后端重新生成的严格对照。
- 结果只能支持“使用小规模数据对 Qwen 1.7B 做试微调，进行可行性实验”这一结论。
- 融合后的模型可以作为后续训练或工程验证的起点，但不能作为临床依据或正式实验结果。

## 旧版 MLX 实验包

`AI-CantoTriage-MLX-LoRA.zip` 是仓库原有的 macOS Apple Silicon MLX 实验包。本次上传没有覆盖它。需要复现原 MLX 实验时，请从该文件的 GitHub 文件页下载原始文件后再解压，并按压缩包内说明安装依赖。

## 许可与致谢

Qwen 模型文件遵循其随附的 Apache-2.0 许可。LCMDC 数据集遵循 CC BY 4.0；使用数据集时请同时遵守原始数据集的引用和授权要求。
