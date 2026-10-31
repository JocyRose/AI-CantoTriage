# AI-CantoTriage

Download **[AI-CantoTriage-MLX-LoRA.zip](AI-CantoTriage-MLX-LoRA.zip)** using its file page → **Download raw file**, then extract it. The ZIP preserves the complete scripts/runtime/adapter directory structure used below.

LoRA fine-tuning of Qwen3-1.7B for Cantonese department recommendation across 15 departments. This repository contains the selected **MLX-format LoRA adapter**, scripts and aggregate pilot metrics.

## Requirements

- macOS Apple Silicon, Python 3.13. The adapter is MLX format, not a Transformers/PEFT adapter.
- Install dependencies in a virtual environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r runtime/requirements.lock.txt
```

## Base model

Download [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B/tree/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e) into `导诊模型/` in this repository. Revision: `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. The base model is distributed by Qwen under Apache-2.0; retrieve its license with the model.

## Load the selected adapter for independent inference

```python
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

model, tokenizer = load(
    "导诊模型",
    adapter_path="试微调/LoRA试验/adapter",
)
# For comparable evaluation, use the exact system prompt and user input format
# in scripts/evaluate_base_model_test.py (also saved in 运行配置.json).
messages = [
    {"role": "system", "content": "<system_prompt from 运行配置.json>"},
    {"role": "user", "content": "<patient input in the documented format>"},
]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
print(generate(model, tokenizer, prompt=prompt, max_tokens=1024,
               sampler=make_sampler(temp=0), verbose=False))
```

## Pilot experiment

281 training cases (one full epoch), 48 validation cases, 99 test cases. Ordinary LoRA, last 16 layers, rank 8, MLX scale 16, learning rate 1e-4, batch size 1, seed 42, response-only loss. The selected checkpoint is after 279 updates, with validation loss 0.8017; all 281 training steps were completed.

| Metric | Historical baseline | LoRA |
| --- | ---: | ---: |
| Exact department-set match | 6.06% | 76.77% |
| At least one correct department | 8.08% | 87.88% |
| Department micro-F1 | 0.0762 | 0.8341 |

Historical baseline and this experiment used different inference backends; this is not a fresh same-backend control. Small, imbalanced pilot results do not establish clinical safety or Cantonese fluency.

## Reproduce with the separately supplied original data

The dataset, patient examples, per-case outputs, base weights and bundled binary dependencies are excluded from this repository. Keep the original `LCMDC_训练数据_428条/` directory and split structure unchanged. Source hashes are in 运行配置.json.

```sh
python scripts/train_trial_lora.py --check
```

`--evaluate-only` requires all original train/validation/test files and `试微调/对照组结果.json` locally. It resumes existing saved test outputs, so use a fresh copy without prior per-case outputs for an independent rerun. Training refuses to overwrite the included adapter; archive the experiment directory first before training again. The legacy evaluation script supplies the prompt; its original Windows-era paths/dependency setup are not the recommended entry point.
