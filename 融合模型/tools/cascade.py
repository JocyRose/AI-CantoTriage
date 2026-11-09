#!/usr/bin/env python3
"""离线原始模型级联：粤语 ASR -> Qwen3-1.7B 粤语文字回答。

这是依次运行两个原始模型的级联 CLI，非端到端模型，也不代表已完成训练。
依赖从融合目录的 _runtime 优先加载；不依赖 qwen_asr pip 包。

示例（可从任意工作目录启动）：
  python cascade.py --audio sample.wav --instruction "用粤语概括呢段录音"
  python cascade.py --transcript "我今日有啲攰" --instruction "用粤语回应" --device cpu

默认使用 CPU / float32；CUDA 需要当前环境的 PyTorch 支持 CUDA。
--max-new-tokens 控制文字回答，ASR 单独使用 --asr-max-new-tokens。
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import math
import os
from pathlib import Path
import sys


SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 30
ASR_TEXT_TAG = "<asr_text>"
ASR_PREFIX = "language Cantonese" + ASR_TEXT_TAG
SYSTEM_PROMPT = (
    "你係一個用廣東話同用戶交流嘅助手。請按照用戶指令，"
    "根據提供嘅轉寫文字，用自然、口語化嘅粵語同繁體中文字回答。"
    "唔好編造轉寫冇提過嘅事實；資料不足就清楚講明。只輸出回答文字。"
)


def extract_forced_transcript(continuation: str) -> str:
    """Forced Cantonese decoding is plain text; a repeated tag keeps only its tail."""
    text = continuation.strip()
    if ASR_TEXT_TAG in text:
        text = text.rsplit(ASR_TEXT_TAG, 1)[-1].strip()
    return text


def nonempty(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("內容不可為空白")
    return value


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必須為正整數")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio", type=Path, help="本地音頻路徑；不接受 URL；最長 30 秒")
    source.add_argument("--transcript", type=nonempty, help="直接提供真實轉寫文字，跳過 ASR")
    parser.add_argument("--instruction", type=nonempty, required=True, help="用戶文字指令")
    parser.add_argument("--output", type=Path, help="可選：將含 transcript 和 answer 的 JSON 寫入此路徑")
    parser.add_argument("--max-new-tokens", type=positive_int, default=256, help="文字回答最大新增 token 數（默認 256）")
    parser.add_argument("--asr-max-new-tokens", type=positive_int, default=1024, help="ASR 最大新增 token 數（默認 1024）；達上限未結束則報錯")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="運行設備（默認 cpu）")
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16"), default="auto", help="auto：CPU 用 float32，CUDA 支持時用 bfloat16，否則 float32")
    return parser.parse_args(argv)


def configure_runtime(root: Path) -> None:
    # Put the package's parent on sys.path for relative imports in official code.
    sys.path.insert(0, str(root.parent))
    sys.path.insert(0, str(root / "_runtime"))
    # Keep imported official modules read-only, including their bytecode caches.
    sys.dont_write_bytecode = True
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def select_device_dtype(torch, device_name: str, dtype_name: str):
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但目前 PyTorch 無可用 CUDA；CPU 版 PyTorch 請使用 --device cpu。")
    if dtype_name == "auto":
        use_bfloat16 = device_name == "cuda" and torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if use_bfloat16 else torch.float32
    else:
        dtype = getattr(torch, dtype_name)
    if device_name == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("目前 CUDA 設備不支持 bfloat16；請使用 --dtype float32。")
    return torch.device(device_name), dtype


def read_audio(path: Path):
    """Read a local file, reject >30 s before decoding, downmix and resample."""
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    if not path.is_file():
        raise FileNotFoundError(f"音頻必須係存在嘅本地文件（不接受 URL）：{path}")
    with sf.SoundFile(str(path)) as stream:
        rate = stream.samplerate
        if rate <= 0 or stream.frames <= 0:
            raise ValueError(f"音頻為空或採樣率無效：{path}")
        if stream.frames > MAX_AUDIO_SECONDS * rate:
            duration = stream.frames / rate
            raise ValueError(f"音頻長度 {duration:.6f} 秒，超過 {MAX_AUDIO_SECONDS} 秒上限；不會自動截斷。")
        audio = stream.read(dtype="float32", always_2d=True)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"音頻為空或包含非有限數值：{path}")
    audio = audio.mean(axis=1, dtype=np.float32)
    if rate != SAMPLE_RATE:
        divisor = math.gcd(rate, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // divisor, rate // divisor)
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("重採樣後音頻為空或包含非有限數值。")
    return audio


def transcribe(root: Path, asr_path: Path, audio, device, dtype, max_new_tokens: int, torch) -> str:
    from transformers import AutoTokenizer, WhisperFeatureExtractor

    # Import the official files as a package; never import the qwen_asr pip package.
    modeling = importlib.import_module(f"{root.name}.modeling_qwen3_asr")
    processing = importlib.import_module(f"{root.name}.processing_qwen3_asr")
    with (asr_path / "chat_template.json").open(encoding="utf-8") as handle:
        chat_template = json.load(handle)["chat_template"]
    if not isinstance(chat_template, str) or not chat_template.strip():
        raise ValueError("ASR chat_template.json 缺少有效 chat_template。")
    processor = processing.Qwen3ASRProcessor(
        feature_extractor=WhisperFeatureExtractor.from_pretrained(str(asr_path), local_files_only=True),
        tokenizer=AutoTokenizer.from_pretrained(str(asr_path), local_files_only=True),
        chat_template=chat_template,
    )
    conversation = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "audio"}]},
    ]
    # The template only emits the audio placeholder; audio itself is supplied below.
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    prompt += ASR_PREFIX
    # Let the official processor expand <|audio_pad|> to the feature-derived length.
    batch = processor(text=prompt, audio=[audio], sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    batch = batch.to(device=device, dtype=dtype)  # BatchFeature preserves integer token/mask dtypes.
    prompt_length = batch["input_ids"].shape[-1]
    model = None
    generated = None
    token_ids = None
    try:
        print("載入原始粵語 ASR（離線）…", file=sys.stderr, flush=True)
        model = modeling.Qwen3ASRForConditionalGeneration.from_pretrained(
            str(asr_path), local_files_only=True, torch_dtype=dtype, attn_implementation="eager"
        )
        model = model.to(device=device).eval()
        with torch.inference_mode():
            # The official wrapper always returns a GenerateOutput with .sequences.
            generated = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False)
        token_ids = generated.sequences[0, prompt_length:].detach().cpu()
        if token_ids.numel() >= max_new_tokens and int(token_ids[-1]) not in (151643, 151645):
            raise RuntimeError("ASR 達到 token 上限但尚未結束；拒絕使用可能不完整嘅轉寫，請增加 --asr-max-new-tokens。")
        continuation = processor.tokenizer.decode(token_ids, skip_special_tokens=True)
        # Forced language is already in the prompt. Do not prepend it: an echoed tag
        # would otherwise survive the first split. Plain text is the transcript.
        transcript = extract_forced_transcript(continuation)
        if not transcript:
            raise RuntimeError("ASR 未產生轉寫文字；不會使用虛構文字代替。")
        return transcript
    finally:
        del model, generated, token_ids, batch, processor
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def answer_text(text_path: Path, transcript: str, instruction: str, device, dtype, max_new_tokens: int, torch) -> str:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(text_path), local_files_only=True)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"用戶指令：\n{instruction}\n\n語音轉寫：\n{transcript}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    model = None
    generated = None
    try:
        print("載入原始 Qwen3-1.7B（離線）…", file=sys.stderr, flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(text_path), local_files_only=True, torch_dtype=dtype, attn_implementation="eager"
        )
        model = model.to(device=device).eval()
        with torch.inference_mode():
            generated = model.generate(
                **batch, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(generated[0, batch["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        if not answer:
            raise RuntimeError("文字模型未產生回答。")
        return answer
    finally:
        del model, generated, batch, tokenizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> int:
    # Parse first: --help must work without torch, transformers or audio dependencies.
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    configure_runtime(root)
    project = root.parent
    asr_path = project / "粤语ASR"
    text_path = project / "基础模型" / "Qwen3-1.7B"
    if not text_path.is_dir():
        raise FileNotFoundError(f"原始文字模型目錄不存在：{text_path}")
    if args.audio is not None and not asr_path.is_dir():
        raise FileNotFoundError(f"原始 ASR 目錄不存在：{asr_path}")
    if args.output is not None:
        output = args.output.resolve()
        protected = [Path(__file__).resolve()]
        if args.audio is not None:
            protected.append(args.audio.resolve())
        if output in protected or any(output.is_relative_to(path.resolve()) for path in (asr_path, text_path)):
            raise ValueError("--output 不可覆寫輸入音頻、CLI 或原始模型目錄內文件。")
    # Invalid/oversized audio fails before loading model weights.
    audio = read_audio(args.audio) if args.audio is not None else None
    import torch

    torch.set_num_threads(8)
    device, dtype = select_device_dtype(torch, args.device, args.dtype)
    print("原始模型級聯（非端到端；不代表已完成訓練）。", file=sys.stderr, flush=True)
    if args.transcript is not None:
        transcript = args.transcript
    else:
        transcript = transcribe(root, asr_path, audio, device, dtype, args.asr_max_new_tokens, torch)
        del audio
    # transcribe's finally block has released ASR before this loads the text model.
    answer = answer_text(text_path, transcript, args.instruction, device, dtype, args.max_new_tokens, torch)
    result = {"transcript": transcript, "answer": answer}
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
