"""CLI for the unaligned continuous-feature fusion checkpoint."""
import argparse
import importlib
import json
from pathlib import Path
import sys

def main():
    parser = argparse.ArgumentParser(description='待训练的连续特征融合模型。输出质量需微调后评估。')
    parser.add_argument('--audio', type=Path)
    parser.add_argument('--instruction', required=True)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--dtype', choices=['auto', 'float32', 'bfloat16'], default='auto')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / '_runtime'))
    sys.path.insert(0, str(root.parent))
    import os
    os.environ['HF_MODULES_CACHE'] = str(root / '_module_cache')
    os.environ['HF_HUB_OFFLINE'] = '1'
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    torch.set_num_threads(8)
    processor = AutoProcessor.from_pretrained(root, trust_remote_code=True, local_files_only=True)
    audio = None
    if args.audio:
        import soundfile as sf
        from scipy.signal import resample_poly
        from math import gcd
        audio, rate = sf.read(args.audio, dtype='float32', always_2d=True)
        audio = audio.mean(axis=1)
        if len(audio) == 0 or len(audio) / rate > 30:
            raise ValueError('音频必须非空且不超过30秒；本入口不会静默截断。')
        if not np.isfinite(audio).all():
            raise ValueError('音频包含非有限数值。')
        if rate != 16000:
            divisor = gcd(rate, 16000)
            audio = resample_poly(audio, 16000 // divisor, rate // divisor).astype(np.float32)
        if len(audio) < 8000:
            audio = np.pad(audio, (0, 8000 - len(audio)))
    inputs = processor.prepare_example(args.instruction, audio=audio)
    dtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(root, trust_remote_code=True,
        local_files_only=True, torch_dtype=dtype, attn_implementation='eager').to(args.device).eval()
    inputs = {key: value.to(args.device) for key, value in inputs.items()}
    print('当前为未对齐初始化模型；语音指令能力尚需训练。', file=sys.stderr)
    with torch.inference_mode():
        ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
            suppress_tokens=[151669,151670,151676])
    answer = processor.tokenizer.decode(ids[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    result = {'mode': 'continuous_fusion_untrained', 'answer': answer,
              'quality_validated': False}
    print(answer)
    if args.output:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
