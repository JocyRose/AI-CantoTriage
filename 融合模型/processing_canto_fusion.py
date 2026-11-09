"""Speech + text processing and response-only training labels."""
import torch
from transformers.feature_extraction_utils import BatchFeature
from .processing_qwen3_asr import Qwen3ASRProcessor

DEFAULT_SYSTEM = "你係一個廣東話助手。請根據用戶語音同文字要求，用自然嘅粵語口語文字回答。資料不足時要講清楚，唔好編造。"

class CantoFusionProcessor(Qwen3ASRProcessor):
    def prepare_example(self, instruction, audio=None, answer=None,
                        system=DEFAULT_SYSTEM, sampling_rate=16000):
        content = []
        if audio is not None:
            content.append({'type': 'audio', 'audio': 'local'})
        content.append({'type': 'text', 'text': instruction})
        messages = [{'role': 'system', 'content': system},
                    {'role': 'user', 'content': content}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
        result = self(text=prompt, audio=audio, sampling_rate=sampling_rate, return_tensors='pt')
        if answer is not None:
            # Encode the target separately so BPE cannot merge the prompt boundary.
            target = self.tokenizer(answer + self.tokenizer.eos_token,
                                    add_special_tokens=False, return_tensors='pt')['input_ids']
            prompt_len = result['input_ids'].shape[1]
            result['input_ids'] = torch.cat([result['input_ids'], target], dim=1)
            result['attention_mask'] = torch.ones_like(result['input_ids'])
            labels = result['input_ids'].clone()
            labels[:, :prompt_len] = -100
            result['labels'] = labels
        return BatchFeature(result)
