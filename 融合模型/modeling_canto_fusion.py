"""ASR pretrained audio tower + identity alignment + original Qwen3 decoder.

Audio features remain continuous: response loss can reach both source modules.
This architecture is an INITIALIZATION and needs paired speech instruction SFT.
"""
import torch
from torch import nn
from transformers import Qwen3ForCausalLM
from .configuration_canto_fusion import CantoFusionConfig
from .configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig
from .modeling_qwen3_asr import Qwen3ASRAudioEncoder, _get_feat_extract_output_lengths

class CantoFusionForCausalLM(Qwen3ForCausalLM):
    config_class = CantoFusionConfig
    _no_split_modules = ["Qwen3DecoderLayer", "Qwen3ASRAudioEncoderLayer"]

    def __init__(self, config):
        super().__init__(config)
        audio_config = Qwen3ASRAudioEncoderConfig(**config.audio_config)
        audio_config._attn_implementation = "eager"
        self.audio_tower = Qwen3ASRAudioEncoder(audio_config)
        self.audio_alignment = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        nn.init.eye_(self.audio_alignment.weight)

    def encode_audio(self, input_features, feature_attention_mask):
        if input_features.ndim != 3 or feature_attention_mask.ndim != 2:
            raise ValueError("Expect mel [audio_count,128,frames] and frame mask [audio_count,frames].")
        lengths = feature_attention_mask.sum(-1).long()
        if (lengths <= 0).any():
            raise ValueError("Empty audio is not supported.")
        if input_features.shape[0] != lengths.shape[0]:
            raise ValueError("Audio batch and mask batch differ.")
        # Follow the official inference path: encode each clip independently.
        tower_param = next(self.audio_tower.parameters())
        outputs = []
        for mel, mask, length in zip(input_features, feature_attention_mask, lengths):
            unpadded = mel[:, mask.bool()].to(tower_param.device, tower_param.dtype)
            outputs.append(self.audio_tower(unpadded,
                feature_lens=length.unsqueeze(0).to(tower_param.device)).last_hidden_state)
        encoded = torch.cat(outputs, dim=0)
        align_param = self.audio_alignment.weight
        encoded = self.audio_alignment(encoded.to(align_param.device, align_param.dtype))
        return encoded, _get_feat_extract_output_lengths(lengths)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None,
                use_cache=None, cache_position=None, logits_to_keep=0,
                input_features=None, feature_attention_mask=None, **kwargs):
        if input_features is not None:
            if input_ids is None or inputs_embeds is not None:
                raise ValueError("For speech, supply input_ids and do not also supply inputs_embeds.")
            if feature_attention_mask is None:
                raise ValueError("Speech requires feature_attention_mask.")
            inputs_embeds = self.get_input_embeddings()(input_ids)
            encoded, audio_lengths = self.encode_audio(input_features, feature_attention_mask)
            mask = input_ids.eq(self.config.audio_token_id)
            # Enforce per-example counts, not just a global count (avoids cross-sample mixing).
            row_counts = mask.sum(-1).tolist()
            lens = audio_lengths.tolist()
            cursor = 0
            for count in row_counts:
                if count == 0:
                    continue
                assigned = 0
                while cursor < len(lens) and assigned < count:
                    assigned += lens[cursor]
                    cursor += 1
                if assigned != count:
                    raise ValueError("Audio feature lengths do not match the placeholder tokens in an example.")
            if cursor != len(lens) or int(mask.sum()) != encoded.shape[0]:
                raise ValueError("Audio placeholder/feature count mismatch.")
            inputs_embeds = inputs_embeds.masked_scatter(
                mask.unsqueeze(-1).expand_as(inputs_embeds),
                encoded.to(inputs_embeds.device, inputs_embeds.dtype))
            input_ids = None
        elif input_ids is not None and input_ids.eq(self.config.audio_token_id).any():
            # Cached decoding should never encounter the input placeholders again.
            raise ValueError("Audio placeholders require actual audio features.")
        return super().forward(input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
            cache_position=cache_position, logits_to_keep=logits_to_keep, **kwargs)

    def _expand_inputs_for_generation(self, expand_size=1, is_encoder_decoder=False,
            input_ids=None, **model_kwargs):
        if expand_size == 1:
            return input_ids, model_kwargs
        features = model_kwargs.pop("input_features", None)
        feature_mask = model_kwargs.pop("feature_attention_mask", None)
        if (features is None) != (feature_mask is None):
            raise ValueError("input_features and feature_attention_mask must both be present or both absent.")
        expanded_features = None
        expanded_mask = None
        if features is not None:
            if input_ids is None:
                raise ValueError("Audio expansion requires input_ids.")
            if (not torch.is_tensor(features)) or (not torch.is_tensor(feature_mask)):
                raise ValueError("Audio features and mask must be tensors.")
            if features.ndim < 1 or feature_mask.ndim != 2 or features.shape[0] != feature_mask.shape[0]:
                raise ValueError("Audio batch and mask batch differ.")
            if features.shape[0] == 0 or feature_mask.shape[0] == 0:
                raise ValueError("Empty audio is not supported.")
            clip_frames = feature_mask.sum(-1)
            if (clip_frames <= 0).any():
                raise ValueError("Empty audio is not supported.")
            clip_tokens = _get_feat_extract_output_lengths(clip_frames.long()).tolist()
            row_counts = input_ids.eq(self.config.audio_token_id).sum(-1).tolist()
            groups = []
            cursor = 0
            for count in row_counts:
                start = cursor
                assigned = 0
                if count != 0:
                    while cursor < len(clip_tokens) and assigned < count:
                        assigned += clip_tokens[cursor]
                        cursor += 1
                    if assigned != count:
                        raise ValueError("Audio feature lengths do not match the placeholder tokens in an example.")
                groups.append((start, cursor))
            if cursor != len(clip_tokens):
                raise ValueError("Audio placeholder/feature count mismatch.")
            # Parent interleaves text rows. Repeat each example's clip group so
            # [A,B] + [C] becomes [A,B,A,B,C,C].
            feature_parts = []
            mask_parts = []
            tile = [1] * (features.ndim - 1)
            mask_tile = [1] * (feature_mask.ndim - 1)
            for start, end in groups:
                if start == end:
                    continue
                feature_parts.append(features[start:end].repeat(expand_size, *tile))
                mask_parts.append(feature_mask[start:end].repeat(expand_size, *mask_tile))
            if not feature_parts:
                raise ValueError("Audio features were provided without placeholder tokens.")
            expanded_features = torch.cat(feature_parts, dim=0)
            expanded_mask = torch.cat(mask_parts, dim=0)
        input_ids, model_kwargs = super()._expand_inputs_for_generation(
            expand_size=expand_size, is_encoder_decoder=is_encoder_decoder,
            input_ids=input_ids, **model_kwargs)
        if expanded_features is not None:
            model_kwargs["input_features"] = expanded_features
            model_kwargs["feature_attention_mask"] = expanded_mask
        return input_ids, model_kwargs

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
            attention_mask=None, inputs_embeds=None, cache_position=None,
            input_features=None, feature_attention_mask=None, **kwargs):
        kwargs.pop("input_features", None)
        kwargs.pop("feature_attention_mask", None)
        result = super().prepare_inputs_for_generation(input_ids,
            past_key_values=past_key_values, attention_mask=attention_mask,
            inputs_embeds=inputs_embeds, cache_position=cache_position, **kwargs)
        sliced = result.get("input_ids")
        keep_audio = (
            sliced is not None
            and torch.is_tensor(sliced)
            and bool(sliced.eq(self.config.audio_token_id).any().item())
        )
        if keep_audio:
            if input_features is not None:
                result["input_features"] = input_features
            if feature_attention_mask is not None:
                result["feature_attention_mask"] = feature_attention_mask
        else:
            result.pop("input_features", None)
            result.pop("feature_attention_mask", None)
        return result
