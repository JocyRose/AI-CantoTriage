"""Continuous speech/text fusion configuration. Unaligned initialization."""
from transformers import Qwen3Config

class CantoFusionConfig(Qwen3Config):
    model_type = "canto_fusion"

    def __init__(self, audio_config=None, audio_token_id=151676,
                 alignment_status="untrained", **kwargs):
        super().__init__(**kwargs)
        self.audio_config = audio_config or {}
        self.audio_token_id = audio_token_id
        self.alignment_status = alignment_status
