"""Thinker 文本主干公开接口。"""

from ..config import OmniPlusConfig
from .model_lm import OmniPlusForCausalLM, OmniPlusModel, TransformerLM

__all__ = ["OmniPlusConfig", "OmniPlusForCausalLM", "OmniPlusModel", "TransformerLM"]
