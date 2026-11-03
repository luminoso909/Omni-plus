"""Thinker 公开接口；导入此包后可使用 HF AutoConfig/AutoModelForCausalLM。"""

from transformers import AutoConfig, AutoModelForCausalLM

from .config import OmniPlusConfig
from .model_lm import OmniPlusForCausalLM, OmniPlusModel, TransformerLM

AutoConfig.register(OmniPlusConfig.model_type, OmniPlusConfig, exist_ok=True)
AutoModelForCausalLM.register(OmniPlusConfig, OmniPlusForCausalLM, exist_ok=True)

__all__ = ["OmniPlusConfig", "OmniPlusModel", "OmniPlusForCausalLM", "TransformerLM"]
