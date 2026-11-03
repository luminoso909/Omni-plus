"""Omni-plus 模型入口；HF AutoModelForCausalLM 对应文本 LM。"""

from transformers import AutoConfig, AutoModelForCausalLM

from .config import OmniPlusConfig
from .model_omni import OmniOutput, OmniPlusOmni
from .thinker.model_thinker import OmniPlusForCausalLM, OmniPlusModel, TransformerLM

AutoConfig.register(OmniPlusConfig.model_type, OmniPlusConfig, exist_ok=True)
AutoModelForCausalLM.register(OmniPlusConfig, OmniPlusForCausalLM, exist_ok=True)

__all__ = [
    "OmniPlusConfig",
    "OmniPlusForCausalLM",
    "OmniPlusModel",
    "TransformerLM",
    "OmniPlusOmni",
    "OmniOutput",
]
