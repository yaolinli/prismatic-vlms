"""
llama2.py
Class definition for all LLMs derived from LlamaForCausalLM.
"""
from typing import Optional, Type

import torch
from torch import nn as nn
from transformers import LlamaForCausalLM, PhiForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from prismatic.models.backbones.llm.base_llm import HFCausalLLMBackbone
from prismatic.models.backbones.llm.prompting import Phi2PromptBuilder 

# Registry =>> Support Qwen2 Models (from HF Transformers)
# fmt: off
PHI2_MODELS = {
    "phi2": {
        "llm_family": "phiv2", "llm_cls": PhiForCausalLM, "hf_hub_path": "/home/yaolinli/code/official_ckpts/phi-2"
    },
}
# fmt: on


class Phi2LLMBackbone(HFCausalLLMBackbone):
    def __init__(
        self,
        llm_backbone_id: str,
        llm_max_length: int = 2048,
        hf_token: Optional[str] = None,
        inference_mode: bool = False,
        use_flash_attention_2: bool = True,
    ) -> None:
        super().__init__(
            llm_backbone_id,
            llm_max_length=llm_max_length,
            hf_token=hf_token,
            inference_mode=inference_mode,
            use_flash_attention_2=False,
            **PHI2_MODELS[llm_backbone_id],
        )
        self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        self.llm.config.pad_token_id = self.tokenizer.pad_token_id
        self.llm.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=64)

    @property
    def prompt_builder_fn(self):
        return Phi2PromptBuilder

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return LlamaDecoderLayer

    @property
    def half_precision_dtype(self) -> torch.dtype:
        """LLaMa-2 was trained in BF16; see https://huggingface.co/docs/transformers/main/model_doc/llama2."""
        return torch.bfloat16