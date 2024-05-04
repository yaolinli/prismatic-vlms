"""
llama2.py
Class definition for all LLMs derived from LlamaForCausalLM.
"""
from typing import Optional, Type

import torch
from torch import nn as nn
from transformers import LlamaForCausalLM, Qwen2ForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from prismatic.models.backbones.llm.base_llm import HFCausalLLMBackbone
from prismatic.models.backbones.llm.prompting import Qwen2PromptBuilder 

# Registry =>> Support Qwen2 Models (from HF Transformers)
# fmt: off
QWEN2_MODELS = {
    "qwen-v15-0.5b-chat": {
        "llm_family": "qwenv15", "llm_cls": Qwen2ForCausalLM, "hf_hub_path": "/home/v-shuhuairen/mycontainer/ckpt/official_ckpts/Qwen1.5-0.5B-Chat"
    },
    "qwen-v15-1.8b-chat": {
        "llm_family": "qwenv15", "llm_cls": Qwen2ForCausalLM, "hf_hub_path": "/home/v-shuhuairen/mycontainer/ckpt/official_ckpts/Qwen1.5-1.8B-Chat"
    },
    "qwen-v15-7b": {
        "llm_family": "qwenv15", "llm_cls": Qwen2ForCausalLM, "hf_hub_path": "Qwen/Qwen1.5-7B-Chat"
    },
    "qwen-v15-14b": {
        "llm_family": "qwenv15", "llm_cls": Qwen2ForCausalLM, "hf_hub_path": "Qwen/Qwen1.5-14B-Chat"
    },
}
# fmt: on


class Qwen2LLMBackbone(HFCausalLLMBackbone):
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
            **QWEN2_MODELS[llm_backbone_id],
        )

        self.llm.config.pad_token_id = self.tokenizer.pad_token_id
        self.llm.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=64)

    @property
    def prompt_builder_fn(self):
        return Qwen2PromptBuilder

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return LlamaDecoderLayer

    @property
    def half_precision_dtype(self) -> torch.dtype:
        """LLaMa-2 was trained in BF16; see https://huggingface.co/docs/transformers/main/model_doc/llama2."""
        return torch.bfloat16