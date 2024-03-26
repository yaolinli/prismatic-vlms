"""
nn_utils.py

Utility functions and PyTorch submodule definitions.
"""
import copy
import os
from typing import Union
import torch
import torch.nn as nn
import re
import math 
from transformers import PretrainedConfig, Blip2PreTrainedModel, Blip2Config, Blip2QFormerModel, Blip2QFormerConfig
from typing import Optional
from einops import rearrange
from timm.layers import LayerNorm, LayerNorm2d
from timm.models.regnet import RegStage
from functools import partial
from transformers.configuration_utils import PretrainedConfig


class HoneybeeVisualProjectorConfig(PretrainedConfig):
    model_type = "mllm_visual_projector"
    def __init__(
        self,
        projector_type: str = "c-abstractor",
        hidden_size: int = 1024,  #
        num_hidden_layers: int = 6,  #
        num_attention_heads: int = 16,  #
        intermediate_size: int = 4096,  #
        attention_probs_dropout_prob: float = 0.1,  #
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-6,  #
        encoder_hidden_size: int = 1024,  # This will be overwritten by vision_model's hidden_size
        depth: int = 1, # the layer num of ResNet Block
        mlp_depth: int = 2,  # the layer num of the MLP Block that mapping projector dim to llm dim
        num_queries: int = 144,
        pos_emb=False,
        feature_layer_index=-1,  # vision feature layer index; -1: last layer
        num_eos_tokens=0,
        use_cls=True,
        prenorm=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.depth = depth
        self.mlp_depth = mlp_depth
        self.num_queries = num_queries
        self.projector_type = projector_type
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.encoder_hidden_size = encoder_hidden_size

        self.pos_emb = pos_emb
        self.feature_layer_index = feature_layer_index
        self.num_eos_tokens = num_eos_tokens
        self.use_cls = use_cls
        self.prenorm = prenorm
    
    
def build_pos_embeds(
    config: HoneybeeVisualProjectorConfig, num_input_tokens: int, vision_hidden_size: int
):
    # pos emb
    if config.pos_emb:
        pos_emb = torch.nn.Parameter(torch.zeros(1, num_input_tokens, vision_hidden_size))
        nn.init.trunc_normal_(pos_emb, mean=0.0, std=0.02)
    else:
        pos_emb = None

    return pos_emb


def build_eos_tokens(config: HoneybeeVisualProjectorConfig, output_hidden_size: int):
    # think tokens
    num_eos_tokens = config.num_eos_tokens
    if num_eos_tokens:
        eos_tokens = torch.nn.Parameter(torch.randn(1, num_eos_tokens, output_hidden_size))
        nn.init.trunc_normal_(eos_tokens, mean=0.0, std=config.initializer_range)
    else:
        eos_tokens = None

    return eos_tokens


def build_prenorm(config: HoneybeeVisualProjectorConfig):
    if config.prenorm:
        prenorm = LayerNorm(config.encoder_hidden_size)
    else:
        prenorm = None
    return prenorm


def build_mlp(depth, hidden_size, output_hidden_size):
    layers = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        layers.append(nn.SiLU())
        layers.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*layers)


class Projector(nn.Module):
    """Base projector class"""
    def __init__(
        self,
        config: HoneybeeVisualProjectorConfig,
        num_input_tokens: int = 576,
        output_hidden_size: int = 4096,
    ):
        super().__init__()
        self.config = config
        self.num_input_tokens = num_input_tokens
        self.output_hidden_size = output_hidden_size

        # think tokens
        self.eos_tokens = build_eos_tokens(config, output_hidden_size)

        # pos emb
        self.pos_emb = build_pos_embeds(config, num_input_tokens, config.encoder_hidden_size)

        self.prenorm = build_prenorm(config)

        self.build_net()

    def build_net(self):
        raise NotImplementedError()

    def _forward(self, x):
        raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, encoder_hidden_size) tensor from the visual backbone (CLIP visual encoder), including cls token.
        """
        if self.prenorm is not None:
            x = self.prenorm(x)

        if self.pos_emb is not None:
            x += self.pos_emb

        x = self._forward(x)  # (B, L, output_hidden_size)

        B = x.size(0)
        if self.eos_tokens is not None:
            x = torch.cat([x, self.eos_tokens.expand(B, -1, -1)], dim=1)
        return x
    

class ConvProjector(Projector):
    def _forward(self, x):
        # x: [B, L, dim]
        # x = x[:, 1:]  # drop cls token and 2d forward
        hw = int(x.size(1) ** 0.5)
        x = rearrange(x, "b (h w) d -> b d h w", h=hw, w=hw)
        x = self.net(x)
        x = rearrange(x, "b d h w -> b (h w) d")
        x = self.readout(x)

        return x
    

class CAbstractor(ConvProjector):
    """C-Abstractor"""
    def build_net(self):
        encoder_hidden_size = self.config.encoder_hidden_size
        hidden_size = self.config.hidden_size
        output_hidden_size = self.output_hidden_size
        depth = self.config.depth
        mlp_depth = self.config.mlp_depth

        n_queries = self.config.num_queries
        assert (n_queries ** 0.5).is_integer(), "n_queries must be square number"
        hw = int(n_queries ** 0.5)

        # RegBlock = ResBlock + SE
        RegBlock = partial(
            RegStage,
            stride=1,
            dilation=1,
            act_layer=nn.SiLU,
            norm_layer=LayerNorm2d,
        )

        s1 = RegBlock(
            depth,
            encoder_hidden_size,
            hidden_size,
        )
        sampler = nn.AdaptiveAvgPool2d((hw, hw))
        s2 = RegBlock(
            depth,
            hidden_size,
            hidden_size,
        )

        self.net = nn.Sequential(s1, sampler, s2)
        self.readout = build_mlp(mlp_depth, hidden_size, output_hidden_size)


class CAbstractorProjector(nn.Module):
    def __init__(
        self,
        vision_dim: int,
        llm_dim: int,
        num_query_tokens: int = 64,
        depth: int = 3,  # the layer num of ResNet Block
        mlp_depth: int = 2,  # the layer num of the MLP Block that mapping projector dim to llm dim
    ) -> None:
        super().__init__()
        # encoder_hidden_size: int = 1024,  # This will be overwritten by vision_model's hidden_size
        # depth: int = 1,
        # mlp_depth: int = 2,
        # num_queries: int = 144,
        CAbstractor_config = HoneybeeVisualProjectorConfig(projector_type='c-abstractor', encoder_hidden_size=vision_dim, depth=depth, mlp_depth=mlp_depth, num_queries=num_query_tokens)
        self.projector = CAbstractor(CAbstractor_config, output_hidden_size=llm_dim)

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        # img_patches: [16, 576, 1024]
        return self.projector(img_patches)
    

# === Definitions for Various Projection Modules, with Signature :: [..., in_dim] --> [..., out_dim] ===
class LinearProjector(nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.projector = nn.Linear(vision_dim, llm_dim, bias=True)

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(img_patches)


class MLPProjector(nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int, mlp_type: str = "gelu-mlp") -> None:
        super().__init__()
        if mlp_type == "gelu-mlp":
            self.projector = nn.Sequential(
                nn.Linear(vision_dim, llm_dim, bias=True),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim, bias=True),
            )
        else:
            raise ValueError(f"Projector with `{mlp_type = }` is not supported!")

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(img_patches)


def qformer_config_template(vision_dim, llm_dim, num_query_tokens, num_qformer_layers):
    num_hidden_layers = num_qformer_layers
    num_query_tokens = num_query_tokens

    qformer_config = type(
        "Blip2Config",
        (PretrainedConfig,),
        {
            "initializer_factor": 1.0,
            "initializer_range": 0.02,
            "model_type": "blip-2",
            "num_query_tokens": num_query_tokens,
            "hidden_size": llm_dim,
            "mm_hidden_size": vision_dim,
            "qformer_config": type(
                "qformer_config",
                (PretrainedConfig,),
                {
                    "_name_or_path": "",
                    "add_cross_attention": False,
                    "architectures": None,
                    "attention_probs_dropout_prob": 0.0,
                    "bad_words_ids": None,
                    "begin_suppress_tokens": None,
                    "bos_token_id": None,
                    "chunk_size_feed_forward": 0,
                    "classifier_dropout": None,
                    "cross_attention_frequency": 2,
                    "cross_attention_hidden_size": None,
                    "decoder_start_token_id": None,
                    "diversity_penalty": 0.0,
                    "do_sample": False,
                    "early_stopping": False,
                    "encoder_hidden_size": 1408,
                    "encoder_no_repeat_ngram_size": 0,
                    "eos_token_id": None,
                    "exponential_decay_length_penalty": None,
                    "finetuning_task": None,
                    "forced_bos_token_id": None,
                    "forced_eos_token_id": None,
                    "hidden_act": "gelu",
                    "hidden_dropout_prob": 0.0,
                    "hidden_size": llm_dim,
                    "id2label": {"0": "LABEL_0", "1": "LABEL_1"},
                    "initializer_range": 0.02,
                    "intermediate_size": llm_dim * 4 ,
                    "is_decoder": False,
                    "is_encoder_decoder": False,
                    "label2id": {"LABEL_0": 0, "LABEL_1": 1},
                    "layer_norm_eps": 1e-12,
                    "length_penalty": 1.0,
                    "max_length": 20,
                    "max_position_embeddings": 512,
                    "min_length": 0,
                    "model_type": "blip_2_qformer",
                    "no_repeat_ngram_size": 0,
                    "num_attention_heads": 64,
                    "num_beam_groups": 1,
                    "num_beams": 1,
                    "num_hidden_layers": num_hidden_layers,
                    "num_return_sequences": 1,
                    "output_attentions": False,
                    "output_hidden_states": False,
                    "output_scores": False,
                    "pad_token_id": 0,
                    "position_embedding_type": "absolute",
                    "prefix": None,
                    "problem_type": None,
                    "pruned_heads": {},
                    "remove_invalid_values": False,
                    "repetition_penalty": 1.0,
                    "return_dict": True,
                    "return_dict_in_generate": False,
                    "sep_token_id": None,
                    "suppress_tokens": None,
                    "task_specific_params": None,
                    "temperature": 1.0,
                    "tf_legacy_loss": False,
                    "tie_encoder_decoder": False,
                    "tie_word_embeddings": True,
                    "tokenizer_class": None,
                    "top_k": 50,
                    "top_p": 1.0,
                    "torch_dtype": None,
                    "torchscript": False,
                    "transformers_version": "4.27.0.dev0",
                    "typical_p": 1.0,
                    "use_bfloat16": False,
                    "vocab_size": 30522,
                },
            )(),
        },
    )()
    return qformer_config


class CustomBlip2QFormerMultiHeadAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False, save_attention=False):
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)"
                % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        if is_cross_attention:
            self.key = nn.Linear(config.encoder_hidden_size, self.all_head_size)
            self.value = nn.Linear(config.encoder_hidden_size, self.all_head_size)
        else:
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        self.save_attention = save_attention
        self.attn = None
        self.attn_gradients = None

    def save_attn_gradients(self, attn_gradients):
        if self.attn_gradients is not None:
            self.attn_gradients = [self.attn_gradients, attn_gradients]
        else:
            self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attn(self, attn):
        if self.attn is not None:
            self.attn = [self.attn, attn]
        else:
            self.attn = attn

    def get_attn(self):
        return self.attn

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        mixed_query_layer = self.query(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)

        past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        if self.save_attention:
            self.save_attn(attention_probs)
            attention_probs.register_hook(self.save_attn_gradients)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs_dropped = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs_dropped = attention_probs_dropped * head_mask

        context_layer = torch.matmul(attention_probs_dropped, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        outputs = outputs + (past_key_value,)
        return outputs


class CustomBlip2QFormerModel(Blip2QFormerModel):
    def __init__(self, config: Blip2QFormerConfig):
        super().__init__(config)

        for layer_idx, layer in enumerate(self.encoder.layer):
            layer.attention.attention = CustomBlip2QFormerMultiHeadAttention(config)

            if layer_idx % config.cross_attention_frequency == 0:
                layer.crossattention.attention = CustomBlip2QFormerMultiHeadAttention(config, is_cross_attention=True)
                layer.has_cross_attention = True
            else:
                layer.has_cross_attention = False


class Blip2Model(Blip2PreTrainedModel):
    def __init__(self, config: Blip2Config):
        super().__init__(config)

        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        self.qformer = CustomBlip2QFormerModel(config.qformer_config)
        # qformer_config.hidden_size = vision_dim
        # hidden_size = llm_dim 
        self.qformer_input_proj = nn.Linear(config.mm_hidden_size, config.qformer_config.encoder_hidden_size)
        self.proj = nn.Linear(config.qformer_config.hidden_size, config.hidden_size)

        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_embeds = self.qformer_input_proj(pixel_values)
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        ).last_hidden_state
        # print('qformer out', query_outputs.shape)
        query_outputs = self.proj(query_outputs)
        return query_outputs


class QFormerProjector(nn.Module):
    def __init__(
        self,
        vision_dim: int,
        llm_dim: int,
        num_query_tokens: int = 64,
        num_qformer_layers: int = 64,
    ) -> None:
        super().__init__()
        qformer_config = qformer_config_template(
            vision_dim=vision_dim, llm_dim=llm_dim, num_query_tokens=num_query_tokens, num_qformer_layers=num_qformer_layers
        )
        self.projector = Blip2Model(qformer_config)

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(img_patches)


class FusedMLPProjector(nn.Module):
    def __init__(self, fused_vision_dim: int, llm_dim: int, mlp_type: str = "fused-gelu-mlp") -> None:
        super().__init__()
        self.initial_projection_dim = fused_vision_dim * 4
        if mlp_type == "fused-gelu-mlp":
            self.projector = nn.Sequential(
                nn.Linear(fused_vision_dim, self.initial_projection_dim, bias=True),
                nn.GELU(),
                nn.Linear(self.initial_projection_dim, llm_dim, bias=True),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim, bias=True),
            )
        else:
            raise ValueError(f"Fused Projector with `{mlp_type = }` is not supported!")

    def forward(self, fused_img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(fused_img_patches)
