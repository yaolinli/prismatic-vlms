"""
materialize.py

Factory class for initializing pretraining datasets on a per-VLM basis; provides and exports individual functions for
clear control flow.
"""
from typing import Tuple, Type

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from prismatic.conf import DatasetConfig
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.preprocessing.datasets import AlignDataset, FinetuneDataset
from prismatic.preprocessing.datasets import get_hf_datasets, hf_collate_fn


from prismatic.util.data_utils import PaddedCollatorForLanguageModeling

# Dataset Initializers =>> Maps Stage --> cls()
DATASET_INITIALIZER = {"align": AlignDataset, "finetune": FinetuneDataset, "full-finetune": FinetuneDataset}


def get_dataset_and_collator_hf(
    stage: str,
    dataset_cfg: DatasetConfig,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
    max_length=1024,
) -> Tuple[Dataset, PaddedCollatorForLanguageModeling]:
    dataset_cls = DATASET_INITIALIZER[stage]
    dataset_root_dir = dataset_cfg.dataset_root_dir
    max_length = tokenizer.model_max_length
    print("Max length", max_length)
    collator = PaddedCollatorForLanguageModeling(
        max_length, tokenizer.pad_token_id, default_image_resolution, padding_side=padding_side
    )

    # Switch on `stage`
    if stage == "align":
        annotation_json, image_dir = dataset_cfg.align_stage_components
        dataset = dataset_cls(
            dataset_root_dir / annotation_json, dataset_root_dir / image_dir, image_transform, tokenizer
        )
        return dataset, collator

    elif stage == "finetune":
        annotation_json, image_dir = dataset_cfg.finetune_stage_components
        dataset = get_hf_datasets(
            dataset_root_dir / annotation_json,
            dataset_root_dir / image_dir,
            image_transform,
            tokenizer,
            prompt_builder_fn=prompt_builder_fn,
            max_length=max_length
        )
        return dataset, hf_collate_fn

    elif stage == "full-finetune":
        annotation_json, image_dir = dataset_cfg.finetune_stage_components
        dataset = dataset_cls(
            dataset_root_dir / annotation_json,
            dataset_root_dir / image_dir,
            image_transform,
            tokenizer,
            prompt_builder_fn=prompt_builder_fn,
        )
        return dataset, collator

    else:
        raise ValueError(f"Stage `{stage}` is not supported!")


def get_dataset_and_collator(
    stage: str,
    dataset_cfg: DatasetConfig,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
    max_length=1024
) -> Tuple[Dataset, PaddedCollatorForLanguageModeling]:
    dataset_cls = DATASET_INITIALIZER[stage]
    dataset_root_dir = dataset_cfg.dataset_root_dir
    max_length = tokenizer.model_max_length
    print("Max length", max_length)
    collator = PaddedCollatorForLanguageModeling(
        max_length, tokenizer.pad_token_id, default_image_resolution, padding_side=padding_side
    )

    # Switch on `stage`
    if stage == "align":
        annotation_json, image_dir = dataset_cfg.align_stage_components
        dataset = dataset_cls(
            dataset_root_dir / annotation_json, dataset_root_dir / image_dir, image_transform, tokenizer
        )
        return dataset, collator

    elif stage == "finetune":
        annotation_json, image_dir = dataset_cfg.finetune_stage_components
        dataset = dataset_cls(
            dataset_root_dir / annotation_json,
            dataset_root_dir / image_dir,
            image_transform,
            tokenizer,
            prompt_builder_fn=prompt_builder_fn,
            max_length=max_length
        )
        return dataset, collator

    elif stage == "full-finetune":
        annotation_json, image_dir = dataset_cfg.finetune_stage_components
        dataset = dataset_cls(
            dataset_root_dir / annotation_json,
            dataset_root_dir / image_dir,
            image_transform,
            tokenizer,
            prompt_builder_fn=prompt_builder_fn,
        )
        return dataset, collator

    else:
        raise ValueError(f"Stage `{stage}` is not supported!")
