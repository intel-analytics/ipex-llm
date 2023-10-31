"""Utilities for selecting and loading models."""
import contextlib
from typing import Type

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from bigdl.llm.vllm.config import ModelConfig
from bigdl.llm.vllm.models.bigdl_llama import BigDLLlamaForCausalLM  # pylint: disable=wildcard-import

_MODEL_REGISTRY = {
    # "AquilaModel": AquilaForCausalLM,
    # "BaiChuanForCausalLM": BaiChuanForCausalLM,  # baichuan-7b
    # "BaichuanForCausalLM": BaichuanForCausalLM,  # baichuan-13b
    # "BloomForCausalLM": BloomForCausalLM,
    # "FalconForCausalLM": FalconForCausalLM,
    # "GPT2LMHeadModel": GPT2LMHeadModel,
    # "GPTBigCodeForCausalLM": GPTBigCodeForCausalLM,
    # "GPTJForCausalLM": GPTJForCausalLM,
    # "GPTNeoXForCausalLM": GPTNeoXForCausalLM,
    # "InternLMForCausalLM": InternLMForCausalLM,
    "LlamaForCausalLM": BigDLLlamaForCausalLM,
    # "LLaMAForCausalLM": LlamaForCausalLM,  # For decapoda-research/llama-*
    # "MistralForCausalLM": MistralForCausalLM,
    # "MPTForCausalLM": MPTForCausalLM,
    # "OPTForCausalLM": OPTForCausalLM,
    # "QWenLMHeadModel": QWenLMHeadModel,
    # "RWForCausalLM": FalconForCausalLM,
}

_MODEL_CLASSES_SUPPORT_QUANTIZATION = [
    #     LlamaForCausalLM,
]


@contextlib.contextmanager
def _set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


def _get_model_architecture(config: PretrainedConfig) -> Type[nn.Module]:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _MODEL_REGISTRY:
            return _MODEL_REGISTRY[arch]
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {list(_MODEL_REGISTRY.keys())}")


def get_model(model_config: ModelConfig) -> nn.Module:
    model_class = _get_model_architecture(model_config.hf_config)

    # Get the quantization config.
    quant_config = None
    if model_config.quantization is not None:
        raise ValueError(f"Quantization is not supported for {model_class}.")

    with _set_default_torch_dtype(model_config.dtype):
        # Create a model instance.
        # The weights will be initialized as empty tensors.
        if model_class in _MODEL_CLASSES_SUPPORT_QUANTIZATION:
            model = model_class(model_config.hf_config, quant_config)
        else:
            model = model_class(model_config.hf_config)
        # Load the weights from the cached or downloaded files.
        model.load_weights(model_config.model, model_config.download_dir,
                           model_config.load_format, model_config.revision)
        if model_config.device != 'cpu':
            model = model.cuda()
    return model.eval()
