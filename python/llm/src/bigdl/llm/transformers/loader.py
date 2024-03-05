#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file provides an interface for loading models in other repos like FastChat

import torch

from bigdl.llm.transformers import AutoModelForCausalLM, AutoModel
from bigdl.llm.utils.common import invalidInputError
from transformers import AutoTokenizer, GPTJForCausalLM, LlamaTokenizer

LLAMA_IDS = ['llama', 'vicuna', 'merged-baize']


def get_tokenizer_cls(model_path: str):
    return LlamaTokenizer if any(llama_id in model_path.lower() for llama_id in LLAMA_IDS) \
        else AutoTokenizer


def get_model_cls(model_path: str, low_bit: str):
    if "chatglm" in model_path.lower() and low_bit == "bf16":
        invalidInputError(False,
                          "Currently, PyTorch does not support "
                          "bfloat16 on CPU for chatglm models.")
    return AutoModelForCausalLM if any(id in model_path.lower() for id in LLAMA_IDS) else AutoModel


def load_model(
    model_path: str,
    device: str = "cpu",
    low_bit: str = 'sym_int4',
):
    """Load a model using BigDL LLM backend."""
    tokenizer_cls = get_tokenizer_cls(model_path)
    model_cls = get_model_cls(model_path, low_bit)

    model_kwargs = {"trust_remote_code": True, "use_cache": True}
    if low_bit == "bf16":
        model_kwargs.update({"torch_dtype": torch.bfloat16})
    else:
        model_kwargs.update({"load_in_low_bit": low_bit, "torch_dtype": 'auto'})

    # Load tokenizer
    tokenizer = tokenizer_cls.from_pretrained(model_path, trust_remote_code=True)
    model = model_cls.from_pretrained(model_path, **model_kwargs).eval()

    if device == "xpu":
        import intel_extension_for_pytorch as ipex
        model = model.to('xpu')

    return model, tokenizer
