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

import torch
from bigdl.llm.transformers import AutoModelForCausalLM, AutoModel
from transformers import AutoTokenizer, GPTJForCausalLM, LlamaTokenizer

LLAMA_IDS = ['llama', 'vicuna', 'merged-baize']


def load_model(
    model_path: str,
    device: str = "cpu",
    low_bit: str = 'sym_int4',
    debug: bool = False,
):
    """Load a model using BigDL LLM backend."""
    if low_bit == "bf16":
        if "chatglm" in model_path.lower():
            print("Currently pytorch do not support bfloat16 on cpu for chatglm models.")
            return
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
            use_cache=True
        )
        tokenizer_cls = (
            LlamaTokenizer if any(llama_id in model_path.lower() for llama_id in LLAMA_IDS)
            else AutoTokenizer
        )
        tokenizer = tokenizer_cls.from_pretrained(model_path, trust_remote_code=True)
    elif low_bit == "fp16" and device == "xpu":
        import intel_extension_for_pytorch as ipex
        model_cls = (
            AutoModelForCausalLM if any(id in model_path.lower() for id in LLAMA_IDS)
            else AutoModel
        )
        model = model_cls.from_pretrained(
            model_path, trust_remote_code=True, use_cache=True
        ).to('xpu')
        tokenizer_cls = (
            LlamaTokenizer if any(id in model_path.lower() for id in LLAMA_IDS)
            else AutoTokenizer
        )
        tokenizer = tokenizer_cls.from_pretrained(model_path, trust_remote_code=True)
    else:
        if "chatglm" in model_path.lower():
            model = AutoModel.from_pretrained(
                model_path, load_in_low_bit=low_bit, trust_remote_code=True, torch_dtype='auto'
            ).eval()
        else:
            model_cls = (
                AutoModelForCausalLM if any(id in model_path.lower() for id in LLAMA_IDS)
                else AutoModel
            )
            model = model_cls.from_pretrained(
                model_path, load_in_low_bit=low_bit, trust_remote_code=True, use_cache=True
            ).eval()

        tokenizer_cls = (
            LlamaTokenizer if any(id in model_path.lower() for id in LLAMA_IDS)
            else AutoTokenizer
        )
        tokenizer = tokenizer_cls.from_pretrained(model_path, trust_remote_code=True)

        if device == "xpu":
            import intel_extension_for_pytorch as ipex
            model = model.to('xpu')
            if isinstance(model, GPTJForCausalLM):
                model = ipex.optimize(model.eval(), inplace=True)

    if debug:
        print(model)

    return model, tokenizer
