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

from bigdl.llm.transformers import AutoModelForCausalLM, AutoModel, get_enable_ipex
import time
from datetime import date
import argparse
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
    return AutoModel if "chatglm" in model_path.lower() else AutoModelForCausalLM


def load_model(
    model_path: str,
    device: str = "cpu",
    low_bit: str = 'sym_int4',
    trust_remote_code: bool = True,
):
    """Load a model using BigDL LLM backend."""

    # Do a sanity check for device:
    invalidInputError(device == 'cpu' or device == 'xpu',
                      "BigDL-LLM only supports device cpu or xpu")

    tokenizer_cls = get_tokenizer_cls(model_path)
    model_cls = get_model_cls(model_path, low_bit)
    model_kwargs = {"use_cache": True}
    if trust_remote_code:
        model_kwargs["trust_remote_code"] = True
    if low_bit == "bf16":
        model_kwargs.update({"load_in_low_bit": low_bit, "torch_dtype": torch.bfloat16})
    else:
        model_kwargs.update({"load_in_low_bit": low_bit, "torch_dtype": 'auto'})

    # Load tokenizer
    tokenizer = tokenizer_cls.from_pretrained(model_path, trust_remote_code=True)
    model = model_cls.from_pretrained(model_path, **model_kwargs)
    if not get_enable_ipex(low_bit):
        model = model.eval()

    if device == "xpu":
        import intel_extension_for_pytorch as ipex
        model = model.to('xpu')

    return model, tokenizer


def try_run_test_generation(local_model_hub, model_path, device, low_bit):
    path = get_model_path(model_path, local_model_hub)
    try:
        run_test_generation(path, device, low_bit)
    except:
        print(f"Loading model failed for model {model_path} \
              with device:{device} and low_bit:{low_bit}")
        return "False"
    return "True"


def get_model_path(repo_id, local_model_hub):
    if local_model_hub:
        repo_model_name = repo_id.split("/")[1]
        local_model_path = local_model_hub + os.path.sep + repo_model_name
        invalidInputError(os.path.isdir(local_model_path),
                          local_model_path + " not exists!, Please check your models' folder.")
        return local_model_path
    else:
        return repo_id


def run_test_generation(model_path, device, low_bit):
    model, tokenizer = load_model(model_path, device, low_bit, True)
    with torch.inference_mode():
        prompt = "What is AI?"
        # TODO: if gpu, will need to move the tensor to xpu
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        if device == 'xpu':
            input_ids = input_ids.to('xpu')
        st = time.time()
        # if your selected model is capable of utilizing previous key/value attentions
        # to enhance decoding speed, but has `"use_cache": false` in its model config,
        # it is important to set `use_cache=True` explicitly in the `generate` function
        # to obtain optimal performance with BigDL-LLM INT4 optimizations
        output = model.generate(input_ids,
                                max_new_tokens=32)
        end = time.time()
        output_str = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f'Inference time: {end-st} s')
        print('-'*20, 'Prompt', '-'*20)
        print(prompt)
        print('-'*20, 'Output', '-'*20)
        print(output_str)


# Provide a main method for test loads
# Note that this only test loading models instead of generation correctness
if __name__ == '__main__':
    import os
    # TODO: move config.yaml to a different folder
    current_dir = os.path.dirname(os.path.realpath(__file__))
    results = []
    from omegaconf import OmegaConf
    conf = OmegaConf.load(f'{current_dir}/load_config.yaml')
    today = date.today()
    import pandas as pd
    csv_name = f'{current_dir}/loader-results-{today}.csv'
    for model in conf.repo_id:
        for low_bit in conf.low_bit:
            for device in conf.device:
                result = try_run_test_generation(conf['local_model_hub'], model, device, low_bit)
                results.append([model, device, low_bit, result])

    df = pd.DataFrame(results, columns=['model', 'device', 'low_bit', 'result'])
    df.to_csv(csv_name)
    results = []
