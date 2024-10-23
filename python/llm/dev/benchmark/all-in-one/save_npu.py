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

# this code is to support converting of model in load bit
# for performance tests using load_low_bit

import time
import torch
import os
import argparse
from ipex_llm.transformers.npu_model import AutoModelForCausalLM
from transformers import AutoTokenizer
from run import get_model_path

current_dir = os.path.dirname(os.path.realpath(__file__))

def save_npu_model_in_low_bit(repo_id,
                          local_model_hub,
                          low_bit,
                          max_output_len, max_prompt_len, intra_pp, inter_pp,
                          disable_transpose_value_cache,
                          quantization_group_size):
    model_path = get_model_path(repo_id, local_model_hub)
    # Load model in 4 bit,
    # which convert the relevant layers in the model into INT4 format
    st = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            attn_implementation="eager",
            load_in_low_bit="sym_int4",
            optimize_model=True,
            max_output_len=max_output_len,
            max_prompt_len=max_prompt_len,
            intra_pp=intra_pp,
            inter_pp=inter_pp,
            transpose_value_cache=not disable_transpose_value_cache,
            quantization_group_size=quantization_group_size
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    end = time.perf_counter()
    print(">> loading of and converting of model costs {}s".format(end - st))

    model.save_low_bit(model_path+'-npu-'+low_bit)
    tokenizer.save_pretrained(model_path+'-npu-'+low_bit)
    print(f"Model saved to {model_path+'-npu-'+low_bit}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict Tokens using `generate()` API for npu model"
    )
    parser.add_argument("--max-output-len", type=int, default=1024)
    parser.add_argument("--max-prompt-len", type=int, default=512)
    parser.add_argument("--disable-transpose-value-cache", action="store_true", default=False)
    parser.add_argument("--intra-pp", type=int, default=2)
    parser.add_argument("--inter-pp", type=int, default=2)
    parser.add_argument("--quantization_group_size", type=int, default=0)

    args = parser.parse_args()
    from omegaconf import OmegaConf
    conf = OmegaConf.load(f'{current_dir}/config.yaml')

    for model in conf.repo_id:
        save_npu_model_in_low_bit(repo_id=model,
                              local_model_hub=conf['local_model_hub'],
                              low_bit=conf['low_bit'],
                              max_output_len=args.max_output_len,
                              max_prompt_len=args.max_prompt_len,
                              intra_pp=args.intra_pp,
                              inter_pp=args.inter_pp,
                              disable_transpose_value_cache=args.disable_transpose_value_cache,
                              quantization_group_size=args.quantization_group_size,
                              )
