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

import os
import torch
import transformers
import deepspeed

def get_int_from_env(env_keys, default):
    """Returns the first positive env value found in the `env_keys` list or the default."""
    for e in env_keys:
        val = int(os.environ.get(e, -1))
        if val >= 0:
            return val
    return int(default)
 
local_rank = get_int_from_env(["LOCAL_RANK","PMI_RANK"], "0")
world_size = get_int_from_env(["WORLD_SIZE","PMI_SIZE"], "1")
os.environ["RANK"] = str(local_rank)
os.environ["WORLD_SIZE"] = str(world_size)
os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")


from ipex_llm import optimize_model

import torch
import time
import argparse

from transformers import AutoModelForCausalLM  # export AutoModelForCausalLM from transformers so that deepspeed use it
from transformers import LlamaTokenizer, AutoTokenizer
from deepspeed.accelerator.cpu_accelerator import CPU_Accelerator
from deepspeed.accelerator import set_accelerator, get_accelerator
from intel_extension_for_deepspeed import XPU_Accelerator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for Llama2 model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="meta-llama/Llama-2-7b-chat-hf",
                        help='The huggingface repo id for the Llama2 (e.g. `meta-llama/Llama-2-7b-chat-hf`, `meta-llama/Llama-2-13b-chat-hf` and `meta-llama/Llama-2-70b-chat-hf`) to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default="Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun",
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=32,
                        help='Max tokens to predict')
    parser.add_argument('--low-bit', type=str, default='sym_int4',
                        help='The quantization type the model will convert to.')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path
    low_bit = args.low_bit

    # First use CPU as accelerator
    # Convert to deepspeed model and apply IPEX-LLM optimization on CPU to decrease GPU memory usage
    current_accel = CPU_Accelerator()
    set_accelerator(current_accel)
    # Avoid OOM caused by parallel loading models into CPU memory
    # Please increase RANK_WAIT_TIME to avoid using too much memory.
    rank_wait_time = os.environ.get("RANK_WAIT_TIME", 0)
    if rank_wait_time != 0:
        time.sleep(local_rank * rank_wait_time)
    model = AutoModelForCausalLM.from_pretrained(args.repo_id_or_model_path,
                                                 device_map={"": "cpu"},
                                                 low_cpu_mem_usage=True,
                                                 torch_dtype=torch.float16,
                                                 trust_remote_code=True,
                                                 use_cache=True)

    model = deepspeed.init_inference(
        model,
        mp_size=world_size,
        dtype=torch.bfloat16,
        replace_method="auto",
    )

    # Use IPEX-LLM `optimize_model` to convert the model into optimized low bit format
    # Convert the rest of the model into float16 to reduce allreduce traffic
    model = optimize_model(model.module.to(f'cpu'), low_bit=low_bit).to(torch.float16)

    # Next, use XPU as accelerator to speed up inference
    current_accel = XPU_Accelerator()
    set_accelerator(current_accel)

    # Move model back to xpu
    model = model.to(f'xpu:{local_rank}')

    # Modify backend related settings 
    if world_size > 1:
        get_accelerator().set_device(local_rank)
    dist_backend = get_accelerator().communication_backend_name()
    import deepspeed.comm.comm
    deepspeed.comm.comm.cdb = None
    from deepspeed.comm.comm import init_distributed
    init_distributed()
    from ipex_llm.utils import BenchmarkWrapper
    model = BenchmarkWrapper(model)
    print(model)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Generate predicted tokens
    with torch.inference_mode():
        prompt = args.prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(f'xpu:{local_rank}')
        # ipex_llm model needs a warmup, then inference time can be accurate
        output = model.generate(input_ids,
                                max_new_tokens=args.n_predict,
                                use_cache=True)

        # start inference
        st = time.time()
        # if your selected model is capable of utilizing previous key/value attentions
        # to enhance decoding speed, but has `"use_cache": false` in its model config,
        # it is important to set `use_cache=True` explicitly in the `generate` function
        # to obtain optimal performance with IPEX-LLM INT4 optimizations
        output = model.generate(input_ids,
                                do_sample=False,
                                max_new_tokens=args.n_predict)
        torch.xpu.synchronize()
        end = time.time()
        if local_rank == 0:
            output = output.cpu()
            actual_output_len = output.shape[1] - input_ids.shape[1]
            output_str = tokenizer.decode(output[0], skip_special_tokens=True)
            avg_time = (end - st) / actual_output_len * 1000
            print(f'Inference time of generating {actual_output_len} tokens: {end-st} s, first token cost {model.first_cost} s, rest tokens average cost {model.rest_cost_mean} s')
            print('-'*20, 'Prompt', '-'*20)
            print(prompt)
            print('-'*20, 'Output', '-'*20)
            print(output_str)
    deepspeed.comm.destroy_process_group()
    print("process group destroyed, exiting...")
