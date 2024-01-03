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

from bigdl.llm import optimize_model

import torch
import intel_extension_for_pytorch as ipex
import time
import argparse

from transformers import AutoModelForCausalLM  # export AutoModelForCausalLM from transformers so that deepspeed use it
from transformers import LlamaTokenizer, AutoTokenizer
from deepspeed.accelerator.cpu_accelerator import CPU_Accelerator
from deepspeed.accelerator import set_accelerator, get_accelerator
from intel_extension_for_deepspeed import XPU_Accelerator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for Llama2 and Vicuna model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="meta-llama/Llama-2-7b-chat-hf",
                        help='The huggingface repo id for the Llama2 (e.g. `meta-llama/Llama-2-7b-chat-hf` and `meta-llama/Llama-2-13b-chat-hf`) '
                             'and Vicuna (e.g. `lmsys/vicuna-33b-v1.3`) to be downloaded, or the path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default="Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun",
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=32,
                        help='Max tokens to predict')
    parser.add_argument('--local_rank', type=int, default=0, help='this is automatically set when using deepspeed launcher')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("RANK", "-1")) # RANK is automatically set by CCL distributed backend
    if local_rank == -1: # args.local_rank is automatically set by deepspeed subprocess command
        local_rank = args.local_rank

    # First use CPU as accelerator
    # Convert to deepspeed model and apply bigdl-llm optimization on CPU to decrease GPU memory usage
    current_accel = CPU_Accelerator()
    set_accelerator(current_accel)
    model = AutoModelForCausalLM.from_pretrained(args.repo_id_or_model_path,
                                                 device_map={"": "cpu"},
                                                 low_cpu_mem_usage=True,
                                                 torch_dtype=torch.float16,
                                                 trust_remote_code=True,
                                                 use_cache=True)

    model = deepspeed.init_inference(
        model,
        mp_size=world_size,
        dtype=torch.float16,
        replace_method="auto",
    )

    # Use bigdl-llm `optimize_model` to convert the model into optimized low bit format
    # Convert the rest of the model into float16 to reduce allreduce traffic
    model = optimize_model(model.module.to(f'cpu'), low_bit='sym_int4').to(torch.float16)

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

    print(model)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Generate predicted tokens
    with torch.inference_mode():
        prompt = args.prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(f'xpu:{local_rank}')
        print(f'Input length: {input_ids.shape[1]}')
        # ipex model needs a warmup, then inference time can be accurate
        output = model.generate(input_ids,
                                max_new_tokens=args.n_predict,
                                use_cache=True)

        # start inference
        st = time.time()
        # if your selected model is capable of utilizing previous key/value attentions
        # to enhance decoding speed, but has `"use_cache": false` in its model config,
        # it is important to set `use_cache=True` explicitly in the `generate` function
        # to obtain optimal performance with BigDL-LLM INT4 optimizations
        output = model.generate(input_ids,
                                do_sample=False,
                                max_new_tokens=args.n_predict)
        torch.xpu.synchronize()
        end = time.time()
        if local_rank == 0:
            output = output.cpu()
            output_str = tokenizer.decode(output[0], skip_special_tokens=True)
            print(f'Inference time: {end-st} s')
            print('-'*20, 'Prompt', '-'*20)
            print(prompt)
            print('-'*20, 'Output', '-'*20)
            print(output_str)
