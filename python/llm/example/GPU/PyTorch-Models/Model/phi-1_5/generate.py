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
import time
import argparse

from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from ipex_llm import optimize_model

# you could tune the prompt based on your own model,
# here the prompt tuning refers to https://huggingface.co/microsoft/phi-1_5/blob/main/modeling_mixformer_sequential.py
PHI_1_5_V1_PROMPT_FORMAT = "Question: {prompt}\n\n Answer:"
generation_config = GenerationConfig(use_cache = True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for phi-1_5 model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="microsoft/phi-1_5",
                        help='The huggingface repo id for the phi-1_5 model to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default="What is AI?",
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=32,
                        help='Max tokens to predict')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

    # With only one line to enable IPEX-LLM optimization on model
    # When running LLMs on Intel iGPUs for Windows users, we recommend setting `cpu_embedding=True` in the optimize_model function.
    # This will allow the memory-intensive embedding layer to utilize the CPU instead of iGPU.
    model = optimize_model(model)
    model = model.to('xpu')
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Generate predicted tokens
    with torch.inference_mode():
        prompt = PHI_1_5_V1_PROMPT_FORMAT.format(prompt=args.prompt)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to('xpu')
        
        # ipex_llm model needs a warmup, then inference time can be accurate
        output = model.generate(input_ids, do_sample=False, max_new_tokens=args.n_predict, generation_config = generation_config)
        # start inference
        st = time.time()
        # Note that phi-1_5 uses GenerationConfig to enable 'use_cache'
        output = model.generate(input_ids, do_sample=False, max_new_tokens=args.n_predict, generation_config = generation_config)
        torch.xpu.synchronize()
        end = time.time()
        output = output.cpu()
        output_str = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f'Inference time: {end-st} s')
        print('-'*20, 'Prompt', '-'*20)
        print(prompt)
        print('-'*20, 'Output', '-'*20)
        print(output_str)
