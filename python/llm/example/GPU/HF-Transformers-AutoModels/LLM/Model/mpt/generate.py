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

from ipex_llm.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, GenerationConfig

# you could tune the prompt based on your own model,
# here the prompt tuning refers to https://huggingface.co/spaces/mosaicml/mpt-30b-chat/blob/main/app.py
MPT_PROMPT_FORMAT = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for MPT model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="mosaicml/mpt-7b-chat",
                        help='The huggingface repo id for the MPT models'
                             '(e.g. `mosaicml/mpt-7b-chat`) to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default="What is AI?",
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=32,
                        help='Max tokens to predict')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path

    # Load model in 4 bit,
    # which convert the relevant layers in the model into INT4 format
    # When running LLMs on Intel iGPUs for Windows users, we recommend setting `cpu_embedding=True` in the from_pretrained function.
    # This will allow the memory-intensive embedding layer to utilize the CPU instead of iGPU.
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 load_in_4bit=True,
                                                 optimize_model=False,
                                                 trust_remote_code=True,
                                                 use_cache=True)
    model = model.to('xpu')

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True)
    
    # Generate predicted tokens
    with torch.inference_mode():
        prompt = MPT_PROMPT_FORMAT.format(prompt=args.prompt)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to('xpu')
        # ipex_llm model needs a warmup, then inference time can be accurate
        output = model.generate(input_ids,
                                max_new_tokens=args.n_predict)

        # start inference
        # enabling `use_cache=True` allows the model to utilize the previous
        # key/values attentions to speed up decoding;
        # to obtain optimal performance with IPEX-LLM INT4 optimizations,
        # it is important to set use_cache=True for MPT models
        mpt_generation_config = GenerationConfig(
            max_new_tokens=args.n_predict, 
            use_cache=True, 
            pad_token_id=tokenizer.eos_token_id, 
            eos_token_id=tokenizer.eos_token_id
        )
        st = time.time()
        output = model.generate(input_ids,
                                generation_config=mpt_generation_config)
        end = time.time()
        output = output.cpu()
        output_str = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f'Inference time: {end-st} s')
        print('-'*20, 'Prompt', '-'*20)
        print(prompt)
        print('-'*20, 'Output', '-'*20)
        print(output_str)
