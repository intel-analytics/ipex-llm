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
from modelscope import AutoTokenizer

# you could tune the prompt based on your own model,
BAICHUAN_PROMPT_FORMAT = "<human>{prompt} <bot>"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of saving and loading the optimized model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="baichuan-inc/Baichuan2-7B-Chat",
                        help='The ModelScope repo id for the Baichuan model to be downloaded to be downloaded'
                             ', or the path to the ModelScope checkpoint folder')
    parser.add_argument('--save-path', type=str, default=None,
                        help='The path to save the low-bit model.')
    parser.add_argument('--load-path', type=str, default=None,
                        help='The path to load the low-bit model.')
    parser.add_argument('--prompt', type=str, default="What is AI?",
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=32,
                        help='Max tokens to predict')
    args = parser.parse_args()
    model_path = args.repo_id_or_model_path
    load_path = args.load_path
    if load_path:
        model = AutoModelForCausalLM.load_low_bit(load_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     load_in_4bit=True,
                                                     trust_remote_code=True,
                                                     model_hub='modelscope')
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    save_path = args.save_path
    if save_path:
        model.save_low_bit(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Model and tokenizer are saved to {save_path}")

    # please save/load model before you run it on GPU
    model = model.to('xpu')
    
    # Generate predicted tokens
    with torch.inference_mode():
        prompt = BAICHUAN_PROMPT_FORMAT.format(prompt=args.prompt)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to('xpu')
        # ipex_llm model needs a warmup, then inference time can be accurate
        output = model.generate(input_ids,
                                max_new_tokens=args.n_predict)

        st = time.time()
        output = model.generate(input_ids,
                                max_new_tokens=args.n_predict)
        torch.xpu.synchronize()
        end = time.time()
        output = output.cpu()
        output_str = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f'Inference time: {end-st} s')
        print('-'*20, 'Output', '-'*20)
        print(output_str)
