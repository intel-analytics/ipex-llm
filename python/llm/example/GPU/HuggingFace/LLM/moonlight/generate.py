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

import time
import argparse
import torch
from ipex_llm.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for Moonlight model')
    parser.add_argument('--converted-model-path', type=str, required=True,
                        help='Model path to the converted Moonlight model by convert.py')
    parser.add_argument('--prompt', type=str, default="What is AI?",
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=32,
                        help='Max tokens to predict')
    
    args = parser.parse_args()
    converted_model_path = args.converted_model_path

    print("start to load")
    # Load model in 4 bit,
    # which convert the relevant layers in the model into INT4 format
    model = AutoModelForCausalLM.from_pretrained(converted_model_path,
                                                 load_in_4bit=True,
                                                 optimize_model=True,
                                                 trust_remote_code=True,
                                                 use_cache=True)
    model = model.half().to('xpu')

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(converted_model_path, trust_remote_code=True)

    # Generate predicted tokens
    with torch.inference_mode():
        # here the prompt tuning refers to
        # https://huggingface.co/moonshotai/Moonlight-16B-A3B-Instruct#inference-with-hugging-face-transformers
        messages = [
            {"role": "system", "content": "You are a helpful assistant provided by Moonshot-AI."},
            {"role": "user", "content": args.prompt}
        ]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to('xpu')

        print(input_ids)

        # ipex_llm model needs a warmup, then inference time can be accurate
        output = model.generate(input_ids,
                                max_new_tokens=args.n_predict)
        
        # start inference
        st = time.time()
        output = model.generate(input_ids,
                                max_new_tokens=args.n_predict)
        torch.xpu.synchronize()
        end = time.time()

        output_str = tokenizer.decode(output[0], skip_special_tokens=False)
        print(f'Inference time: {end-st} s')
        print('-'*20, 'Prompt', '-'*20)
        print(args.prompt)
        print('-'*20, 'Output', '-'*20)
        print(output_str)
