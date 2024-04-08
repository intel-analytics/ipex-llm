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
from transformers import AutoTokenizer

# you could tune the prompt based on your own model,
# here the prompt tuning refers to https://huggingface.co/togethercomputer/RedPajama-INCITE-7B-Chat#gpu-inference
RedPajama_PROMPT_FORMAT = "<human>: {prompt}\n<bot>:"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer INT4 gpu example for RedPajama model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="togethercomputer/RedPajama-INCITE-7B-Chat",
                        help='The huggingface repo id for the RedPajama to be downloaded'
                            ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default="What is AI?",
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=32,
                        help='Max tokens to predict')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path

    # Load model in 4 bit,
    # which convert the relevant layers in the model into INT4 format
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                trust_remote_code=True,
                                                load_in_4bit=True,
                                                optimize_model=True,
                                                use_cache=True)
    model = model.to('xpu')

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                            trust_remote_code=True)

    # Generate predicted tokens
    with torch.inference_mode():
        prompt = RedPajama_PROMPT_FORMAT.format(prompt=args.prompt)
        inputs = tokenizer(prompt, return_tensors='pt').to('xpu')

        # ipex_llm model needs a warmup, then inference time can be accurate
        output = model.generate(**inputs,
                                max_new_tokens=args.n_predict,
                                do_sample=True,
                                temperature=0.7,
                                top_p=0.7,
                                top_k=50,
                                return_dict_in_generate=True)

        # start inference
        st = time.time()
        # if your selected model is capable of utilizing previous key/value attentions
        # to enhance decoding speed, but has `"use_cache": false` in its model config,
        # it is important to set `use_cache=True` explicitly in the `generate` function
        # to obtain optimal performance with IPEX-LLM INT4 optimizations
        output = model.generate(**inputs,
                                max_new_tokens=args.n_predict,
                                do_sample=True,
                                temperature=0.7,
                                top_p=0.7,
                                top_k=50,
                                return_dict_in_generate=True)
        torch.xpu.synchronize()
        end = time.time()
        output_str = tokenizer.decode(output.sequences[0])
        print(f'Inference time: {end-st} s')
        print('-'*20, 'Prompt', '-'*20)
        print(prompt)
        print('-'*20, 'Output', '-'*20)
        print(output_str)