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
# here the prompt tuning refers to [CHANGE REFERENCE URL HERE]
DEFAULT_SYSTEM_PROMPT = """\
"""

def get_prompt(user_input: str, chat_history: list[tuple[str, str]],
               system_prompt: str) -> str:
    prompt_texts = [f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|>']

    for history_input, history_response in chat_history:
        prompt_texts.append(f'<|start_header_id|>user<|end_header_id|>\n{history_input.strip()}<|eot_id|>')
        prompt_texts.append(f'<|start_header_id|>assistant<|end_header_id|>\n{history_response.strip()}<|eot_id|>')

    prompt_texts.append(f'<|start_header_id|>user<|end_header_id|>\n{user_input.strip()}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n')
    return ''.join(prompt_texts)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for Llama3 model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="meta-llama/Llama-3-8b-instruction-hf",
                        help='The huggingface repo id for the Llama3 (e.g. `meta-llama/Llama-3-8b-instruction-hf` and `meta-llama/Llama-2-13b-chat-hf`) to be downloaded'
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
                                                 optimize_model=True,
                                                 trust_remote_code=True,
                                                 use_cache=True)
    model = model.half().to('xpu')

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Generate predicted tokens
    with torch.inference_mode():
        prompt = get_prompt(args.prompt, [], system_prompt=DEFAULT_SYSTEM_PROMPT)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to('xpu')
        # ipex_llm model needs a warmup, then inference time can be accurate
        output = model.generate(input_ids,
                                max_new_tokens=args.n_predict)

        # start inference
        st = time.time()
        # if your selected model is capable of utilizing previous key/value attentions
        # to enhance decoding speed, but has `"use_cache": false` in its model config,
        # it is important to set `use_cache=True` explicitly in the `generate` function
        # to obtain optimal performance with IPEX-LLM INT4 optimizations
        output = model.generate(input_ids,
                                max_new_tokens=args.n_predict)
        torch.xpu.synchronize()
        end = time.time()
        output = output.cpu()
        output_str = tokenizer.decode(output[0], skip_special_tokens=False)
        print(f'Inference time: {end-st} s')
        print('-'*20, 'Prompt', '-'*20)
        print(prompt)
        print('-'*20, 'Output', '-'*20)
        print(output_str)
