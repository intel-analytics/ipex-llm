
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
import intel_extension_for_pytorch as ipex
import time
import argparse

from ipex_llm.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

# you could tune the prompt based on your own model,
# here the prompt tuning refers to https://huggingface.co/georgesung/llama2_7b_chat_uncensored#prompt-style
DEFAULT_SYSTEM_PROMPT = """\
"""

def get_prompt(message: str, chat_history: list[tuple[str, str]],
               system_prompt: str) -> str:
    texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
    # The first user input is _not_ stripped
    do_strip = False
    for user_input, response in chat_history:
        user_input = user_input.strip() if do_strip else user_input
        do_strip = True
        texts.append(f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
    message = message.strip() if do_strip else message
    texts.append(f'{message} [/INST]')
    return ''.join(texts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for Llama2 model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="meta-llama/Llama-2-7b-chat-hf",
                        help='The huggingface repo id for the Llama2 (e.g. `meta-llama/Llama-2-7b-chat-hf` and `meta-llama/Llama-2-13b-chat-hf`) to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default="What is AI?",
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=32,
                        help='Max tokens to predict')
    parser.add_argument('--gpu-num', type=int, default=2, help='GPU number to use')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path

    # Load model in 4 bit,
    # which convert the relevant layers in the model into INT4 format
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 load_in_4bit=True,
                                                 optimize_model=True,
                                                 trust_remote_code=True,
                                                 use_cache=True)

    model_layers = ['model.embed_tokens']
    for i in range(model.config.num_hidden_layers):
        model_layers.append(f'model.layers.{i}')
    model_layers = model_layers + ['model.norm', 'lm_head']

    device_map = {}
    split_len = len(model_layers) // args.gpu_num
    for i in range(args.gpu_num):
        device_map.update({key: f'xpu:{i}' for key in model_layers[split_len * i: split_len * (i + 1)]})
        if i == args.gpu_num - 1:
            device_map.update({key: f'xpu:{i}' for key in model_layers[split_len * (i + 1): ]})

    from accelerate import dispatch_model
    model = dispatch_model(
        model,
        device_map=device_map,
        offload_dir=None,
        skip_keys=["past_key_value", "past_key_values"],
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Generate predicted tokens
    with torch.inference_mode():
        prompt = get_prompt(args.prompt, [], system_prompt=DEFAULT_SYSTEM_PROMPT)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to('xpu:0')
        # ipex_llm model needs a warmup, then inference time can be accurate
        output = model.generate(input_ids,
                                do_sample=False,
                                max_new_tokens=args.n_predict)
        output = model.generate(input_ids,
                                do_sample=False,
                                max_new_tokens=args.n_predict)

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
        output = output.cpu()
        output_str = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f'Inference time: {end-st} s')
        print('-'*20, 'Prompt', '-'*20)
        print(prompt)
        print('-'*20, 'Output', '-'*20)
        print(output_str)

