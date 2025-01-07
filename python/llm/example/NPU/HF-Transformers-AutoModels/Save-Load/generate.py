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
from ipex_llm.transformers.npu_model import AutoModelForCausalLM
from transformers import AutoTokenizer
from ipex_llm.utils.common.log4Error import invalidInputError


# you could tune the prompt based on your own model,
LLAMA2_PROMPT_FORMAT = """<s> [INST] <<SYS>>

<</SYS>>

{prompt} [/INST]
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of saving and loading the optimized model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="meta-llama/Llama-2-7b-chat-hf",
                        help='The huggingface repo id for the Llama2 (e.g. `meta-llama/Llama-2-7b-chat-hf` and `meta-llama/Llama-2-13b-chat-hf`) to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--save-directory', type=str, default=None,
                        help='The path to save the low-bit model.')
    parser.add_argument('--load-directory', type=str, default=None,
                        help='The path to load the low-bit model.')
    parser.add_argument('--prompt', type=str, default="What is AI?",
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=32,
                        help='Max tokens to predict')
    parser.add_argument("--max-context-len", type=int, default=1024)
    parser.add_argument("--max-prompt-len", type=int, default=512)
    parser.add_argument('--low-bit', type=str, default="sym_int4",
                        help='Low bit optimizations that will be applied to the model.')
    
    args = parser.parse_args()
    model_path = args.repo_id_or_model_path
    save_directory = args.save_directory
    load_directory = args.load_directory

    if save_directory:
        # first time to load and save
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            attn_implementation="eager",
            load_in_low_bit=args.low_bit,
            optimize_model=True,
            max_context_len=args.max_context_len,
            max_prompt_len=args.max_prompt_len,
            save_directory=save_directory
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.save_pretrained(save_directory)
        print(f"Finish to load model from {model_path} and save to {save_directory}")
    elif load_directory:
        # load low-bit model
        model = AutoModelForCausalLM.load_low_bit(
            load_directory,
            attn_implementation="eager",
            torch_dtype=torch.float16,
            optimize_model=True,
            max_context_len=args.max_context_len,
            max_prompt_len=args.max_prompt_len
        )
        tokenizer = AutoTokenizer.from_pretrained(load_directory, trust_remote_code=True)
        print(f"Finish to load model from {load_directory}")
    else:
        invalidInputError(False,
                          "Both `--save-directory` and `--load-directory` are None, please provide one of this.")

    # Generate predicted tokens
    with torch.inference_mode():
        for i in range(3):
            prompt = LLAMA2_PROMPT_FORMAT.format(prompt=args.prompt)
            _input_ids = tokenizer.encode(prompt, return_tensors="pt")

            st = time.time()
            output = model.generate(
                _input_ids, num_beams=1, do_sample=False, max_new_tokens=args.n_predict
            )
            end = time.time()

            print(f"Inference time: {end-st} s")
            input_str = tokenizer.decode(_input_ids[0], skip_special_tokens=False)
            print("-" * 20, "Input", "-" * 20)
            print(input_str)
            output_str = tokenizer.decode(output[0], skip_special_tokens=False)
            print("-" * 20, "Output", "-" * 20)
            print(output_str)
