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
from transformers.utils import logging

logger = logging.get_logger(__name__)

# you could tune the prompt based on your own model,
# here the prompt tuning refers to https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3
DEFAULT_SYSTEM_PROMPT = """\
"""

def get_prompt(user_input: str, chat_history: list[tuple[str, str]],
               system_prompt: str) -> str:
    prompt_texts = [f'<|begin_of_text|>']

    if system_prompt != '':
        prompt_texts.append(f'<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>')

    for history_input, history_response in chat_history:
        prompt_texts.append(f'<|start_header_id|>user<|end_header_id|>\n\n{history_input.strip()}<|eot_id|>')
        prompt_texts.append(f'<|start_header_id|>assistant<|end_header_id|>\n\n{history_response.strip()}<|eot_id|>')

    prompt_texts.append(f'<|start_header_id|>user<|end_header_id|>\n\n{user_input.strip()}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n')
    return ''.join(prompt_texts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict Tokens using `generate()` API for npu model"
    )
    parser.add_argument(
        "--repo-id-or-model-path",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="The huggingface repo id for the Llama3 model to be downloaded"
        ", or the path to the huggingface checkpoint folder",
    )
    parser.add_argument('--prompt', type=str, default="What is AI?",
                        help='Prompt to infer')
    parser.add_argument("--n-predict", type=int, default=32, help="Max tokens to predict")
    parser.add_argument("--max-output-len", type=int, default=1024)
    parser.add_argument("--max-prompt-len", type=int, default=960)
    parser.add_argument("--disable-transpose-value-cache", action="store_true", default=False)

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path

    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 torch_dtype=torch.float16,
                                                 optimize_model=True,
                                                 pipeline=True,
                                                 max_output_len=args.max_output_len,
                                                 max_prompt_len=args.max_prompt_len,
                                                 torch_dtype=torch.float16,
                                                 attn_implementation="eager",
                                                 transpose_value_cache=not args.disable_transpose_value_cache)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print("-" * 80)
    print("done")
    with torch.inference_mode():
        print("finish to load")
        for i in range(5):
            prompt = get_prompt(args.prompt, [], system_prompt=DEFAULT_SYSTEM_PROMPT)
            _input_ids = tokenizer.encode(prompt, return_tensors="pt")
            print("input length:", len(_input_ids[0]))
            st = time.time()
            output = model.generate(
                _input_ids, max_new_tokens=args.n_predict, do_print=True
            )
            end = time.time()
            print(f"Inference time: {end-st} s")
            input_str = tokenizer.decode(_input_ids[0], skip_special_tokens=False)
            print("-" * 20, "Input", "-" * 20)
            print(input_str)
            output_str = tokenizer.decode(output[0], skip_special_tokens=False)
            print("-" * 20, "Output", "-" * 20)
            print(output_str)

    print("-" * 80)
    print("done")
    print("success shut down")
