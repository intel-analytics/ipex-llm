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
import time
import argparse
from ipex_llm.transformers.npu_model import AutoModelForCausalLM
from transformers import AutoTokenizer, TextStreamer
from transformers.utils import logging

logger = logging.get_logger(__name__)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict Tokens using `generate()` API for npu model"
    )
    parser.add_argument(
        "--repo-id-or-model-path",
        type=str,
        default="baichuan-inc/Baichuan2-7B-Chat",
        help="The huggingface repo id for the Baichuan2 model to be downloaded"
        ", or the path to the huggingface checkpoint folder",
    )
    parser.add_argument('--prompt', type=str, default="What is AI?",
                        help='Prompt to infer')
    parser.add_argument("--n-predict", type=int, default=32, help="Max tokens to predict")
    parser.add_argument("--max-context-len", type=int, default=1024)
    parser.add_argument("--max-prompt-len", type=int, default=512)
    parser.add_argument("--quantization_group_size", type=int, default=0)
    parser.add_argument("--disable-transpose-value-cache", action="store_true", default=False)
    parser.add_argument("--disable-streaming", action="store_true", default=False)
    parser.add_argument("--save-directory", type=str,
        required=True,
        help="The path of folder to save converted model, "
             "If path not exists, lowbit model will be saved there. "
             "Else, lowbit model will be loaded.",
    )

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path

    if not os.path.exists(args.save_directory):
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     optimize_model=True,
                                                     pipeline=True,
                                                     max_context_len=args.max_context_len,
                                                     max_prompt_len=args.max_prompt_len,
                                                     quantization_group_size=args.quantization_group_size,
                                                     torch_dtype=torch.float16,
                                                     attn_implementation="eager",
                                                     transpose_value_cache=not args.disable_transpose_value_cache,
                                                     trust_remote_code=True,
                                                     save_directory=args.save_directory)
    else:
        model = AutoModelForCausalLM.load_low_bit(
            args.save_directory,
            attn_implementation="eager",
            torch_dtype=torch.float16,
            max_context_len=args.max_context_len,
            max_prompt_len=args.max_prompt_len,
            pipeline=True,
            transpose_value_cache=not args.disable_transpose_value_cache,
            trust_remote_code=True
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if args.disable_streaming:
        streamer = None
    else:
        streamer = TextStreamer(tokenizer=tokenizer, skip_special_tokens=True)

    DEFAULT_SYSTEM_PROMPT = """\
    """

    print("-" * 80)
    print("done")
    with torch.inference_mode():
        print("finish to load")
        for i in range(3):
            prompt = get_prompt(args.prompt, [], system_prompt=DEFAULT_SYSTEM_PROMPT)
            _input_ids = tokenizer.encode(prompt, return_tensors="pt")
            print("-" * 20, "Input", "-" * 20)
            print("input length:", len(_input_ids[0]))
            print(prompt)
            print("-" * 20, "Output", "-" * 20)
            st = time.time()
            output = model.generate(
                _input_ids, max_new_tokens=args.n_predict, streamer=streamer
            )
            end = time.time()
            if args.disable_streaming:
                output_str = tokenizer.decode(output[0], skip_special_tokens=False)
                print(output_str)
            print(f"Inference time: {end-st} s")

    print("-" * 80)
    print("done")
    print("success shut down")
