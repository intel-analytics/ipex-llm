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

from ipex_llm.transformers.npu_model import AutoModel, AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers.utils import logging

import requests
from PIL import Image

logger = logging.get_logger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict Tokens using `chat()` API for npu model"
    )
    parser.add_argument(
        "--repo-id-or-model-path",
        type=str,
        default="openbmb/MiniCPM-Llama3-V-2_5",
        help="The huggingface repo id for the MiniCPM-Llama3-V-2_5 model to be downloaded"
        ", or the path to the huggingface checkpoint folder.",
    )
    parser.add_argument('--image-url-or-path', type=str,
                        default='http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg',
                        help='The URL or path to the image to infer')
    parser.add_argument('--prompt', type=str, default="What is in this image?",
                        help='Prompt to infer')
    parser.add_argument("--n-predict", type=int, default=32, help="Max tokens to predict.")
    parser.add_argument("--max-context-len", type=int, default=1024)
    parser.add_argument("--max-prompt-len", type=int, default=960)
    parser.add_argument('--low-bit', type=str, default="sym_int4",
                        help='Low bit optimizations that will be applied to the model.')
    parser.add_argument("--save-directory", type=str,
        required=True,
        help="The path of folder to save converted model, "
             "If path not exists, lowbit model will be saved there. "
             "Else, lowbit model will be loaded.",
    )
    args = parser.parse_args()
    model_path = args.repo_id_or_model_path

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager",
        load_in_low_bit=args.low_bit,
        optimize_model=True,
        max_context_len=args.max_context_len,
        max_prompt_len=args.max_prompt_len,
        save_directory=args.save_directory
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print("-" * 80)
    print("done")

    msgs = [{'role': 'user', 'content': args.prompt}]
    image_path = args.image_url_or_path
    if os.path.exists(image_path):
       image = Image.open(image_path).convert('RGB')
    else:
       image = Image.open(requests.get(image_path, stream=True).raw).convert('RGB')

    st = time.time()
    res = model.chat(
        image=image,
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=True,
        temperature=0.7,
        # system_prompt='' # pass system_prompt if needed
        )
    end = time.time()

    print(f'Inference time: {end-st} s')
    print('-'*20, 'Input', '-'*20)
    print(image_path)
    print('-'*20, 'Prompt', '-'*20)
    print(args.prompt)
    output_str = res
    print('-'*20, 'Output', '-'*20)
    print(output_str)

    print("done")
    print("success shut down")
