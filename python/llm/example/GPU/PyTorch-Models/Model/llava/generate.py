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

import argparse
import os

import requests
import time
import torch
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor

from ipex_llm import optimize_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for LLaVA model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="llava-hf/llava-1.5-7b-hf",
                        help='The huggingface repo id for the LLaVA model to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--image-url-or-path', type=str,
                        default='http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg',                        
                        help='The URL or path to the image to infer')
    parser.add_argument('--prompt', type=str, default="Describe image in detail",
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=32,
                        help='Max tokens to predict') 
    args = parser.parse_args()
    model_path = args.repo_id_or_model_path
    image_path = args.image_url_or_path
    prompt = args.prompt

    model = LlavaForConditionalGeneration.from_pretrained(model_path)
    model = optimize_model(model, low_bit='sym_int4').eval()
    model = model.half().to("xpu")

    processor = AutoProcessor.from_pretrained(model_path)

    # here the prompt tuning refers to https://huggingface.co/llava-hf/llava-1.5-7b-hf#using-pure-transformers
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)

    if os.path.exists(image_path):
       image = Image.open(image_path)
    else:
       image = Image.open(requests.get(image_path, stream=True).raw)

    inputs = processor(text=text, images=image, return_tensors="pt").to('xpu')

    with torch.inference_mode():
        # warmup
        output = model.generate(**inputs, do_sample=False, max_new_tokens=args.n_predict)

        # start inference
        st = time.time()
        output = model.generate(**inputs, do_sample=False, max_new_tokens=args.n_predict)
        et = time.time()

    output_str = processor.decode(output[0])
    print(f'Inference time: {et-st} s')
    print('-'*20, 'Input Image', '-'*20)
    print(image_path)
    print('-'*20, 'Prompt', '-'*20)
    print(prompt)
    print('-'*20, 'Output', '-'*20)
    print(output_str)
