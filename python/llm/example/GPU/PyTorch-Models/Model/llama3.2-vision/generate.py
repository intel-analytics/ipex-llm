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
from transformers import MllamaForConditionalGeneration, AutoProcessor

from ipex_llm import optimize_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for Llama3.2-Vision model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="meta-llama/Llama-3.2-11B-Vision-Instruct",
                        help='The huggingface repo id for the Llama3.2-Vision model to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--image-url-or-path', type=str,
                        default='https://hf-mirror.com/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg',                        
                        help='The URL or path to the image to infer')
    parser.add_argument('--prompt', type=str, default="Describe image in detail",
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=32,
                        help='Max tokens to predict')
    
    args = parser.parse_args()
    model_path = args.repo_id_or_model_path
    image_path = args.image_url_or_path
    prompt = args.prompt

    model = MllamaForConditionalGeneration.from_pretrained(model_path)
    model = optimize_model(model, modules_to_not_convert=["multi_modal_projector"])
    model = model.half().eval()
    model = model.to('xpu')

    processor = AutoProcessor.from_pretrained(model_path)

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

    inputs = processor(text=text, images=image, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        for i in range(3):
            st = time.time()
            output = model.generate(**inputs, do_sample=False, max_new_tokens=args.n_predict)
            et = time.time()
            print(et - st)
    print(processor.decode(output[0]))
