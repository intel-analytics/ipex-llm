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
import time
import torch
import argparse
import requests

from PIL import Image
from ipex_llm.transformers import AutoModelForCausalLM
from transformers import AutoProcessor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for phi-3 model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="microsoft/Phi-3-vision-128k-instruct",
                        help='The huggingface repo id for the phi-3-vision model to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--image-url-or-path', type=str,
                        default="http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg",
                        help='The URL or path to the image to infer')
    parser.add_argument('--prompt', type=str, default="What is in the image?",
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=32,
                        help='Max tokens to predict')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path
    image_path = args.image_url_or_path

    # Load model in INT8,
    # which convert the relevant layers in the model into INT8 format
    # We here use INT8 instead of INT4 for better output
    # `_attn_implementation="eager"` is required for phi-3-vision
    # `modules_to_not_convert=["vision_embed_tokens"]` is for acceleration and is optional
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 trust_remote_code=True,
                                                 load_in_low_bit="sym_int8",
                                                 _attn_implementation="eager",
                                                 modules_to_not_convert=["vision_embed_tokens"])
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # here the message formatting refers to https://huggingface.co/microsoft/Phi-3-vision-128k-instruct#sample-inference-code
    messages = [
        {"role": "user", "content": "<|image_1|>\n{prompt}".format(prompt=args.prompt)},
    ]
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if os.path.exists(image_path):
       image = Image.open(image_path)
    else:
       image = Image.open(requests.get(image_path, stream=True).raw)
    
    # Generate predicted tokens
    with torch.inference_mode():
        inputs = processor(prompt, [image], return_tensors="pt")
        st = time.time()
        output = model.generate(**inputs,
                                eos_token_id=processor.tokenizer.eos_token_id,
                                num_beams=1,
                                do_sample=False,
                                max_new_tokens=args.n_predict,
                                temperature=0.0)
        end = time.time()
        print(f'Inference time: {end-st} s')
        output_str = processor.decode(output[0],
                                      skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)
        print('-'*20, 'Prompt', '-'*20)
        print(f'Message: {messages}')
        print(f'Image link/path: {image_path}')
        print('-'*20, 'Output', '-'*20)
        print(output_str)
