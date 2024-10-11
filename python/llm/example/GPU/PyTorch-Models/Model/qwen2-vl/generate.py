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
import numpy as np

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from ipex_llm import optimize_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using generate() API for Qwen2-VL-7B-Instruct model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="Qwen/Qwen2-VL-7B-Instruct",
                        help='The huggingface repo id for the Qwen2-VL model to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default="Describe this image.",
                        help='Prompt to infer') 
    parser.add_argument('--image-url-or-path', type=str,
                        default='http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg' ,
                        help='The URL or path to the image to infer')
    
    parser.add_argument('--n-predict', type=int, default=32,
                        help='Max tokens to predict')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path

    model = Qwen2VLForConditionalGeneration.from_pretrained(model_path,
                                                 trust_remote_code=True,
                                                 torch_dtype='auto',
                                                 low_cpu_mem_usage=True,
                                                 use_cache=True,)

    model = optimize_model(model, low_bit='sym_int4', modules_to_not_convert=["visual"])

    # Use .float() for better output, and use .half() for better speed
    model = model.half().to("xpu")

    # The following code for generation is adapted from https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct#quickstart

    # The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
    min_pixels = 256*28*28
    max_pixels = 1280*28*28
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)

    prompt = args.prompt
    image_path = args.image_url_or_path

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to('xpu')

    with torch.inference_mode():
        # warmup
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.n_predict
            )

        st = time.time()
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.n_predict
            )
        torch.xpu.synchronize()
        end = time.time()
        generated_ids = generated_ids.cpu()
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]

        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f'Inference time: {end-st} s')
        print('-'*20, 'Input Image', '-'*20)
        print(image_path)
        print('-'*20, 'Prompt', '-'*20)
        print(prompt)
        print('-'*20, 'Output', '-'*20)
        print(response)
