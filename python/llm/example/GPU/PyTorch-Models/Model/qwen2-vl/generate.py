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

from transformers import AutoTokenizer
from ipex_llm import optimize_model
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Qwen2-VL-7B-Instruct')
    parser.add_argument('--repo-id-or-model-path', type=str, default="Qwen/Qwen2-VL-7B-Instruct",
                        help='The huggingface repo id for the Qwen2-VL model to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default="Describe this image.",
                        help='Prompt to infer') 
    parser.add_argument('--n-predict', type=int, default=32,
                        help='Max tokens to predict')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path


    from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
    from qwen_vl_utils import process_vision_info
    from ipex_llm import optimize_model
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_path,
                                                 trust_remote_code=True,
                                                 torch_dtype = 'auto',
                                                 use_cache=True)
    model = optimize_model(model, low_bit='sym_int4')
    model = model.to("xpu")

    # Load processor
    min_pixels = 256*28*28
    max_pixels = 1280*28*28
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)

    prompt = args.prompt

    # The following code for generation is adapted from https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct#quickstart
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
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
        print('-'*20, 'Prompt', '-'*20)
        print(prompt)
        print('-'*20, 'Output', '-'*20)
        print(response)