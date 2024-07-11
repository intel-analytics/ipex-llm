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
from ipex_llm.transformers import AutoModelForCausalLM, init_pipeline_parallel
from transformers import AutoTokenizer

init_pipeline_parallel()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for THUDM/glm-4v-9b model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="THUDM/glm-4v-9b",
                        help='The huggingface repo id for the THUDM/glm-4v-9b model to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--image-url-or-path', type=str,
                        default='http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg',
                        help='The URL or path to the image to infer')
    parser.add_argument('--prompt', type=str, default="这是什么？",
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=32,
                        help='Max tokens to predict')
    parser.add_argument('--low-bit', type=str, default='sym_int4', help='The quantization type the model will convert to.')
    parser.add_argument('--gpu-num', type=int, default=2, help='GPU number to use')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path
    image_path = args.image_url_or_path

    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 load_in_low_bit=args.low_bit,
                                                 optimize_model=True,
                                                 trust_remote_code=True,
                                                 use_cache=True,
                                                 pipeline_parallel_stages=args.gpu_num)
    model = model.half()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    local_rank = torch.distributed.get_rank()

    query = args.prompt
    if os.path.exists(image_path):
       image = Image.open(image_path)
    else:
       image = Image.open(requests.get(image_path, stream=True).raw)

    # here the prompt tuning refers to https://huggingface.co/THUDM/glm-4v-9b/blob/main/README.md
    inputs = tokenizer.apply_chat_template([{"role": "user", "image": image, "content": query}],
                                           add_generation_prompt=True,
                                           tokenize=True,
                                           return_tensors="pt",
                                           return_dict=True)  # chat mode
    inputs = inputs.to(f'xpu:{local_rank}')
    all_input = [{'image': image_path}, {'text': query}]

    # Generate predicted tokens
    with torch.inference_mode():
        gen_kwargs = {"max_new_tokens": args.n_predict, "do_sample": False,}
        st = time.time()
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        end = time.time()
        if local_rank == args.gpu_num - 1:
            print(f'Inference time: {end-st} s')
            output_str = tokenizer.decode(outputs[0])
            print('-'*20, 'Input', '-'*20)
            print(f'Message: {all_input}')
            print('-'*20, 'Output', '-'*20)
            print(output_str)
