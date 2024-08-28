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
import argparse
import requests
from PIL import Image
from ipex_llm.transformers import AutoModel
from transformers import AutoTokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for openbmb/MiniCPM-Llama3-V-2_5 model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="openbmb/MiniCPM-Llama3-V-2_5",
                        help='The huggingface repo id for the openbmb/MiniCPM-Llama3-V-2_5 model to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--image-url-or-path', type=str,
                        default='http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg',
                        help='The URL or path to the image to infer')
    parser.add_argument('--prompt', type=str, default="What is in the image?",
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=32,
                        help='Max tokens to predict')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path
    image_path = args.image_url_or_path
    
    # Load model in 4 bit,
    # which convert the relevant layers in the model into INT4 format
    # When running LLMs on Intel iGPUs for Windows users, we recommend setting `cpu_embedding=True` in the from_pretrained function.
    # This will allow the memory-intensive embedding layer to utilize the CPU instead of iGPU.
    model = AutoModel.from_pretrained(model_path, 
                                      load_in_4bit=True,
                                      optimize_model=False,
                                      trust_remote_code=True,
                                      use_cache=True)
    model = model.half().to(device='xpu')
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True)
    model.eval()

    query = args.prompt
    if os.path.exists(image_path):
       image = Image.open(image_path).convert('RGB')
    else:
       image = Image.open(requests.get(image_path, stream=True).raw).convert('RGB')

    # Generate predicted tokens
    # here the prompt tuning refers to https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5/blob/main/README.md
    msgs = [{'role': 'user', 'content': args.prompt}]
    st = time.time()
    res = model.chat(
     image=image,
     msgs=msgs,
     context=None,
     tokenizer=tokenizer,
     sampling=False,
     temperature=0.7
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
