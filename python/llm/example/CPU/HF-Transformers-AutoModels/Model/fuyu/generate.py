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

from transformers import FuyuProcessor
import torch
import argparse
import time
from PIL import Image
from ipex_llm.transformers import AutoModelForCausalLM

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for Fuyu model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="adept/fuyu-8b",
                        help='The huggingface repo id for the Fuyu model to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default="Generate a coco-style caption.",
                        help='Prompt to infer')
    parser.add_argument('--image-path', type=str, required=True,
                        help='Image path for the input image that the chat will focus on')
    parser.add_argument('--n-predict', type=int, default=512, help='Max tokens to predict')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path
    prompt = args.prompt
    image = Image.open(args.image_path)

    # Load model
    # For successful IPEX-LLM optimization on Fuyu, skip the 'vision_embed_tokens' module during optimization
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='cpu',
                                                 load_in_4bit = True,
                                                 trust_remote_code=True,
                                                 modules_to_not_convert=['vision_embed_tokens'])

    # Load processor
    processor = FuyuProcessor.from_pretrained(model_path)

    # Generate predicted tokens
    with torch.inference_mode():
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        st = time.time()
        generation_outputs = model.generate(**inputs,
                                max_new_tokens=args.n_predict)
        end = time.time()
        outputs = processor.batch_decode(generation_outputs[:, -args.n_predict:], skip_special_tokens=True)
        print(f'Inference time: {end-st} s')
        print('-'*20, 'Prompt', '-'*20)
        print(prompt)
        print('-'*20, 'Output', '-'*20)
        for output in outputs:
            # '\x04' is the "beginning of answer" token
            # See https://huggingface.co/adept/fuyu-8b#how-to-use
            answer = output.split('\x04 ', 1)[1] if '\x04' in output else ''
            print(answer)
