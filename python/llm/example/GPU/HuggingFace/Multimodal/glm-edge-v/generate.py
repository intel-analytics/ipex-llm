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
import requests

from ipex_llm.transformers import AutoModelForCausalLM

from PIL import Image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for glm-edge-v model')
    parser.add_argument('--repo-id-or-model-path', type=str,
                        help='The Hugging Face or ModelScope repo id for the glm-edge-v model to be downloaded'
                             ', or the path to the checkpoint folder')
    parser.add_argument('--image-url-or-path', type=str,
                        default='http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg',
                        help='The URL or path to the image to infer')
    parser.add_argument('--prompt', type=str, default="What is in the image?",
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=32,
                        help='Max tokens to predict')
    parser.add_argument('--modelscope', action="store_true", default=False, 
                        help="Use models from modelscope")
    
    args = parser.parse_args()

    if args.modelscope:
        from modelscope import AutoTokenizer, AutoImageProcessor
        model_hub = 'modelscope'
    else:
        from transformers import AutoTokenizer, AutoImageProcessor
        model_hub = 'huggingface'
    
    model_path = args.repo_id_or_model_path if args.repo_id_or_model_path else \
        ("ZhipuAI/glm-edge-v-5b" if args.modelscope else "THUDM/glm-edge-v-5b")
    image_path = args.image_url_or_path

    # Load model in 4 bit,
    # which convert the relevant layers in the model into INT4 format
    # When running LLMs on Intel iGPUs for Windows users, we recommend setting `cpu_embedding=True` in the from_pretrained function.
    # This will allow the memory-intensive embedding layer to utilize the CPU instead of iGPU.
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 load_in_4bit=True,
                                                 optimize_model=True,
                                                 trust_remote_code=True,
                                                 modules_to_not_convert=["vision"],
                                                 use_cache=True,
                                                 model_hub=model_hub)
    model = model.half().to('xpu')
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    query = args.prompt
    if os.path.exists(image_path):
       image = Image.open(image_path)
    else:
       image = Image.open(requests.get(image_path, stream=True).raw)

    image_processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)

    with torch.inference_mode():
        pixel_values = image_processor(images=[image], return_tensors='pt').pixel_values
        pixel_values = pixel_values.to('xpu')

        # The following code for generation is adapted from https://huggingface.co/THUDM/glm-edge-v-5b#inference
        messages = [{
            "role": "user", 
            "content": [{"type": "image"}, 
                        {"type": "text", 
                         "text": args.prompt}]
        }]

        inputs = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_dict=True, 
            tokenize=True, 
            return_tensors="pt"
        )
        inputs = inputs.to('xpu')
        
        generate_kwargs = {
            **inputs,
            "pixel_values": pixel_values,
            "max_new_tokens": args.n_predict,
        }
        
        # ipex_llm model needs a warmup, then inference time can be accurate
        output = model.generate(**generate_kwargs)

        st = time.time()
        output = model.generate(**generate_kwargs)
        torch.xpu.synchronize()
        end = time.time()

        output_str = tokenizer.decode(
            output[0][len(inputs["input_ids"][0]):], 
            skip_special_tokens=True
        )
        
        print(f'Inference time: {end-st} s')
        print('-'*20, 'Prompt', '-'*20)
        print(args.prompt)
        print('-'*20, 'Output', '-'*20)
        print(output_str)
