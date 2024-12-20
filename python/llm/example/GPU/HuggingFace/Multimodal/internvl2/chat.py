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
import torch
from PIL import Image
from ipex_llm.transformers import AutoModelForCausalLM
from transformers import CLIPImageProcessor


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `chat()` API for OpenGVLab/InternVL2-4B model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="OpenGVLab/InternVL2-4B",
                        help='The Hugging Face or ModelScope repo id for the InternVL2 model to be downloaded'
                             ', or the path to the checkpoint folder')
    parser.add_argument('--image-url-or-path', type=str,
                        default='https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg',
                        help='The URL or path to the image to infer')
    parser.add_argument('--prompt', type=str, default="What is in the image?",
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=64, help='Max tokens to predict')
    parser.add_argument('--modelscope', action="store_true", default=False, 
                        help="Use models from modelscope")

    args = parser.parse_args()

    if args.modelscope:
        from modelscope import AutoTokenizer
        model_hub = 'modelscope'
    else:
        from transformers import AutoTokenizer
        model_hub = 'huggingface'
    
    model_path = args.repo_id_or_model_path
    image_path = args.image_url_or_path
    n_predict = args.n_predict
    
    # Load model in 4 bit,
    # which convert the relevant layers in the model into INT4 format
    # When running LLMs on Intel iGPUs for Windows users, we recommend setting `cpu_embedding=True` in the from_pretrained function.
    # This will allow the memory-intensive embedding layer to utilize the CPU instead of iGPU.
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,
                                                 load_in_low_bit="sym_int4",
                                                 modules_to_not_convert=["vision_model"],
                                                 model_hub=model_hub)
    model = model.half().to('xpu')
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True)
    model.eval()

    query = args.prompt
    image_processor = CLIPImageProcessor.from_pretrained(model_path)

    if os.path.exists(image_path):
       image = Image.open(image_path).convert('RGB')
    else:
       image = Image.open(requests.get(image_path, stream=True).raw).convert('RGB')
       
    pixel_values = image_processor(images=[image], return_tensors='pt').pixel_values
    pixel_values = pixel_values.to('xpu')

    question = "<image>" + query

    generation_config = {
        "max_new_tokens": n_predict,
        "do_sample": False,
    }

    with torch.inference_mode():
        # ipex_llm model needs a warmup, then inference time can be accurate
        model.chat(
            pixel_values=None,
            question=question,
            generation_config=generation_config,
            tokenizer=tokenizer,
        )

        st = time.time()
        res = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=question,
            generation_config=generation_config,
            history=[]
        )
        torch.xpu.synchronize()
        end = time.time()

    print(f'Inference time: {end-st} s')
    print('-'*20, 'Input Image', '-'*20)
    print(image_path)
    print('-'*20, 'Input Prompt', '-'*20)
    print(args.prompt)
    print('-'*20, 'Chat Output', '-'*20)
    print(res)
