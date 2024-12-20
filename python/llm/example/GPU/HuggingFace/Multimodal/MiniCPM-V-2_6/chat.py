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
from ipex_llm.transformers import AutoModel
from transformers import AutoProcessor


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `chat()` API for openbmb/MiniCPM-V-2_6 model')
    parser.add_argument('--repo-id-or-model-path', type=str,
                        help='The Hugging Face or ModelScope repo id for the MiniCPM-V-2_6 model to be downloaded'
                             ', or the path to the checkpoint folder')
    parser.add_argument("--lowbit-path", type=str,
        default="",
        help="The path to the saved model folder with IPEX-LLM low-bit optimization. "
             "Leave it blank if you want to load from the original model. "
             "If the path does not exist, model with low-bit optimization will be saved there."
             "Otherwise, model with low-bit optimization will be loaded from the path.",
    )
    parser.add_argument('--image-url-or-path', type=str,
                        default='http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg',
                        help='The URL or path to the image to infer')
    parser.add_argument('--prompt', type=str, default="What is in the image?",
                        help='Prompt to infer')
    parser.add_argument('--stream', action='store_true',
                        help='Whether to chat in streaming mode')
    parser.add_argument('--modelscope', action="store_true", default=False, 
                        help="Use models from modelscope")

    args = parser.parse_args()

    if args.modelscope:
        from modelscope import AutoTokenizer
        model_hub = 'modelscope'
    else:
        from transformers import AutoTokenizer
        model_hub = 'huggingface'
    
    model_path = args.repo_id_or_model_path if args.repo_id_or_model_path else \
        ("OpenBMB/MiniCPM-V-2_6" if args.modelscope else "openbmb/MiniCPM-V-2_6")
    image_path = args.image_url_or_path

    lowbit_path = args.lowbit_path
    
    if not lowbit_path or not os.path.exists(lowbit_path):
        # Load model in 4 bit,
        # which convert the relevant layers in the model into INT4 format
        # When running LLMs on Intel iGPUs for Windows users, we recommend setting `cpu_embedding=True` in the from_pretrained function.
        # This will allow the memory-intensive embedding layer to utilize the CPU instead of iGPU.
        model = AutoModel.from_pretrained(model_path, 
                                        load_in_low_bit="sym_int4",
                                        optimize_model=True,
                                        trust_remote_code=True,
                                        use_cache=True,
                                        modules_to_not_convert=["vpm", "resampler"],
                                        model_hub=model_hub)

        tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                  trust_remote_code=True)
    else:
        model = AutoModel.load_low_bit(lowbit_path, 
                                       optimize_model=True,
                                       trust_remote_code=True,
                                       use_cache=True,
                                       modules_to_not_convert=["vpm", "resampler"])
        tokenizer = AutoTokenizer.from_pretrained(lowbit_path,
                                                  trust_remote_code=True)
    
    model.eval()

    if lowbit_path and not os.path.exists(lowbit_path):
        processor = AutoProcessor.from_pretrained(model_path,
                                                trust_remote_code=True)
        model.save_low_bit(lowbit_path)
        tokenizer.save_pretrained(lowbit_path)
        processor.save_pretrained(lowbit_path)

    model = model.half().to('xpu')

    query = args.prompt
    if os.path.exists(image_path):
       image = Image.open(image_path).convert('RGB')
    else:
       image = Image.open(requests.get(image_path, stream=True).raw).convert('RGB')

    # Generate predicted tokens
    # here the prompt tuning refers to https://huggingface.co/openbmb/MiniCPM-V-2_6/blob/main/README.md
    msgs = [{'role': 'user', 'content': [image, args.prompt]}]

    # ipex_llm model needs a warmup, then inference time can be accurate
    model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tokenizer,
    )

    if args.stream:
        res = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer,
            stream=True
        )

        print('-'*20, 'Input Image', '-'*20)
        print(image_path)
        print('-'*20, 'Input Prompt', '-'*20)
        print(args.prompt)
        print('-'*20, 'Stream Chat Output', '-'*20)
        for new_text in res:
            print(new_text, flush=True, end='')
    else:
        st = time.time()
        res = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer,
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
