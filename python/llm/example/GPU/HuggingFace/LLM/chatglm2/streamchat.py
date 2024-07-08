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

from ipex_llm.transformers import AutoModel
from transformers import AutoTokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stream Chat for ChatGLM2 model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="THUDM/chatglm2-6b",
                        help='The huggingface repo id for the ChatGLM2 model to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--question', type=str, default="晚上睡不着应该怎么办",
                        help='Qustion you want to ask')
    parser.add_argument('--disable-stream', action="store_true",
                        help='Disable stream chat')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path
    disable_stream = args.disable_stream

    # Load model in 4 bit,
    # which convert the relevant layers in the model into INT4 format
    # When running LLMs on Intel iGPUs for Windows users, we recommend setting `cpu_embedding=True` in the from_pretrained function.
    # This will allow the memory-intensive embedding layer to utilize the CPU instead of iGPU.
    model = AutoModel.from_pretrained(model_path,
                                      load_in_4bit=True,
                                      trust_remote_code=True,
                                      optimize_model=False)
    model.to('xpu')

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True)

    with torch.inference_mode():
        prompt = args.question
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to('xpu')
        # ipex_llm model needs a warmup, then inference time can be accurate
        output = model.generate(input_ids,
                                max_new_tokens=32)

        # start inference
        if disable_stream:
            # Chat
            response, history = model.chat(tokenizer, args.question, history=[])
            print('-'*20, 'Chat Output', '-'*20)
            print(response)
        else:
            # Stream chat
            response_ = ""
            print('-'*20, 'Stream Chat Output', '-'*20)
            for response, history in model.stream_chat(tokenizer, args.question, history=[]):
                print(response.replace(response_, ""), end="")
                response_ = response
