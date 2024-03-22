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
    parser = argparse.ArgumentParser(description='Stream Chat for ChatGLM3 model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="THUDM/chatglm3-6b",
                        help='The huggingface repo id for the ChatGLM3 model to be downloaded'
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
    model = AutoModel.from_pretrained(model_path,
                                      load_in_4bit=True,
                                      trust_remote_code=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True)

    with torch.inference_mode():
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
