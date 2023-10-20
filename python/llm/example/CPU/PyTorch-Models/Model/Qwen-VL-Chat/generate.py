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
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import time
import os
import argparse
from bigdl.llm import optimize_model
torch.manual_seed(1234)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `chat()` API for Qw-VL-Chat model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="Qwen/Qwen_VL_Chat",
                        help='The huggingface repo id for the Qwen_VL_Chat model to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--n-predict', type=int, default=32, help='Max tokens to predict')
    
    current_path = os.path.dirname(os.path.abspath(__file__))
    args = parser.parse_args()
    model_path = args.repo_id_or_model_path
    sys.path.append(model_path)

    try:
       from qwen_generation_utils import *

    except ImportError:
       print('无法导入 qwen_generation_util模块')

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu",  trust_remote_code=True).eval()

    # With only one line to enable BigDL-LLM optimization on model
    # Note that currently there is no optimization support 'c_fc' and 'out_proj' moudule, 'modules_to_not_convert' should be set
    model = optimize_model(model, low_bit='sym_int4', modules_to_not_convert=['c_fc', 'out_proj'] , optimize_llm = True)

    # Specify hyperparameters for generation (No need to do this if you are using transformers>=4.32.0)
    model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Session ID
    session_id = 1

    while True:
      print('-'*20, 'Session %d' % session_id, '-'*20)
      image_input = input(f' 请输入你想询问的图片: ')
      if image_input.lower() == 'exit' : # type 'exit' to quit the dialouge
         break

      text_input = input(f' 请输入你想询问的文字信息: ')
      if text_input.lower() == 'exit' : # type 'exit' to quit the dialouge
         break
      
      if session_id == 1:
         history = []
         stop_words_ids = []
         max_window_size = model.generation_config.max_window_size

      all_input = [{'image': image_input}, {'text': text_input}]
      input_list = [_input for _input in all_input if list(_input.values())[0] != '']

      if len(input_list) == 0:
         print("请输入有效图片或文字或prompt")
         continue
      
      query = tokenizer.from_list_format(input_list)
      
      # use `generate` API to generate response
      raw_text, context_tokens = make_context(
                                             tokenizer,
                                             query,
                                             history=history,
                                             system=" You are a helpful assistant.",
                                             max_window_size=max_window_size,
                                             chat_format=model.generation_config.chat_format)
      
      input_ids = torch.tensor([context_tokens])

      stop_words_ids.extend(get_stop_words_ids(
                                               model.generation_config.chat_format, tokenizer))
      
      outputs = model.generate(
                              input_ids,
                              stop_words_ids=stop_words_ids,
                              return_dict_in_generate=False,
                              generation_config=model.generation_config,
                              )
      
      response = decode_tokens(
                              outputs[0],
                              tokenizer,
                              raw_text_len=len(raw_text),
                              context_length=len(context_tokens),
                              chat_format=model.generation_config.chat_format,
                              verbose=False,
                              errors='replace'
                              )
      
      history.append((query, response))
      
      print('-'*10, 'Response', '-'*10)
      print(response, '\n')

      image = tokenizer.draw_bbox_on_latest_picture(response, history)
      if image is not None:
         image.save(os.path.join(current_path, f'Session_{session_id}.png'), )

      session_id += 1