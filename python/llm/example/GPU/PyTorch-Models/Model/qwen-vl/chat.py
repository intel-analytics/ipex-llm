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

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

from ipex_llm import optimize_model

torch.manual_seed(1234)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `chat()` API for Qwen-VL model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="Qwen/Qwen-VL-Chat",
                        help='The huggingface repo id for the Qwen-VL model to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--n-predict', type=int, default=32, help='Max tokens to predict')
    
    current_path = os.path.dirname(os.path.abspath(__file__))
    args = parser.parse_args()
    model_path = args.repo_id_or_model_path  
        
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu", trust_remote_code=True)

    # With only one line to enable IPEX-LLM optimization on model
    # For successful IPEX-LLM optimization on Qwen-VL-Chat, skip the 'c_fc' and 'out_proj' modules during optimization
    # When running LLMs on Intel iGPUs for Windows users, we recommend setting `cpu_embedding=True` in the optimize_model function.
    # This will allow the memory-intensive embedding layer to utilize the CPU instead of iGPU.
    model = optimize_model(model, 
                           low_bit='sym_int4', 
                           modules_to_not_convert=['c_fc', 'out_proj'])
    model = model.to('xpu')

    # Specify hyperparameters for generation (No need to do this if you are using transformers>=4.32.0)
    model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Session ID
    session_id = 1

    while True:
      print('-'*20, 'Session %d' % session_id, '-'*20)
      image_input = input(f' Please input a picture: ')
      if image_input.lower() == 'exit' : # type 'exit' to quit the dialouge
         break

      text_input = input(f' Please enter the text: ')
      if text_input.lower() == 'exit' : # type 'exit' to quit the dialouge
         break
      
      if session_id == 1:
         history = None

      all_input = [{'image': image_input}, {'text': text_input}]
      input_list = [_input for _input in all_input if list(_input.values())[0] != '']

      if len(input_list) == 0:
         print("Input list should not be empty. Please try again with valid input.")
         continue
      
      query = tokenizer.from_list_format(input_list)
      response, history = model.chat(tokenizer, query = query, history = history)
      torch.xpu.synchronize()

      print('-'*10, 'Response', '-'*10)
      print(response, '\n')

      image = tokenizer.draw_bbox_on_latest_picture(response, history)
      if image is not None:
         image.save(os.path.join(current_path, f'Session_{session_id}.png'), )

      session_id += 1
