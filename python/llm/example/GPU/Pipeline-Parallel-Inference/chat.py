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
import time
from transformers import AutoTokenizer
from ipex_llm.transformers import AutoModelForCausalLM, init_pipeline_parallel

init_pipeline_parallel()
torch.manual_seed(1234)

if __name__ == '__main__':
   parser = argparse.ArgumentParser(description='Predict Tokens using `chat()` API for large vision language model')
   parser.add_argument('--repo-id-or-model-path', type=str, default="Qwen/Qwen-VL-Chat",
                       help='The huggingface repo id for the Qwen-VL-Chat model to be downloaded'
                            ', or the path to the huggingface checkpoint folder')
   parser.add_argument('--image-url-or-path', type=str,
                       default="http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg",
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

   # Load model
   # For successful IPEX-LLM optimization on Qwen-VL-Chat, skip the 'c_fc' and 'out_proj' modules during optimization
   # When running LLMs on Intel iGPUs for Windows users, we recommend setting `cpu_embedding=True` in the from_pretrained function.
   # This will allow the memory-intensive embedding layer to utilize the CPU instead of iGPU.
   model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                load_in_low_bit=args.low_bit,
                                                optimize_model=True,
                                                trust_remote_code=True,
                                                use_cache=True,
                                                torch_dtype=torch.float32,
                                                modules_to_not_convert=['c_fc', 'out_proj'],
                                                pipeline_parallel_stages=args.gpu_num)

   # Load tokenizer
   tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
   local_rank = torch.distributed.get_rank()

   all_input = [{'image': args.image_url_or_path}, {'text': args.prompt}]
   input_list = [_input for _input in all_input if list(_input.values())[0] != '']
   query = tokenizer.from_list_format(input_list)

   with torch.inference_mode():
      response, _ = model.chat(tokenizer, query=query, history=[])
      torch.xpu.synchronize()

      if local_rank == args.gpu_num - 1:
         print('-'*20, 'Input', '-'*20)
         print(f'Message: {all_input}')
         print('-'*20, 'Output', '-'*20)
         print(response)
