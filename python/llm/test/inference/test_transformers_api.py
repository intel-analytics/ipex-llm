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


from bigdl.llm.models import Llama, Bloom, Gptneox, Starcoder
from bigdl.llm.utils import get_avx_flags
import unittest
import os

import time
import torch
from bigdl.llm.transformers import AutoModelForCausalLM, AutoModel
from transformers import LlamaTokenizer, AutoTokenizer

class TestTransformersAPI(unittest.TestCase):

    def setUp(self):        
        thread_num = os.environ.get('THREAD_NUM')
        if thread_num is not None:
            self.n_threads = int(thread_num)
        else:
            self.n_threads = 2

    def test_transformers_int4(self):
        model_path = os.environ.get('ORIGINAL_CHATGLM_6B_PATH')
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, load_in_4bit=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        input_str = "晚上睡不着应该怎么办"

        with torch.inference_mode():
            st = time.time()
            input_ids = tokenizer.encode(input_str, return_tensors="pt")
            output = model.generate(input_ids, do_sample=False, max_new_tokens=32)
            output_str = tokenizer.decode(output[0], skip_special_tokens=True)
            end = time.time()
        print('Prompt:', input_str)
        print('Output:', output_str)
        print(f'Inference time: {end-st} s')    


if __name__ == '__main__':
    unittest.main()
