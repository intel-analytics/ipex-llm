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
import pytest
from unittest import TestCase
import os

import time
import torch
from bigdl.llm.transformers import AutoModelForCausalLM, AutoModel
from transformers import LlamaTokenizer, AutoTokenizer

class TestTransformersAPI(TestCase):

    def setUp(self):
        self.llama_model_path = os.environ.get('LLAMA_INT4_CKPT_PATH')
        self.bloom_model_path = os.environ.get('BLOOM_INT4_CKPT_PATH')
        self.gptneox_model_path = os.environ.get('GPTNEOX_INT4_CKPT_PATH')
        self.starcoder_model_path = os.environ.get('STARCODER_INT4_CKPT_PATH')
        thread_num = os.environ.get('THREAD_NUM')
        if thread_num is not None:
            self.n_threads = int(thread_num)
        else:
            self.n_threads = 2

    def test_llama_completion_success(self):
        model_path = self.llama_model_path
        model = AutoModelForCausalLM.from_pretrained(model_path, load_in_4bit=True)
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
    
        input_str = "What is the capital of France?"

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
    pytest.main([__file__])
