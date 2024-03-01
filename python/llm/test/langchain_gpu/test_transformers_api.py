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

from bigdl.llm.langchain.llms import TransformersLLM, TransformersPipelineLLM, \
    LlamaLLM, BloomLLM
from bigdl.llm.langchain.embeddings import TransformersEmbeddings, LlamaEmbeddings, \
    BloomEmbeddings

import pytest
from unittest import TestCase
import os

device = os.environ['DEVICE']
print(f'Running on {device}')

class Test_Langchain_Transformers_API(TestCase):
    def setUp(self):
        self.auto_model_path = os.environ.get('ORIGINAL_CHATGLM2_6B_PATH')
        self.auto_causal_model_path = os.environ.get('ORIGINAL_REPLIT_CODE_PATH')
        self.llama_model_path = os.environ.get('LLAMA_ORIGIN_PATH')
        self.bloom_model_path = os.environ.get('BLOOM_ORIGIN_PATH')
        thread_num = os.environ.get('THREAD_NUM')
        if thread_num is not None:
            self.n_threads = int(thread_num)
        else:
            self.n_threads = 2         


    def test_bigdl_llm(self):
        texts = 'def hello():\n  print("hello world")\n'
        bigdl_llm = TransformersLLM.from_model_id(model_id=self.auto_causal_model_path, model_kwargs={'trust_remote_code': True, 'device_map': device})
        
        output = bigdl_llm(texts)
        res = "hello()" in output
        self.assertTrue(res)

if __name__ == '__main__':
    pytest.main([__file__])
