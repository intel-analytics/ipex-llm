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

from bigdl.llm.utils import get_avx_flags
from bigdl.llm.langchain.embeddings import BigdlLLMEmbeddings
from bigdl.llm.langchain.llms import BigdlLLM
import pytest
from unittest import TestCase
import os


class Test_Models_Basics(TestCase):
    def setUp(self):
        self.llama_model_path = os.environ.get('LLAMA_INT4_CKPT_PATH')
        self.bloom_model_path = os.environ.get('BLOOM_INT4_CKPT_PATH')
        self.gptneox_model_path = os.environ.get('GPTNEOX_INT4_CKPT_PATH')
        
        
    def test_langchain_llm_embedding_llama(self):
        bigdl_embeddings = BigdlLLMEmbeddings(
            model_path=self.llama_model_path,
            model_family="llama")
        text = "This is a test document."
        query_result = bigdl_embeddings.embed_query(text)
        doc_result = bigdl_embeddings.embed_documents([text])
    
    def test_langchain_llm_embedding_gptneox(self):
        bigdl_embeddings = BigdlLLMEmbeddings(
            model_path=self.gptneox_model_path,
            model_family="gptneox")
        text = "This is a test document."
        query_result = bigdl_embeddings.embed_query(text)
        doc_result = bigdl_embeddings.embed_documents([text])
        
    def test_langchain_llm_llama(self):
        llm = BigdlLLM(
            model_path=self.llama_model_path, 
            max_tokens=32,
            n_threads=2)
        question = "What is AI?"
        result = llm(question)
        
    def test_langchain_llm_gptneox(self):
        llm = BigdlLLM(
            model_path=self.gptneox_model_path,
            model_family="gptneox", 
            max_tokens=32,
            n_threads=2)
        question = "What is AI?"
        result = llm(question)
        
    def test_langchain_llm_bloom(self):
        llm = BigdlLLM(
            model_path=self.bloom_model_path, 
            model_family="bloom",
            max_tokens=32,
            n_threads=2)
        question = "What is AI?"
        result = llm(question)
        
if __name__ == '__main__':
    pytest.main([__file__])