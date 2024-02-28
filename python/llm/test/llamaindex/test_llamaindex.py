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


from langchain.document_loaders import WebBaseLoader
from langchain.indexes import VectorstoreIndexCreator


from langchain.chains.question_answering import load_qa_chain
from langchain.chains.chat_vector_db.prompts import (CONDENSE_QUESTION_PROMPT,
                                                     QA_PROMPT)
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

import pytest
from unittest import TestCase
import os
from bigdl.llm.llamaindex.llms import BigdlLLM

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
        def completion_to_prompt(completion):
            return f"<|system|>\n</s>\n<|user|>\n{completion}</s>\n<|assistant|>\n"
        def messages_to_prompt(messages):
            prompt = ""
            for message in messages:
                if message.role == "system":
                    prompt += f"<|system|>\n{message.content}</s>\n"
                elif message.role == "user":
                    prompt += f"<|user|>\n{message.content}</s>\n"
                elif message.role == "assistant":
                    prompt += f"<|assistant|>\n{message.content}</s>\n"

            # ensure we start with a system prompt, insert blank if needed
            if not prompt.startswith("<|system|>\n"):
                prompt = "<|system|>\n</s>\n" + prompt

            # add final assistant prompt
            prompt = prompt + "<|assistant|>\n"
            return prompt
        
        llm = BigdlLLM(
            model_name=self.llama_model_path,
            tokenizer_name=self.llama_model_path,
            context_window=3900,
            max_new_tokens=256,
            model_kwargs={},
            generate_kwargs={"temperature": 0.7, "do_sample": False},
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            device_map="cpu",
        )
        res = llm.complete("What is AI?")
        

if __name__ == '__main__':
    pytest.main([__file__])