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

from bigdl.llm.langchain.llms import TransformersLLM, TransformersPipelineLLM
from bigdl.llm.langchain.embeddings import TransformersEmbeddings


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





class Test_Langchain_Transformers_API(TestCase):
    def setUp(self):
        self.auto_model_path = os.environ.get('ORIGINAL_CHATGLM2_6B_PATH')
        self.auto_causal_model_path = os.environ.get('ORIGINAL_REPLIT_CODE_PATH')
        thread_num = os.environ.get('THREAD_NUM')
        if thread_num is not None:
            self.n_threads = int(thread_num)
        else:
            self.n_threads = 2         

    def test_pipeline_llm(self):
        texts = 'def hello():\n  print("hello world")\n'
        bigdl_llm = TransformersPipelineLLM.from_model_id(model_id=self.auto_causal_model_path, task='text-generation', model_kwargs={'trust_remote_code': True})
        
        output = bigdl_llm(texts)
        res = "hello()" in output
        self.assertTrue(res)

        
    def test_qa_chain(self):
        texts = '''
AI is a machine’s ability to perform the cognitive functions 
we associate with human minds, such as perceiving, reasoning, 
learning, interacting with an environment, problem solving,
and even exercising creativity. You’ve probably interacted 
with AI even if you didn’t realize it—voice assistants like Siri 
and Alexa are founded on AI technology, as are some customer 
service chatbots that pop up to help you navigate websites.
        '''
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_text(texts)
        query = 'What is AI?'
        embeddings = TransformersEmbeddings.from_model_id(model_id=self.auto_model_path, model_kwargs={'trust_remote_code': True})

        docsearch = Chroma.from_texts(texts, embeddings, metadatas=[{"source": str(i)} for i in range(len(texts))]).as_retriever()

        #get relavant texts
        docs = docsearch.get_relevant_documents(query)
        bigdl_llm = TransformersLLM.from_model_id(model_id=self.auto_model_path, model_kwargs={'trust_remote_code': True})
        doc_chain = load_qa_chain(bigdl_llm, chain_type="stuff", prompt=QA_PROMPT)
        output = doc_chain.run(input_documents=docs, question=query)
        res = "AI" in output
        self.assertTrue(res)
        
if __name__ == '__main__':
    pytest.main([__file__])
