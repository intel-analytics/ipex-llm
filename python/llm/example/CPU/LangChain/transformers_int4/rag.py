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

# This would makes sure Python is aware there is more than one sub-package within bigdl,
# physically located elsewhere.
# Otherwise there would be module not found error in non-pip's setting as Python would
# only search the first bigdl package and end up finding only one sub-package.

# Code is adapted from https://python.langchain.com/docs/modules/chains/additional/question_answering.html

import argparse

from langchain.vectorstores import Chroma
from langchain.chains.chat_vector_db.prompts import (CONDENSE_QUESTION_PROMPT,
                                                     QA_PROMPT)
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks.manager import CallbackManager

from bigdl.llm.langchain.llms import TransformersLLM
from bigdl.llm.langchain.embeddings import TransformersEmbeddings

text_doc = '''
BigDL seamlessly scales your data analytics & AI applications from laptop to cloud, with the following libraries:
LLM: Low-bit (INT3/INT4/INT5/INT8) large language model library for Intel CPU/GPU
Orca: Distributed Big Data & AI (TF & PyTorch) Pipeline on Spark and Ray
Nano: Transparent Acceleration of Tensorflow & PyTorch Programs on Intel CPU/GPU
DLlib: “Equivalent of Spark MLlib” for Deep Learning
Chronos: Scalable Time Series Analysis using AutoML
Friesian: End-to-End Recommendation Systems
PPML: Secure Big Data and AI (with SGX Hardware Security)
'''

def main(args):

    input_path = args.input_path 
    model_path = args.model_path
    query = args.question

    # split texts of input doc
    if input_path is None:
        input_doc = text_doc
    else:
        with open(input_path) as f:
            input_doc = f.read()
            
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_text(input_doc)

    # create embeddings and store into vectordb
    embeddings = TransformersEmbeddings.from_model_id(
        model_id=model_path, 
        model_kwargs={"trust_remote_code": True}
        )
    docsearch = Chroma.from_texts(texts, embeddings, metadatas=[{"source": str(i)} for i in range(len(texts))]).as_retriever()

    #get relavant texts
    docs = docsearch.get_relevant_documents(query)

    bigdl_llm = TransformersLLM.from_model_id(
        model_id=model_path,
        model_kwargs={"temperature": 0, "max_length": 1024, "trust_remote_code": True},
    )

    doc_chain = load_qa_chain(
        bigdl_llm, chain_type="stuff", prompt=QA_PROMPT
    )

    output = doc_chain.run(input_documents=docs, question=query)
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TransformersLLM Langchain QA over Docs Example')
    parser.add_argument('-m','--model-path', type=str, required=True,
                        help='the path to transformers model')
    parser.add_argument('-i', '--input-path', type=str,
                        help='the path to the input doc.')
    parser.add_argument('-q', '--question', type=str, default='What is BigDL?',
                        help='qustion you want to ask.')
    args = parser.parse_args()
    
    main(args)
