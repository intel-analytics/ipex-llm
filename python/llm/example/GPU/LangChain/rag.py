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

# Code is adapted from https://github.com/langchain-ai/langchain/blob/master/docs/docs/tutorials/rag.ipynb

import argparse
import warnings

from langchain import hub
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import IpexLLMBgeEmbeddings
from langchain_community.llms import IpexLLM
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma

warnings.filterwarnings("ignore", category=UserWarning, message=".*padding_mask.*")


text_doc = '''
IPEX-LLM is an LLM acceleration library for Intel CPU, GPU (e.g., local PC with iGPU, discrete GPU such as Arc, Flex and Max) and NPU. It is built on top of the excellent work of llama.cpp, transformers, bitsandbytes, vLLM, qlora, AutoGPTQ, AutoAWQ, etc. It provides seamless integration with llama.cpp, Ollama, HuggingFace transformers, LangChain, LlamaIndex, vLLM, Text-Generation-WebUI, DeepSpeed-AutoTP, FastChat, Axolotl, HuggingFace PEFT, HuggingFace TRL, AutoGen, ModeScope, etc. 70+ models have been optimized/verified on ipex-llm (e.g., Llama, Phi, Mistral, Mixtral, Whisper, Qwen, MiniCPM, Qwen-VL, MiniCPM-V and more), with state-of-art LLM optimizations, XPU acceleration and low-bit (FP8/FP6/FP4/INT4) support.
'''

def main(args):

    input_path = args.input_path 
    model_path = args.model_path
    embed_model_path = args.embed_model_path
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
    embeddings = IpexLLMBgeEmbeddings(
        model_name=embed_model_path,
        model_kwargs={"device": "xpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    retriever = Chroma.from_texts(texts, embeddings, metadatas=[{"source": str(i)} for i in range(len(texts))]).as_retriever()

    llm = IpexLLM.from_model_id(
        model_id=model_path,
        model_kwargs={
            "temperature": 0,
            "max_length": 512,
            "trust_remote_code": True,
            "device": "xpu",
        },
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    prompt = hub.pull("rlm/rag-prompt")

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    rag_chain.invoke(query)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TransformersLLM Langchain QA over Docs Example')
    parser.add_argument('-m','--model-path', type=str, required=True,
                        help='the path to transformers model')
    parser.add_argument('-e','--embed-model-path', type=str, required=True,
                        help='the path to embedding model')
    parser.add_argument('-i', '--input-path', type=str,
                        help='the path to the input doc.')
    parser.add_argument('-q', '--question', type=str, default='What is IPEX-LLM?',
                        help='qustion you want to ask.')
    args = parser.parse_args()
    
    main(args)
