from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sqlalchemy import make_url
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.llms.llama_cpp import LlamaCPP
import psycopg2
from pathlib import Path
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.schema import NodeWithScore
from typing import Optional
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from typing import Any, List
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import VectorStoreQuery

from bigdl.llm.langchain.llms import TransformersLLM, TransformersPipelineLLM
from langchain import PromptTemplate, LLMChain
from langchain import HuggingFacePipeline
from llama_index.llms.langchain import LangChainLLM
import time

# model_path = "/mnt/disk1/models/gguf/llama-2-7b-chat.Q4_0.gguf"
model_path = "/mnt/disk1/models/Llama-2-7b-chat-hf"
question = "What is AI?"
template ="""{question}"""

prompt = PromptTemplate(template=template, input_variables=["question"])
llama_llm = TransformersLLM.from_model_id(
        model_id=model_path,
        model_kwargs={"temperature": 0, "max_length": 64, "trust_remote_code": True},
    )
# llama_llm = LlamaCPP(
#         # You can pass in the URL to a GGML model to download it automatically
#         model_url=None,
#         # optionally, you can set the path to a pre-downloaded model instead of model_url
#         model_path=model_path,
#         temperature=0,
#         max_new_tokens=64,
#         # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
#         context_window=3900,
#         # kwargs to pass to __call__()
#         generate_kwargs={},
#         # kwargs to pass to __init__()
#         # set to at least 1 to use GPU
#         model_kwargs={"n_gpu_layers": 0},
#         verbose=True,
#     )

llm = LangChainLLM(llm=llama_llm)
st = time.time()
response_gen = llm.stream_complete("Hi this is")
end = time.time()
print(f"time cost = {end-st}")
print(response_gen)
for delta in response_gen:
    print(delta.text, end="")