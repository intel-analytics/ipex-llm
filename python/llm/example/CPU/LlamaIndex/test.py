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


def load_vector_database():
    db_name = "example_db"
    host = "localhost"
    password = "1234qwer"
    port = "5432"
    user = "arda"
    # conn = psycopg2.connect(connection_string)
    conn = psycopg2.connect(
        dbname="postgres",
        host=host,
        password=password,
        port=port,
        user=user,
    )
    conn.autocommit = True

    with conn.cursor() as c:
        c.execute(f"DROP DATABASE IF EXISTS {db_name}")
        c.execute(f"CREATE DATABASE {db_name}")
    
    vector_store = PGVectorStore.from_params(
        database=db_name,
        host=host,
        password=password,
        port=port,
        user=user,
        table_name="llama2_paper",
        embed_dim=384,  # openai embedding dimension
    )
    return vector_store


def load_data():
    loader = PyMuPDFReader()
    documents = loader.load(file_path="/home/arda/zhicunlv/code/BigDL/python/llm/example/CPU/LlamaIndex/data/llama2.pdf")


    text_parser = SentenceSplitter(
        chunk_size=1024,
        # separator=" ",
    )
    text_chunks = []
    # maintain relationship with source doc index, to help inject doc metadata in (3)
    doc_idxs = []
    for doc_idx, doc in enumerate(documents):
        cur_text_chunks = text_parser.split_text(doc.text)
        text_chunks.extend(cur_text_chunks)
        doc_idxs.extend([doc_idx] * len(cur_text_chunks))

    from llama_index.core.schema import TextNode
    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        node = TextNode(
            text=text_chunk,
        )
        src_doc = documents[doc_idxs[idx]]
        node.metadata = src_doc.metadata
        nodes.append(node)
    return nodes
    





class VectorDBRetriever(BaseRetriever):
    """Retriever over a postgres vector store."""

    def __init__(
        self,
        vector_store: PGVectorStore,
        embed_model: Any,
        query_mode: str = "default",
        similarity_top_k: int = 2,
    ) -> None:
        """Init params."""
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        query_embedding = self._embed_model.get_query_embedding(
            query_bundle.query_str
        )
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )
        query_result = self._vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        return nodes_with_scores
    
def completion_to_prompt(completion):
    return f"<|system|>\n</s>\n<|user|>\n{completion}</s>\n<|assistant|>\n"


# Transform a list of chat messages into zephyr-specific input
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

def main():
    embed_model = HuggingFaceEmbedding(model_name="/mnt/disk1/models/bge-small-en")
    
    # Use LlamaCPP
    # llm = LlamaCPP(
    #     # You can pass in the URL to a GGML model to download it automatically
    #     model_url=None,
    #     # optionally, you can set the path to a pre-downloaded model instead of model_url
    #     model_path="/mnt/disk1/models/gguf/llama-2-7b-chat.Q4_0.gguf",
    #     temperature=0.1,
    #     max_new_tokens=256,
    #     # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    #     context_window=3900,
    #     # kwargs to pass to __call__()
    #     generate_kwargs={},
    #     # kwargs to pass to __init__()
    #     # set to at least 1 to use GPU
    #     model_kwargs={"n_gpu_layers": 0},
    #     verbose=True,
    # )
    
    # Use custom LLM in BigDL
    from custom_LLM import BigdlLLM
    llm = BigdlLLM(
        model_name="/mnt/disk1/models/Llama-2-7b-chat-hf",
        tokenizer_name="/mnt/disk1/models/Llama-2-7b-chat-hf",
        context_window=3900,
        max_new_tokens=256,
        generate_kwargs={"temperature": 0.7, "do_sample": False},
        model_kwargs={},
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        device_map="cpu",
    )
    
    # Use LangChain API in llamaindex with BigDL
    # from bigdl.llm.langchain.llms import TransformersLLM
    # from llama_index.llms.langchain import LangChainLLM
    # llama_llm = TransformersLLM.from_model_id(
    #     model_id="/mnt/disk1/models/Llama-2-7b-chat-hf",
    #     model_kwargs={"temperature": 0.6, "max_length": 256, "trust_remote_code": True},
    # )
    # llm = LangChainLLM(llm=llama_llm)
    
    
    vector_store = load_vector_database()
    nodes = load_data()
    for node in nodes:
        node_embedding = embed_model.get_text_embedding(
            node.get_content(metadata_mode="all")
        )
        node.embedding = node_embedding
    
    vector_store.add(nodes)
    
    query_str = "Can you tell me about the key concepts for safety finetuning"
    query_embedding = embed_model.get_query_embedding(query_str)
    # construct vector store query
    

    query_mode = "default"
    # query_mode = "sparse"
    # query_mode = "hybrid"

    vector_store_query = VectorStoreQuery(
        query_embedding=query_embedding, similarity_top_k=2, mode=query_mode
    )
    # returns a VectorStoreQueryResult
    query_result = vector_store.query(vector_store_query)
    print("Retrieval Results: ")
    print(query_result.nodes[0].get_content())



    nodes_with_scores = []
    for index, node in enumerate(query_result.nodes):
        score: Optional[float] = None
        if query_result.similarities is not None:
            score = query_result.similarities[index]
        nodes_with_scores.append(NodeWithScore(node=node, score=score))
    
    retriever = VectorDBRetriever(
        vector_store, embed_model, query_mode="default", similarity_top_k=1
    )
    
    
    query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)

    query_str = "How does Llama 2 perform compared to other open-source models?"

    response = query_engine.query(query_str)


    print("------------RESPONSE GENERATION---------------------")
    print(str(response))


if __name__ == "__main__":
    main()