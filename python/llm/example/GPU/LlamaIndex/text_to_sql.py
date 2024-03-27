import torch
from llama_index.core import SQLDatabase
from llama_index.core.retrievers import NLSQLRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from ipex_llm.llamaindex.llms import BigdlLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, insert
import argparse

def create_database_schema():
    engine = create_engine("sqlite:///:memory:")
    metadata_obj = MetaData()

    # create city SQL table
    table_name = "city_stats"
    city_stats_table = Table(
        table_name,
        metadata_obj,
        Column("city_name", String(16), primary_key=True),
        Column("population", Integer),
        Column("country", String(16), nullable=False),
    )
    metadata_obj.create_all(engine)
    return engine, city_stats_table

def define_sql_database(engine, city_stats_table):
    sql_database = SQLDatabase(engine, include_tables=["city_stats"])

    rows = [
        {"city_name": "Toronto", "population": 2930000, "country": "Canada"},
        {"city_name": "Tokyo", "population": 13960000, "country": "Japan"},
        {
            "city_name": "Chicago",
            "population": 2679000,
            "country": "United States",
        },
        {"city_name": "Seoul", "population": 9776000, "country": "South Korea"},
    ]
    for row in rows:
        stmt = insert(city_stats_table).values(**row)
        with engine.begin() as connection:
            cursor = connection.execute(stmt)

    return sql_database

def main(args):
    engine, city_stats_table = create_database_schema()
    sql_database = define_sql_database(engine, city_stats_table)

    embed_model = HuggingFaceEmbedding(model_name=args.embedding_model_path)

    llm = BigdlLLM(
        model_name=args.model_path,
        tokenizer_name=args.model_path,
        context_window=512,
        max_new_tokens=args.n_predict,
        generate_kwargs={"temperature": 0.7, "do_sample": False},
        model_kwargs={},
        device_map="xpu"
    )

    # default retrieval (return_raw=True)
    nl_sql_retriever = NLSQLRetriever(
                    sql_database, 
                    tables=["city_stats"], 
                    llm=llm, 
                    embed_model=embed_model, 
                    return_raw=True
    )


    query_engine = RetrieverQueryEngine.from_args(nl_sql_retriever, llm=llm)

    query_str = args.question
    response = query_engine.query(query_str)
    print(str(response))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LlamaIndex BigdlLLM Example')
    parser.add_argument('-m','--model-path', type=str, required=True,
                        help='the path to transformers model')
    parser.add_argument('-q', '--question', type=str, default='Which city has the highest population?',
                        help='qustion you want to ask.')
    parser.add_argument('-e','--embedding-model-path',default="BAAI/bge-small-en",
                        help="the path to embedding model path")
    parser.add_argument('-n','--n-predict', type=int, default=32,
                        help='max number of predict tokens')
    args = parser.parse_args()
    
    main(args)