# LlamaIndex Examples

The examples here show how to use LlamaIndex with `bigdl-llm`.
The RAG example is modified from the [demo](https://docs.llamaindex.ai/en/stable/examples/low_level/oss_ingestion_retrieval.html). 

## Install bigdl-llm
Follow the instructions in [Install](https://github.com/intel-analytics/BigDL/tree/main/python/llm#install).

## Install Required Dependencies for llamaindex examples. 

### Install Site-packages
```bash
pip install llama-index-readers-file
pip install llama-index-vector-stores-postgres
pip install llama-index-embeddings-huggingface
```

### Install Postgres
> Note: There are plenty of open-source databases you can use. Here we provide an example using Postgres. 
* Download and install postgres by running the commands below. 
    ```bash
    sudo apt-get install postgresql-client
    sudo apt-get install postgresql
    ```
* Initilize postgres. 
    ```bash
    sudo su - postgres
    psql
    ```
    After running the commands in the shell, we reach the console of postgres. Then we can add a role like the following
    ```bash
    CREATE ROLE <user> WITH LOGIN PASSWORD '<password>';
    ALTER ROLE <user> SUPERUSER;    
    ```
* Install pgvector according to the [page](https://github.com/pgvector/pgvector). If you encounter problem about the installation, please refer to the [notes](https://github.com/pgvector/pgvector#installation-notes) which may be helpful. 
* Download the database.
    ```bash
    mkdir data
    wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O "data/llama2.pdf"
    ```


## Run the examples

### Retrieval-augmented Generation
```bash
python rag.py -m MODEL_PATH -e EMBEDDING_MODEL_PATH -u USERNAME -p PASSWORD -q QUESTION -d DATA
```
arguments info:
- `-m MODEL_PATH`: **required**, path to the llama model
- `-e EMBEDDING_MODEL_PATH`: path to the embedding model
- `-u USERNAME`: username in the postgres database
- `-p PASSWORD`: password in the postgres database
- `-q QUESTION`: question you want to ask
- `-d DATA`: path to data used during retrieval

Here is the sample output when applying Llama-2-7b-chat-hf as the generatio model when we ask "How does Llama 2 perform compared to other open-source models?" and use llama.pdf as database. 
```
Llama 2 performs better than most open-source models on the benchmarks we tested. Specifically, it outperforms all open-source models on MMLU and BBH, and is close to GPT-3.5 on these benchmarks. Additionally, Llama 2 is on par or better than PaLM-2-L on almost all benchmarks. The only exception is the coding benchmarks, where Llama 2 lags significantly behind GPT-4 and PaLM-2-L. Overall, Llama 2 demonstrates strong performance on a wide range of natural language processing tasks.
```
