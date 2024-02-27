# LlamaIndex Examples

The examples here show how to use LlamaIndex with `bigdl-llm`.

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

Here is the sample output when applying Llama-2-7b-chat-hf as the generatio model. 
```
AI stands for Artificial Intelligence. It refers to the development of computer systems that can perform tasks that typically require human intelligence, such as understanding natural language, recognizing images, making decisions, and solving problems. AI systems use algorithms and machine learning techniques to analyze data and make predictions or decisions based on that data.
```
