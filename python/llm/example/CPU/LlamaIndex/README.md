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

```bash
python 
```
