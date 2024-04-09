# LlamaIndex Examples


This folder contains examples showcasing how to use [**LlamaIndex**](https://github.com/run-llama/llama_index) with `ipex-llm`.
> [**LlamaIndex**](https://github.com/run-llama/llama_index) is a data framework designed to improve large language models by providing tools for easier data ingestion, management, and application integration. 


## Retrieval-Augmented Generation (RAG) Example
The RAG example ([rag.py](./rag.py)) is adapted from the [Official llama index RAG example](https://docs.llamaindex.ai/en/stable/examples/low_level/oss_ingestion_retrieval.html). This example builds a pipeline to ingest data (e.g. llama2 paper in pdf format) into a vector database (e.g. PostgreSQL), and then build a retrieval pipeline from that vector database. 



### Setting up Dependencies 

* **Install LlamaIndex Packages**
    ```bash
    pip install llama-index-readers-file llama-index-vector-stores-postgres llama-index-embeddings-huggingface
    ```

* **Install IPEX-LLM**
Ensure `ipex-llm` is installed by following the [IPEX-LLM Installation Guide](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Overview/install.html) before proceeding with the examples provided here. 


* **Database Setup (using PostgreSQL)**:
    * Installation: 
        ```bash
        sudo apt-get install postgresql-client
        sudo apt-get install postgresql
        ```
    * Initialization:

      Switch to the **postgres** user and launch **psql** console:
        ```bash
        sudo su - postgres
        psql
        ```
      Then, create a new user role:
        ```bash
        CREATE ROLE <user> WITH LOGIN PASSWORD '<password>';
        ALTER ROLE <user> SUPERUSER;    
        ```
* **Pgvector Installation**:
    Follow installation instructions on [pgvector's GitHub](https://github.com/pgvector/pgvector) and refer to the [installation notes](https://github.com/pgvector/pgvector#installation-notes) for additional help.


* **Data Preparation**: Download the Llama2 paper and save it as `data/llama2.pdf`, which serves as the default source file for retrieval.
    ```bash
    mkdir data
    wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O "data/llama2.pdf"
    ```


### Running the RAG example

In the current directory, run the example with command:

```bash
python rag.py -m <path_to_model> -t <path_to_tokenizer>
```
**Additional Parameters for Configuration**:
- `-m MODEL_PATH`: **Required**, path to the LLM model
- `-e EMBEDDING_MODEL_PATH`: path to the embedding model
- `-u USERNAME`: username in the PostgreSQL database
- `-p PASSWORD`: password in the PostgreSQL database
- `-q QUESTION`: question you want to ask
- `-d DATA`: path to source data used for retrieval (in pdf format)
- `-n N_PREDICT`: max predict tokens
- `-t TOKENIZER_PATH`: **Required**, path to the tokenizer model

### Example Output

A query such as **"How does Llama 2 compare to other open-source models?"** with the Llama2 paper as the data source, using the `Llama-2-7b-chat-hf` model, will produce the output like below:

```
Llama 2 performs better than most open-source models on the benchmarks we tested. Specifically, it outperforms all open-source models on MMLU and BBH, and is close to GPT-3.5 on these benchmarks. Additionally, Llama 2 is on par or better than PaLM-2-L on almost all benchmarks. The only exception is the coding benchmarks, where Llama 2 lags significantly behind GPT-4 and PaLM-2-L. Overall, Llama 2 demonstrates strong performance on a wide range of natural language processing tasks.
```
