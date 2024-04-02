# LlamaIndex Examples


This folder contains examples showcasing how to use [**LlamaIndex**](https://github.com/run-llama/llama_index) with `ipex-llm`.
> [**LlamaIndex**](https://github.com/run-llama/llama_index) is a data framework designed to improve large language models by providing tools for easier data ingestion, management, and application integration. 


## 1. Setting up Dependencies 

* **Install ipex-llm** 
  
  Ensure `ipex-llm` is installed by following the [IPEX-LLM Installation Guide](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm#install) before proceeding with the examples provided here. 

* **Install LlamaIndex Packages**
    ```bash
    pip install llama-index-readers-file llama-index-vector-stores-postgres llama-index-embeddings-huggingface
    ```

* **Database Setup (using PostgreSQL)**:
  > Note: Database Setup is only required in RAG example.

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
  > Note: Database Setup is only required in RAG example.
  
  Follow installation instructions on [pgvector's GitHub](https://github.com/pgvector/pgvector) and refer to the [installation notes](https://github.com/pgvector/pgvector#installation-notes) for additional help.


* **Data Preparation**: 
  > Note: Database Setup is only required in RAG example.
  
  Download the Llama2 paper and save it as `data/llama2.pdf`, which serves as the default source file for retrieval.
  ```bash
  mkdir data
  wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O "data/llama2.pdf"
  ```


## 2. Run the examples

### 2.1 RAG (Retrival Augmented Generation)

The RAG example ([rag.py](./rag.py)) is adapted from the [Official llama index RAG example](https://docs.llamaindex.ai/en/stable/examples/low_level/oss_ingestion_retrieval.html). This example builds a pipeline to ingest data (e.g. llama2 paper in pdf format) into a vector database (e.g. PostgreSQL), and then build a retrieval pipeline from that vector database. 

In the current directory, run the example with command:

```bash
python rag.py -m <path_to_model>
```
**Additional Parameters for Configuration**:
- `-m MODEL_PATH`: **Required**, path to the LLM model
- `-e EMBEDDING_MODEL_PATH`: path to the embedding model
- `-u USERNAME`: username in the PostgreSQL database
- `-p PASSWORD`: password in the PostgreSQL database
- `-q QUESTION`: question you want to ask
- `-d DATA`: path to source data used for retrieval (in pdf format)
- `-n N_PREDICT`: max predict tokens

**Example Output**：

A query such as **"How does Llama 2 compare to other open-source models?"** with the Llama2 paper as the data source, using the `Llama-2-7b-chat-hf` model, will produce the output like below:

```
Llama 2 performs better than most open-source models on the benchmarks we tested. Specifically, it outperforms all open-source models on MMLU and BBH, and is close to GPT-3.5 on these benchmarks. Additionally, Llama 2 is on par or better than PaLM-2-L on almost all benchmarks. The only exception is the coding benchmarks, where Llama 2 lags significantly behind GPT-4 and PaLM-2-L. Overall, Llama 2 demonstrates strong performance on a wide range of natural language processing tasks.
```

### 2.2 Text to SQL

> Note: Text to SQL example is varified on `zephyr-7b-alpha`. This model requires transformers==4.37.0. Please use `pip install transformers==4.37.0` to upgrade transformers version to 4.37.0.

The Text to SQL example ([text_to_sql.py](./text_to_sql.py)) is adapted from the [Official llama index Text-to-SQL example](https://docs.llamaindex.ai/en/stable/examples/index_structs/struct_indices/SQLIndexDemo/#part-3-text-to-sql-retriever). This example shows how to define a text-to-SQL retriever on its own and plug it into `RetrieverQueryEngine` to build a retrival pipeline.

In the current directory, run the example with command:

```bash
python text_to_sql.py -m <path_to_model> -e <path_to_embedding_model>
```
**Additional Parameters for Configuration**:
- `-m MODEL_PATH`: **Required**, path to the LLM model
- `-e EMBEDDING_MODEL_PATH`: **Required**, path to the embedding model
- `-q QUESTION`: question you want to ask
- `-n N_PREDICT`: max predict tokens
**Example Output**:
A query such as **"Which city has the highest population?"** using the `zephyr-7b-alpha` model, will produce the output like below:
```
The city with the highest population is Tokyo, with a population of 13,960,000.
```

### 2.3 React Agent

> Note: React Agent example is varified on `zephyr-7b-alpha`. This model requires transformers==4.37.0. Please use `pip install transformers==4.37.0` to upgrade transformers version to 4.37.0.

The Text to SQL example ([react_agent.py](./react_agent.py)) is adapted from the [Official llama index React Agent example](https://docs.llamaindex.ai/en/stable/examples/agent/react_agent/). This example shows the ReAct agent over very simple calculator tools (no fancy RAG pipelines or API calls).

In the current directory, run the example with command:

```bash
python react_agent.py -m <path_to_model>
```
**Additional Parameters for Configuration**:
- `-m MODEL_PATH`: **Required**, path to the LLM model
- `-q QUESTION`: question you want to ask
- `-n N_PREDICT`: max predict tokens

**Example Output**:

A query such as **"What is 3+5?"** using the `zephyr-7b-alpha` model, will produce the output like below:
```
Thought: (Implicit) I can answer without any more tools!
Answer: <|assistant|>
The sum of 3 and 5 is 8.
=========response=========
<|assistant|>
The sum of 3 and 5 is 8.
```

### 2.4 JSON Query Engine

> Note: 
    > - JSON Query Engine example is varified on `zephyr-7b-alpha`. This model requires transformers==4.37.0. Please use `pip install transformers==4.37.0` to upgrade transformers version to 4.37.0.
    > - This example also requires `jsonpath-ng`. Use `pip install jsonpath-ng` to install.

The JSON Query Engine example ([json_query_engine.py](./json_query_engine.py)) is adapted from the [Official llama index JSON Query Engine example](https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/query_engine/json_query_engine.ipynb). This example shows how to query JSON documents that conform to a JSON schema. This JSON schema is then used in the context of a prompt to convert a natural language query into a structured JSON Path query. This JSON Path query is then used to retrieve data to answer the given question.

In the current directory, run the example with command:

```bash
python json_query_engine.py -m <path_to_model>
```

**Additional Parameters for Configuration**:
- `-m MODEL_PATH`: **Required**, path to the LLM model
- `-q QUESTION`: question you want to ask
- `-n N_PREDICT`: max predict tokens

**Example Output**:

A query such as **"What comments has simon been writing?"** using the `zephyr-7b-alpha` model, will produce the output like below:
```
Interesting thoughts, Loved reading this!

This is a comment written by simon.

This is a blog post written by admin.

This is a comment written by john.

This is a blog post written by admin.

This is a comment written by simon.

This is a blog post written by admin.

This is a comment written by john.

This is a blog post written by admin.

This is a comment written by simon.

This is a blog post written by admin.

This is a comment
```