# LlamaIndex Examples


This folder contains examples showcasing how to use [**LlamaIndex**](https://github.com/run-llama/llama_index) with `ipex-llm`.
> [**LlamaIndex**](https://github.com/run-llama/llama_index) is a data framework designed to improve large language models by providing tools for easier data ingestion, management, and application integration. 

## 1. Setting up Dependencies 

* **Install LlamaIndex Packages**
    ```bash
    pip install llama-index-readers-file llama-index-vector-stores-postgres llama-index-embeddings-huggingface
    ```
* **Install IPEX-LLM**

    Follow the instructions in [GPU Install Guide](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Overview/install.html) to install ipex-llm.

* **Database Setup (using PostgreSQL)**:
  > Note: Database Setup is only required in RAG example.

  * Linux
      * Installation: 
          ```bash
          sudo apt-get install postgresql-client
          sudo apt-get install postgresql
          ```
      * Initialization:

          Switch to the **postgres** user and launch **psql** console
          ```bash
          sudo su - postgres
          psql
          ```

          Then, create a new user role:
          ```bash
          CREATE ROLE <user> WITH LOGIN PASSWORD '<password>';
          ALTER ROLE <user> SUPERUSER;    
          ```
  * Windows
      * click `Download the installer` in [PostgreSQL](https://www.postgresql.org/download/windows/).  
      * Run the downloaded installation package as administrator, then click `next` continuously.  
      * Open PowerShell:
      ```bash
          cd C:\Program Files\PostgreSQL\14\bin
      ```   
      The exact path will vary depending on your PostgreSQL location.  
      * Then in PowerShell:  
          
        ```bash
            .\psql -U postgres    
        ``` 
          
          Input the password you set in the previous installation. If PowerShell shows `postgres=#`, it indicates a successful connection.
      * Create a new user role:
      ```bash
      CREATE ROLE <user> WITH LOGIN PASSWORD '<password>';
      ALTER ROLE <user> SUPERUSER;    
        ```
* **Pgvector Installation**:
  > Note: Pgvector Installation is only required in RAG example.


  * Linux
      * Follow installation instructions on [pgvector's GitHub](https://github.com/pgvector/pgvector) and refer to the [installation notes](https://github.com/pgvector/pgvector#installation-notes) for additional help.
  * Windows 
      * It is recommended to use [pgvector for Windows](https://github.com/pgvector/pgvector?tab=readme-ov-file#windows) instead of others (such as conda-force) to avoid potential errors. Some steps may require running as administrator.


* **Data Preparation**: 
  > Note: Data Preparation is only required in RAG example.

  Download the Llama2 paper and save it as `data/llama2.pdf`, which serves as the default source file for retrieval.
    ```bash
    mkdir data
    wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O "data/llama2.pdf"
    ```

## 2. Configures OneAPI environment variables
### 2.1 Configurations for Linux
```bash
source /opt/intel/oneapi/setvars.sh
```
### 2.2 Configurations for Windows
```cmd
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
```
> Note: Please make sure you are using **CMD** (**Anaconda Prompt** if using conda) to run the command as PowerShell is not supported.

## 3. Runtime Configurations
For optimal performance, it is recommended to set several environment variables. Please check out the suggestions based on your device.
### 3.1 Configurations for Linux
<details>

<summary>For Intel Arc™ A-Series Graphics and Intel Data Center GPU Flex Series</summary>

```bash
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
```

</details>

<details>

<summary>For Intel Data Center GPU Max Series</summary>

```bash
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export ENABLE_SDP_FUSION=1
```
> Note: Please note that `libtcmalloc.so` can be installed by `conda install -c conda-forge -y gperftools=2.10`.
</details>

### 3.2 Configurations for Windows
<details>

<summary>For Intel iGPU</summary>

```cmd
set SYCL_CACHE_PERSISTENT=1
set BIGDL_LLM_XMX_DISABLED=1
```

</details>

<details>

<summary>For Intel Arc™ A300-Series or Pro A60</summary>

```cmd
set SYCL_CACHE_PERSISTENT=1
```

</details>

<details>

<summary>For other Intel dGPU Series</summary>

There is no need to set further environment variables.

</details>

> Note: For the first time that each model runs on Intel iGPU/Intel Arc™ A300-Series or Pro A60, it may take several minutes to compile.


## 4. Run the examples

### 4.1 RAG (Retrival Augmented Generation)

The RAG example ([rag.py](./rag.py)) is adapted from the [Official llama index RAG example](https://docs.llamaindex.ai/en/stable/examples/low_level/oss_ingestion_retrieval.html). This example builds a pipeline to ingest data (e.g. llama2 paper in pdf format) into a vector database (e.g. PostgreSQL), and then build a retrieval pipeline from that vector database. 

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

**Example Output**:

A query such as **"How does Llama 2 compare to other open-source models?"** with the Llama2 paper as the data source, using the `Llama-2-7b-chat-hf` model, will produce the output like below:

```
The comparison between Llama 2 and other open-source models is complex and depends on various factors such as the specific benchmarks used, the model size, and the task at hand.

In terms of performance on the benchmarks provided in the table, Llama 2 outperforms other open-source models on most categories. For example, on the MMLU benchmark, Llama 2 achieves a score of 22.5, while the next best open-source model, Poplar Aggregated Benchmarks, scores 17.5. Similarly, on the BBH benchmark, Llama 2 scores 20.5, while the next best open-source model scores 16.5.

However, it's important to note that the performance of Llama 2 can vary depending on the specific task and dataset being used. For example, on the coding benchmarks, Llama 2 performs significantly worse than other open-source models, such as PaLM (540B) and GPT-4.

In conclusion, while Llama 2 performs well on most benchmarks compared to other open-source models, its performance
```

### 4.2 Text to SQL

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

### 4.3 React Agent

> Note: Text to SQL example is varified on `zephyr-7b-alpha`. This model requires transformers==4.37.0. Please use `pip install transformers==4.37.0` to upgrade transformers version to 4.37.0.

The React Agent example ([react_agent.py](./react_agent.py)) is adapted from the [Official llama index React Agent example](https://docs.llamaindex.ai/en/stable/examples/agent/react_agent/). This example shows the ReAct agent over very simple calculator tools (no fancy RAG pipelines or API calls).

In the current directory, run the example with command:

```bash
python react_agent.py -m <path_to_model>
```
**Additional Parameters for Configuration**:
- `-m MODEL_PATH`: **Required**, path to the LLM model
- `-q QUESTION`: question you want to ask
- `-n N_PREDICT`: max predict tokens

**Example Output**:

A query such as **"What is 20+(2*4)?"** using the `zephyr-7b-alpha` model, will produce the output like below:
```
Thought: The current language of the user is: English. I need to use a tool to help me answer the question.
Action: add
Action Input: {}
Observation: Error: add() missing 2 required positional arguments: 'a' and 'b'
Thought: The current language of the user is: English. I need to use a tool to help me answer the question.
Action: add
Action Input: {'a': 20, 'b': 8}
Observation: 28
Thought: I can answer without using any more tools. I'll use the user's language to answer
Answer: 28
=========response=========
28
```

### 4.4 JSON Query Engine

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

A query such as **"What comments has Jerry been writing?"** using the `zephyr-7b-alpha` model, will produce the output like below:
```
Jerry has written the following comments:
- "Nice post!" on blog post with ID 1.
```

### 4.5 Query from Dataframe

> Note: Query from Dataframe example is varified on `zephyr-7b-alpha`. This model requires transformers==4.37.0. Please use `pip install transformers==4.37.0` to upgrade transformers version to 4.37.0.

The Query from Dataframe example ([query_pipeline_pandas.py](./query_pipeline_pandas.py)) is adapted from the [Official llama index Query from Dataframe example](https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/pipeline/query_pipeline_pandas.ipynb). This example builds a query pipeline that can perform structured operations over a Pandas DataFrame to satisfy a user query, using LLMs to infer the set of operations.

In the current directory, run the example with command:

```bash
# Download data
wget 'https://raw.githubusercontent.com/jerryjliu/llama_index/main/docs/docs/examples/data/csv/titanic_train.csv' -O 'titanic_train.csv'
# Run the example
python json_query_engine.py -m <path_to_model>
```

**Additional Parameters for Configuration**:
- `-m MODEL_PATH`: **Required**, path to the LLM model
- `-q QUESTION`: question you want to ask
- `-n N_PREDICT`: max predict tokens

**Example Output**:

A query such as **"What is the correlation between survival and age?"** using the `zephyr-7b-alpha` model, will produce the output like below:
```
The correlation between survival and age is -0.077. This suggests a weak negative correlation, meaning that as age increases, the likelihood of survival decreases slightly. However, the correlation is not statistically significant, as the p-value would need to be less than 0.05 to be considered significant.
```

## 5. Trouble shooting
### 5.1 Core dump
If you encounter a core dump error in your Python code, it is crucial to verify that the `import torch` statement is placed at the top of your Python file, just as what we did in `rag.py`.