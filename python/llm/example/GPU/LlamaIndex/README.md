# LlamaIndex Examples


This folder contains examples showcasing how to use [**LlamaIndex**](https://github.com/run-llama/llama_index) with `ipex-llm`.
> [**LlamaIndex**](https://github.com/run-llama/llama_index) is a data framework designed to improve large language models by providing tools for easier data ingestion, management, and application integration. 


## Retrieval-Augmented Generation (RAG) Example
The RAG example ([rag.py](./rag.py)) is adapted from the [Official llama index RAG example](https://docs.llamaindex.ai/en/stable/examples/low_level/oss_ingestion_retrieval.html). This example builds a pipeline to ingest data (e.g. llama2 paper in pdf format) into a vector database (e.g. PostgreSQL), and then build a retrieval pipeline from that vector database. 



### 1. Setting up Dependencies 

* **Install LlamaIndex Packages**
    ```bash
    pip install llama-index-readers-file llama-index-vector-stores-postgres llama-index-embeddings-huggingface
    ```
* **Install IPEX-LLM**

    Follow the instructions in [GPU Install Guide](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Overview/install.html) to install ipex-llm.

* **Database Setup (using PostgreSQL)**:
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
    * Linux
        * Follow installation instructions on [pgvector's GitHub](https://github.com/pgvector/pgvector) and refer to the [installation notes](https://github.com/pgvector/pgvector#installation-notes) for additional help.
    * Windows 
        * It is recommended to use [pgvector for Windows](https://github.com/pgvector/pgvector?tab=readme-ov-file#windows) instead of others (such as conda-force) to avoid potential errors. Some steps may require running as administrator.


* **Data Preparation**: Download the Llama2 paper and save it as `data/llama2.pdf`, which serves as the default source file for retrieval.
    ```bash
    mkdir data
    wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O "data/llama2.pdf"
    ```

### 2. Configures OneAPI environment variables
#### 2.1 Configurations for Linux
```bash
source /opt/intel/oneapi/setvars.sh
```
#### 2.2 Configurations for Windows
```cmd
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
```
> Note: Please make sure you are using **CMD** (**Anaconda Prompt** if using conda) to run the command as PowerShell is not supported.
### 3. Runtime Configurations
For optimal performance, it is recommended to set several environment variables. Please check out the suggestions based on your device.
#### 3.1 Configurations for Linux
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

#### 3.2 Configurations for Windows
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


### 4. Running the RAG example

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

### 5. Example Output

A query such as **"How does Llama 2 compare to other open-source models?"** with the Llama2 paper as the data source, using the `Llama-2-7b-chat-hf` model, will produce the output like below:

```
The comparison between Llama 2 and other open-source models is complex and depends on various factors such as the specific benchmarks used, the model size, and the task at hand.

In terms of performance on the benchmarks provided in the table, Llama 2 outperforms other open-source models on most categories. For example, on the MMLU benchmark, Llama 2 achieves a score of 22.5, while the next best open-source model, Poplar Aggregated Benchmarks, scores 17.5. Similarly, on the BBH benchmark, Llama 2 scores 20.5, while the next best open-source model scores 16.5.

However, it's important to note that the performance of Llama 2 can vary depending on the specific task and dataset being used. For example, on the coding benchmarks, Llama 2 performs significantly worse than other open-source models, such as PaLM (540B) and GPT-4.

In conclusion, while Llama 2 performs well on most benchmarks compared to other open-source models, its performance
```

### 6. Trouble shooting
#### 6.1 Core dump
If you encounter a core dump error in your Python code, it is crucial to verify that the `import torch` statement is placed at the top of your Python file, just as what we did in `rag.py`.