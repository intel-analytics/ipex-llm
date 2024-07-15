# Run GraphRAG with IPEX-LLM on Intel GPU

The [GraphRAG project](https://github.com/microsoft/graphrag) is designed to leverage large language models (LLMs) for extracting structured and meaningful data from unstructured texts; by integrating it with [`ipex-llm`](https://github.com/intel-analytics/ipex-llm), users can now easily utilize local LLMs running on Intel GPU (e.g., local PC with iGPU, discrete GPU such as Arc, Flex and Max).

## Table of Contents

## Quickstart

### 1. Install and Start `Ollama` Service on Intel GPU 

Follow the steps in [Run Ollama with IPEX-LLM on Intel GPU Guide](./ollama_quickstart.md) to install and run Ollama on Intel GPU. Ensure that `ollama serve` is running correctly and can be accessed through a local URL (e.g., `https://127.0.0.1:11434`).

> [!TIP]
> If your local LLM is running on Intel Arc™ A-Series Graphics with Linux OS (Kernel 6.2), it is recommended to additionaly set the following environment variable for optimal performance before executing `ollama serve`:
>
> ```bash
> export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
> ```

### 2. Prepare LLM and Embedding Model

In another terminal window, separate from where you executed `ollama serve`, you can download the LLM and embedding model using the following commands:

- For **Linux users**:

  ```bash
  export no_proxy=localhost,127.0.0.1
  # LLM
  ./ollama pull mistral
  # Embedding model
  ./ollama pull nomic-embed-text
  ```

- For **Windows users**:

  Please run the following command in Miniforge or Anaconda Prompt.

  ```cmd
  set no_proxy=localhost,127.0.0.1
  :: LLM
  ollama pull mistral
  :: Embedding model
  ollama pull nomic-embed-text
  ```

> [!TIP]
> Here we take [`mistral`](https://ollama.com/library/mistral) and [`nomic-embed-text`](https://ollama.com/library/nomic-embed-text) as an example. You could have a try on other LLMs or embedding models in the [`Ollama model library`](https://ollama.com/library).

### 3. Setup Python Environment for GraphRAG

To run the LLM and embedding model on a local machine, we utilize the [`graphrag-local-ollama`](https://github.com/TheAiSingularity/graphrag-local-ollama) repository:

```shell
git clone https://github.com/TheAiSingularity/graphrag-local-ollama.git
cd graphrag-local-ollama

conda create -n graphrag-local-ollama python=3.10
conda activate graphrag-local-ollama

pip install -e .

pip install ollama
pip install plotly
```

in which `pip install ollama` is for enabling restful APIs through python, and `pip install plotly` is for visualizing the knowledge graph.

### 4. Index GraphRAG

The environment is now ready to conduct GraphRAG with local LLMs and embedding models running on Intel GPUs. Before querying GraphRAG, it is necessary to first index GraphRAG, which could be a resource-intensive operation.

> [!TIP]
> Refer to [here](https://microsoft.github.io/graphrag/) for more details in GraphRAG process explanation.

Some [sample documents](https://github.com/TheAiSingularity/graphrag-local-ollama/tree/main/input) are used here as input corpus for indexing GraphRAG.

First, perpare the input corpus, based on which LLM will create a knowledge graph, and initialize the workspace:

- For **Linux users**:

  ```bash
  # define inputs corpus
  mkdir -p ./ragtest/input
  cp input/* ./ragtest/input

  export no_proxy=localhost,127.0.0.1

  # initialize ragtest folder
  python -m graphrag.index --init --root ./ragtest

  # prepare settings.yml, please make sure the initialized settings.yml in ragtest folder is replaced by settings.yml in graphrag-ollama-local folder
  move settings.yaml ./ragtest
  ```

- For **Windows users**:

  Please run the following command in Miniforge or Anaconda Prompt.

  ```cmd
  :: define inputs corpus
  mkdir ragtest && cd ragtest && mkdir input && cd .. 
  xcopy input\* .\ragtest\input

  set no_proxy=localhost,127.0.0.1

  :: initialize ragtest folder
  python -m graphrag.index --init --root .\ragtest

  :: prepare settings.yml, please make sure the initialized settings.yml in ragtest folder is replaced by settings.yml in graphrag-ollama-local folder
  move settings.yaml .\ragtest
  ```

> [!NOTE]
> If you would like to use LLMs or embedding models other than `mistral` or `nomic-embed-text`, you are required to update the `settings.yml` in `ragtest` folder accordingly:
>
> ```yml
> llm:
>   api_key: ${GRAPHRAG_API_KEY}
>   type: openai_chat
>   model: mistral # change it to the LLM from Ollama model library as you want
>   model_supports_json: true
>   api_base: http://localhost:11434/v1
> 
> embeddings:
>   async_mode: threaded
>   llm:
>     api_key: ${GRAPHRAG_API_KEY}
>     type: openai_embedding
>     model: nomic_embed_text # change it to the embedding model from Ollama model library as you want
>     api_base: http://localhost:11434/api
> ```

Finally, conduct GraphRAG indexing, which may take a while:

```shell
python -m graphrag.index --root ragtest
```

You will got message `🚀 All workflows completed successfully.` after the GraphRAG indexing is successfully finished.

#### (Optional) Visualize Knowledge Graph

For a clearer view of the knowledge graph, you can visualize it by specifying the path to the `.graphml` file in the `visualize-graphml.py` script, like below:

- For **Windows users**:

  ```python
  graph = nx.read_graphml('ragtest\\output\\20240715-151518\\artifacts\\summarized_graph.graphml') 
  ```

and run:

```shell
python visualize-graphml.py
```

[image]

### 5. Query GraphRAG

After the GraphRAG has been successfully indexed, you could conduct query based on the knowledge graph through:

```shell
python -m graphrag.query --root ragtest --method global "What is Transformer?"
```

> [!NOTE]
> Only the `global` query method is supported for now.

The sample output looks like:

```log
INFO: Reading settings from ragtest\settings.yaml
creating llm client with {'api_key': 'REDACTED,len=9', 'type': "openai_chat", 'model': 'mistral', 'max_tokens': 4000, 'temperature': 0.0, 'top_p': 1.0, 'request_timeout': 180.0, 'api_base': 'http://localhost:11434/v1', 'api_version': None, 'organization': None, 'proxy': None, 'cognitive_services_endpoint': None, 'deployment_name': None, 'model_supports_json': True, 'tokens_per_minute': 0, 'requests_per_minute': 0, 'max_retries': 10, 'max_retry_wait': 10.0, 'sleep_on_rate_limit_recommendation': True, 'concurrent_requests': 25}

SUCCESS: Global Search Response:  The Transformer is a type of model architecture used primarily in machine learning, particularly in natural language processing (NLP). It was initially introduced in the paper 'Attention is All You Need' by Vaswani et al. [Data: Reports (1)]

Transformer models employ self-attention mechanisms to focus on different parts of input sequences when making predictions. This allows them to better capture long-range dependencies and enhance performance on tasks such as translation and text summarization [Data: Reports (1, 2)]

One popular implementation of the Transformer model is BERT (Bidirectional Encoder Representations from Transformers), which has been pre-trained on a vast corpus of text and can be fine-tuned for various NLP tasks [Data: Reports (2, 4)]

Transformer models have gained widespread adoption in various NLP applications due to their superior performance compared to traditional recurrent neural network architectures [Data: Reports (1, 3)]

However, Transformer models are computationally expensive due to their use of self-attention mechanisms. This has led to the development of efficient variants such as Longformer and Linformer [Data: Reports (5, 6)]
```
