# Run GraphRAG with IPEX-LLM on Intel GPU

The [GraphRAG project](https://github.com/microsoft/graphrag) is designed to leverage large language models (LLMs) for extracting structured and meaningful data from unstructured texts; by integrating it with [`ipex-llm`](https://github.com/intel-analytics/ipex-llm), users can now easily utilize local LLMs running on Intel GPU (e.g., local PC with iGPU, discrete GPU such as Arc, Flex and Max).

## Table of Contents

- [Install and Start `Ollama` Service on Intel GPU](#1-install-and-start-ollama-service-on-intel-gpu)
- [Prepare LLM and Embedding Model](#2-prepare-llm-and-embedding-model)
- [Setup Python Environment for GraphRAG](#3-setup-python-environment-for-graphrag)
- [Index GraphRAG](#4-index-graphrag)
- [Query GraphRAG](#5-query-graphrag)
- [Query GraphRAG](#5-query-graphrag)
- [Troubleshooting](#troubleshooting)

## Quickstart

### 1. Install and Start `Ollama` Service on Intel GPU 

Follow the steps in [Run Ollama with IPEX-LLM on Intel GPU Guide](./ollama_quickstart.md) to install `ipex-llm[cpp]==2.1.0` and run Ollama on Intel GPU. Ensure that `ollama serve` is running correctly and can be accessed through a local URL (e.g., `https://127.0.0.1:11434`).

> [!NOTE]
> Please note that for GraphRAG, we highly recommand using the stable version of ipex-llm through `pip install ipex-llm[cpp]==2.1.0`.

### 2. Prepare LLM and Embedding Model

In another terminal window, separate from where you executed `ollama serve`, download the LLM and embedding model using the following commands:

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
> Here we take [`mistral`](https://ollama.com/library/mistral) and [`nomic-embed-text`](https://ollama.com/library/nomic-embed-text) as an example. You could have a try on other LLMs or embedding models in [`ollama.com`](https://ollama.com/search?p=1).

### 3. Setup Python Environment for GraphRAG

To run the LLM and embedding model on a local machine, we utilize the [`graphrag-local-ollama`](https://github.com/TheAiSingularity/graphrag-local-ollama) repository:

```shell
git clone https://github.com/TheAiSingularity/graphrag-local-ollama.git
cd graphrag-local-ollama

conda create -n graphrag-local-ollama python=3.10
conda activate graphrag-local-ollama

pip install -e .
pip install future

pip install ollama
pip install plotly
```

in which `pip install ollama` is for enabling restful APIs through python, and `pip install plotly` is for visualizing the knowledge graph.

> [!NOTE]
> Please note that the Python environment for GraphRAG setup here is separate from the one for Ollama server on Intel GPUs.

### 4. Index GraphRAG

The environment is now ready for GraphRAG with local LLMs and embedding models running on Intel GPUs. Before querying GraphRAG, it is necessary to first index GraphRAG, which could be a resource-intensive operation.

> [!TIP]
> Refer to [here](https://microsoft.github.io/graphrag/) for more details in GraphRAG process explanation.

#### Prepare Input Corpus

Some [sample documents](https://github.com/TheAiSingularity/graphrag-local-ollama/tree/main/input) are used here as input corpus for indexing GraphRAG, based on which LLM will create a knowledge graph.

Perpare the input corpus, and then initialize the workspace:

- For **Linux users**:

  ```bash
  # define inputs corpus
  mkdir -p ./ragtest/input
  cp input/* ./ragtest/input

  export no_proxy=localhost,127.0.0.1

  # initialize ragtest folder
  python -m graphrag.index --init --root ./ragtest

  # prepare settings.yml, please make sure the initialized settings.yml in ragtest folder is replaced by settings.yml in graphrag-ollama-local folder
  cp settings.yaml ./ragtest
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
  copy settings.yaml .\ragtest /y
  ```

#### Update `settings.yml`

In the `settings.yml` file inside the `ragtest` folder, add the configuration `request_timeout: 1800.0` for `llm`. Besides, if you would like to use LLMs or embedding models other than `mistral` or `nomic-embed-text`, you are required to update the `settings.yml` in `ragtest` folder accordingly:


```yml
llm:
  api_key: ${GRAPHRAG_API_KEY}
  type: openai_chat
  model: mistral # change it accordingly if using another LLM
  model_supports_json: true
  request_timeout: 1800.0 # add this configuration; you could also increase the request_timeout
  api_base: http://localhost:11434/v1

embeddings:
  async_mode: threaded
  llm:
    api_key: ${GRAPHRAG_API_KEY}
    type: openai_embedding
    model: nomic_embed_text # change it accordingly if using another embedding model
    api_base: http://localhost:11434/api
```

#### Conduct GraphRAG indexing

Finally, conduct GraphRAG indexing, which may take a while:

```shell
python -m graphrag.index --root ragtest
```

You will got message `🚀 All workflows completed successfully.` after the GraphRAG indexing is successfully finished.

#### (Optional) Visualize Knowledge Graph

For a clearer view of the knowledge graph, you can visualize it by specifying the path to the `.graphml` file in the `visualize-graphml.py` script, like below:

- For **Linux users**:

  ```python
  graph = nx.read_graphml('ragtest/output/20240715-151518/artifacts/summarized_graph.graphml') 
  ```

- For **Windows users**:

  ```python
  graph = nx.read_graphml('ragtest\\output\\20240715-151518\\artifacts\\summarized_graph.graphml') 
  ```

and run the following command to interactively visualize the knowledge graph:

```shell
python visualize-graphml.py
```

<table width="100%">
  <tr>
    <td><a href="https://llm-assets.readthedocs.io/en/latest/_images/knowledge_graph_1.png"><img src="https://llm-assets.readthedocs.io/en/latest/_images/knowledge_graph_1.png"/></a></td>
    <td><a href="https://llm-assets.readthedocs.io/en/latest/_images/knowledge_graph_2.png"><img src="https://llm-assets.readthedocs.io/en/latest/_images/knowledge_graph_2.png"/></a></td>
</tr>
</table>

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
creating llm client with {'api_key': 'REDACTED,len=9', 'type': "openai_chat", 'model': 'mistral', 'max_tokens': 4000, 'temperature': 0.0, 'top_p': 1.0, 'request_timeout': 1800.0, 'api_base': 'http://localhost:11434/v1', 'api_version': None, 'organization': None, 'proxy': None, 'cognitive_services_endpoint': None, 'deployment_name': None, 'model_supports_json': True, 'tokens_per_minute': 0, 'requests_per_minute': 0, 'max_retries': 10, 'max_retry_wait': 10.0, 'sleep_on_rate_limit_recommendation': True, 'concurrent_requests': 25}

SUCCESS: Global Search Response:  The Transformer is a type of neural network architecture primarily used for sequence-to-sequence tasks, such as machine translation and text summarization [Data: Reports (1, 2, 34, 46, 64, +more)]. It was introduced in the paper 'Attention is All You Need' by Vaswani et al. The Transformer model uses self-attention mechanisms to focus on relevant parts of the input sequence when generating an output.

The Transformer architecture was designed to overcome some limitations of Recurrent Neural Networks (RNNs), such as the vanishing gradient problem and the need for long sequences to be processed sequentially [Data: Reports (1, 2)]. The Transformer model processes input sequences in parallel, making it more efficient for handling long sequences.

The Transformer model has been very successful in various natural language processing tasks, such as machine translation and text summarization [Data: Reports (1, 34, 46, 64, +more)]. It has also been applied to other domains, such as computer vision and speech recognition. The key components of the Transformer architecture include self-attention layers, position-wise feedforward networks, and multi-head attention mechanisms [Data: Reports (1, 2, 3)].

Since its initial introduction, the Transformer model has been further developed and improved upon. Variants of the Transformer architecture, such as BERT (Bidirectional Encoder Representations from Transformers) and RoBERTa (Robustly Optimized BERT Pretraining Approach), have achieved state-of-the-art performance on a wide range of natural language processing tasks [Data: Reports (1, 2, 34, 46, 64, +more)].
```

### Troubleshooting

#### `failed to find free space in the KV cache, retrying with smaller n_batch` when conducting GraphRAG Indexing, and `JSONDecodeError` when querying GraphRAG

If you observe the Ollama server log showing `failed to find free space in the KV cache, retrying with smaller n_batch` while conducting GraphRAG indexing, and receive `JSONDecodeError` when querying GraphRAG, try to increase context length for the LLM model and index/query GraphRAG again.

Here introduce how to make the LLM model support larger context. To do this, we need to first create a file named `Modelfile`:

```
FROM mistral:latest
PARAMETER num_ctx 4096
```

> [!TIP]
> Here we increase `num_ctx` to 4096 as an example. You could adjust it accordingly.

and then use the following commands to create a new model in Ollama named `mistral:latest-nctx4096`:

- For **Linux users**:

  ```bash
  ./ollama create mistral:latest-nctx4096 -f Modelfile
  ```

- For **Windows users**:

  Please run the following command in Miniforge or Anaconda Prompt.

  ```cmd
  ollama create mistral:latest-nctx4096 -f Modelfile
  ```

Finally, update `settings.yml` inside the `ragtest` folder to use `llm` model `mistral:latest-nctx4096`:

```yml
llm:
  api_key: ${GRAPHRAG_API_KEY}
  type: openai_chat
  model: mistral:latest-nctx4096 # change it accordingly if using another LLM, or LLM model with larger num_ctx
  model_supports_json: true
  request_timeout: 1800.0 # add this configuration; you could also increase the request_timeout
  api_base: http://localhost:11434/v1

embeddings:
  async_mode: threaded
  llm:
    api_key: ${GRAPHRAG_API_KEY}
    type: openai_embedding
    model: nomic_embed_text # change it accordingly if using another embedding model
    api_base: http://localhost:11434/api
```