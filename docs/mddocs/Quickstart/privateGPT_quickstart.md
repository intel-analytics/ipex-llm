# Run PrivateGPT with IPEX-LLM on Intel GPU

[PrivateGPT](https://github.com/zylon-ai/private-gpt) is a production-ready AI project that allows users to chat over documents, etc.; by integrating it with [`ipex-llm`](https://github.com/intel-analytics/ipex-llm), users can now easily leverage local LLMs running on Intel GPU (e.g., local PC with iGPU, discrete GPU such as Arc, Flex and Max).

*See the demo of privateGPT running Mistral:7B on Intel Arc A770 below.*


<video src="https://llm-assets.readthedocs.io/en/latest/_images/PrivateGPT-ARC.mp4" width="100%" controls></video>

## Quickstart


### 1. Install and Start `Ollama` Service on Intel GPU 

Follow the steps in [Run Ollama on Intel GPU Guide](./ollama_quickstart.md) to install and run Ollama on Intel GPU. Ensure that `ollama serve` is running correctly and can be accessed through a local URL (e.g., `https://127.0.0.1:11434`) or a remote URL (e.g., `http://your_ip:11434`).

We recommend pulling the desired model before proceeding with PrivateGPT. For instance, to pull the Mistral:7B model, you can use the following command:

```bash
ollama pull mistral:7b
```

### 2. Install PrivateGPT

#### Download PrivateGPT

You can either clone the repository or download the source zip from [github](https://github.com/zylon-ai/private-gpt/archive/refs/heads/main.zip):
```bash
git clone https://github.com/zylon-ai/private-gpt
```

#### Install Dependencies

Execute the following commands in a terminal to install the dependencies of PrivateGPT:

```cmd
cd private-gpt
pip install poetry
pip install ffmpy==0.3.1
poetry install --extras "ui llms-ollama embeddings-ollama vector-stores-qdrant"
```
For more details, refer to the [PrivateGPT installation Guide](https://docs.privategpt.dev/installation/getting-started/main-concepts).


### 3. Start PrivateGPT

#### Configure PrivateGPT

To configure PrivateGPT to use Ollama for running local LLMs, you should edit the `private-gpt/settings-ollama.yaml` file. Modify the `ollama` section by setting the `llm_model` and `embedding_model` you wish to use, and updating the `api_base` and `embedding_api_base` to direct to your Ollama URL.

Below is an example of how `settings-ollama.yaml` should look.


<p align="center"><a href="https://llm-assets.readthedocs.io/en/latest/_images/privateGPT-ollama-setting.png" target="_blank" align="center">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/privateGPT-ollama-setting.png" alt="image-p1" width=100%; />
</a></p>


```eval_rst

.. note::

  `settings-ollama.yaml` is loaded when the Ollama profile is specified in the PGPT_PROFILES environment variable. This can override configurations from the default `settings.yaml`.

```

For more information on configuring PrivateGPT, please visit the [PrivateGPT Main Concepts](https://docs.privategpt.dev/installation/getting-started/main-concepts) page.


#### Start the service
Please ensure that the Ollama server continues to run in a terminal while you're using the PrivateGPT. 

Run below commands to start the service in another terminal:

```eval_rst
.. tabs::
  .. tab:: Linux

    .. code-block:: bash

       export no_proxy=localhost,127.0.0.1
       PGPT_PROFILES=ollama make run

    .. note:

       Setting ``PGPT_PROFILES=ollama`` will load the configuration from ``settings.yaml`` and ``settings-ollama.yaml``.

  .. tab:: Windows
    
    .. code-block:: bash
       
       set no_proxy=localhost,127.0.0.1
       set PGPT_PROFILES=ollama
       make run

   .. note:

       Setting ``PGPT_PROFILES=ollama`` will load the configuration from ``settings.yaml`` and ``settings-ollama.yaml``.
```

Upon successful deployment, you will see logs in the terminal similar to the following:

<p align="center"><a href="https://llm-assets.readthedocs.io/en/latest/_images/privateGPT-service-success.png" target="_blank" align="center">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/privateGPT-service-success.png" alt="image-p1" width=100%; />
</a></p>

Open a browser (if it doesn't open automatically) and navigate to the URL displayed in the terminal. If it shows http://0.0.0.0:8001, you can access it locally via `http://127.0.0.1:8001` or remotely via `http://your_ip:8001`.


### 4. Using PrivateGPT

#### Chat with the Model

To chat with the LLM, select the "LLM Chat" option located in the upper left corner of the page. Type your messages at the bottom of the page and click the "Submit" button to receive responses from the model.


<p align="center"><a href="https://llm-assets.readthedocs.io/en/latest/_images/privateGPT-LLM-Chat.png" target="_blank" align="center">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/privateGPT-LLM-Chat.png" alt="image-p1" width=100%; />
</a></p>



#### Chat over Documents (RAG)

To interact with documents, select the "Query Files" option in the upper left corner of the page. Click the "Upload File(s)" button to upload documents. After the documents have been vectorized, you can type your messages at the bottom of the page and click the "Submit" button to receive responses from the model based on the uploaded content.

<a href="https://llm-assets.readthedocs.io/en/latest/_images/privateGPT-Query-Files.png" target="_blank">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/privateGPT-Query-Files.png" width=100%; />
</a>
