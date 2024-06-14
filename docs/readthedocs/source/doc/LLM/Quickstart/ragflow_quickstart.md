# Run Ragflow

[ollama/ollama](https://github.com/ollama/ollama) is popular framework designed to build and run language models on a local machine; you can now use the C++ interface of [`ipex-llm`](https://github.com/intel-analytics/ipex-llm) as an accelerated backend for `ollama` running on Intel **GPU** *(e.g., local PC with iGPU, discrete GPU such as Arc, Flex and Max)*.

See the demo of running LLaMA2-7B on Intel Arc GPU below.

<video src="https://llm-assets.readthedocs.io/en/latest/_images/ollama-linux-arc.mp4" width="100%" controls></video>

```eval_rst
.. note::

  `ipex-llm[cpp]==2.5.0b20240527` is consistent with `v0.1.34 <https://github.com/ollama/ollama/releases/tag/v0.1.34>`_ of ollama.

  Our current version is consistent with `v0.1.39 <https://github.com/ollama/ollama/releases/tag/v0.1.39>`_ of ollama.
```

## Quickstart

### 0 Prerequisites

- CPU >= 4 cores
- RAM >= 16 GB
- Disk >= 50 GB
- Docker >= 24.0.0 & Docker Compose >= v2.26.1
- Ollama service initialized

### 1. Install and Run Ollama Serve

Visit [Run Ollama with IPEX-LLM on Intel GPU](./ollama_quickstart.html), and follow the steps 1) [Install IPEX-LLM for Ollama](./ollama_quickstart.html#install-ipex-llm-for-ollama), 2) [Initialize Ollama](./ollama_quickstart.html#initialize-ollama) 3) [Run Ollama Serve](./ollama_quickstart.html#run-ollama-serve) to install, init and start the Ollama Service. 


```eval_rst

.. tip::

  If your local LLM is running on Intel Arc™ A-Series Graphics with Linux OS (Kernel 6.2), it is recommended to additionaly set the following environment variable for optimal performance before executing `ollama serve`:

  .. code-block:: bash

      export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
```

### 2. Pull and Prepare the Model

#### 2.1 Pull Model 

Now we need to pull a model for coding. Here we use [Qwen/Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B) model as an example. Open a new terminal window, run the following command to pull [`qwen2:latest`](https://ollama.com/library/qwen2). 


```eval_rst
.. tabs::
   .. tab:: Linux

      .. code-block:: bash

         export no_proxy=localhost,127.0.0.1
         ./ollama pull qwen2:latest

   .. tab:: Windows

      Please run the following command in Miniforge Prompt.

      .. code-block:: cmd

         set no_proxy=localhost,127.0.0.1
         ollama pull qwen2:latest

.. seealso::

   Besides Qwen2, there are other coding models you might want to explore, such as Magicoder, Wizardcoder, Codellama, Codegemma, Starcoder, Starcoder2, and etc. You can find these models in the `Ollama model library <https://ollama.com/library>`_. Simply search for the model, pull it in a similar manner, and give it a try.
```

### 1 Initialize Ragflow

Ensure `vm.max_map_count` >= 262144:

> To check the value of `vm.max_map_count`:
>
> ```bash
> $ sysctl vm.max_map_count
> ```
>
> Reset `vm.max_map_count` to a value at least 262144 if it is not.
>
> ```bash
> # In this case, we set it to 262144:
> $ sudo sysctl -w vm.max_map_count=262144
> ```
>
> This change will be reset after a system reboot. To ensure your change remains permanent, add or update the `vm.max_map_count` value in **/etc/sysctl.conf** accordingly:
>
> ```bash
> vm.max_map_count=262144
> ```

Clone the repo:

```bash
$ git clone https://github.com/infiniflow/ragflow.git
```

### 2. Start up Ragflow server from Docker

Build the pre-built Docker images and start up the server:

> Running the following commands automatically downloads the *dev* version RAGFlow Docker image. To download and run a specified Docker version, update `RAGFLOW_VERSION` in **docker/.env** to the intended version, for example `RAGFLOW_VERSION=v0.7.0`, before running the following commands.

```bash
$ export no_proxy=localhost,127.0.0.1
$ cd ragflow/docker
$ chmod +x ./entrypoint.sh
$ docker compose up -d
```


> The core image is about 9 GB in size and may take a while to load.

Check the server status after having the server up and running:

```bash
$ docker logs -f ragflow-server
```

_The following output confirms a successful launch of the system:_

```bash
    ____                 ______ __
   / __ \ ____ _ ____ _ / ____// /____  _      __
  / /_/ // __ `// __ `// /_   / // __ \| | /| / /
 / _, _// /_/ // /_/ // __/  / // /_/ /| |/ |/ /
/_/ |_| \__,_/ \__, //_/    /_/ \____/ |__/|__/
              /____/

* Running on all addresses (0.0.0.0)
* Running on http://127.0.0.1:9380
* Running on http://x.x.x.x:9380
INFO:werkzeug:Press CTRL+C to quit
```
> If you skip this confirmation step and directly log in to RAGFlow, your browser may prompt a `network anomaly` error because, at that moment, your RAGFlow may not be fully initialized.  

In your web browser, enter the IP address of your server and log in to RAGFlow.
> With the default settings, you only need to enter `http://IP_OF_YOUR_MACHINE` (**sans** port number) as the default HTTP serving port `80` can be omitted when using the default configurations.
In [service_conf.yaml](./docker/service_conf.yaml), select the desired LLM factory in `user_default_llm` and update the `API_KEY` field with the corresponding API key.

> See [llm_api_key_setup](https://ragflow.io/docs/dev/llm_api_key_setup) for more information.

### 3 Run Ollama Serve

You may launch the Ollama service as below:

```eval_rst
.. tabs::
   .. tab:: Linux

      .. code-block:: bash

         export OLLAMA_NUM_GPU=999
         export no_proxy=localhost,127.0.0.1
         export ZES_ENABLE_SYSMAN=1
         source /opt/intel/oneapi/setvars.sh
         export SYCL_CACHE_PERSISTENT=1

         ./ollama serve

   .. tab:: Windows

      Please run the following command in Miniforge Prompt.

      .. code-block:: bash

         set OLLAMA_NUM_GPU=999
         set no_proxy=localhost,127.0.0.1
         set ZES_ENABLE_SYSMAN=1
         set SYCL_CACHE_PERSISTENT=1

         ollama serve

```

```eval_rst
.. note::

  Please set environment variable ``OLLAMA_NUM_GPU`` to ``999`` to make sure all layers of your model are running on Intel GPU, otherwise, some layers may run on CPU.
```

```eval_rst
.. tip::

  If your local LLM is running on Intel Arc™ A-Series Graphics with Linux OS (Kernel 6.2), it is recommended to additionaly set the following environment variable for optimal performance before executing `ollama serve`:

  .. code-block:: bash

      export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1

```

```eval_rst
.. note::

  To allow the service to accept connections from all IP addresses, use `OLLAMA_HOST=0.0.0.0 ./ollama serve` instead of just `./ollama serve`.
```

The console will display messages similar to the following:

<a href="https://llm-assets.readthedocs.io/en/latest/_images/ollama_serve.png" target="_blank">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/ollama_serve.png" width=100%; />
</a>


### 4 Pull Model
Keep the Ollama service on and open another terminal and run `./ollama pull <model_name>` in Linux (`ollama.exe pull <model_name>` in Windows) to automatically pull a model. e.g. `dolphin-phi:latest`:

<a href="https://llm-assets.readthedocs.io/en/latest/_images/ollama_pull.png" target="_blank">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/ollama_pull.png" width=100%; />
</a>


### 5 Using Ollama

#### Using Curl 

Using `curl` is the easiest way to verify the API service and model. Execute the following commands in a terminal. **Replace the <model_name> with your pulled 
model**, e.g. `dolphin-phi`.

```eval_rst
.. tabs::
   .. tab:: Linux

      .. code-block:: bash

         curl http://localhost:11434/api/generate -d '
         { 
            "model": "<model_name>", 
            "prompt": "Why is the sky blue?", 
            "stream": false
         }'

   .. tab:: Windows

      Please run the following command in Miniforge Prompt.

      .. code-block:: bash

         curl http://localhost:11434/api/generate -d "
         {
            \"model\": \"<model_name>\",
            \"prompt\": \"Why is the sky blue?\",
            \"stream\": false
         }"

```


#### Using Ollama Run GGUF models

Ollama supports importing GGUF models in the Modelfile, for example, suppose you have downloaded a `mistral-7b-instruct-v0.1.Q4_K_M.gguf` from [Mistral-7B-Instruct-v0.1-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/tree/main), then you can create a file named `Modelfile`:

```bash
FROM ./mistral-7b-instruct-v0.1.Q4_K_M.gguf
TEMPLATE [INST] {{ .Prompt }} [/INST]
PARAMETER num_predict 64
```

Then you can create the model in Ollama by `ollama create example -f Modelfile` and use `ollama run` to run the model directly on console.

```eval_rst
.. tabs::
   .. tab:: Linux

      .. code-block:: bash

         export no_proxy=localhost,127.0.0.1
         ./ollama create example -f Modelfile
         ./ollama run example

   .. tab:: Windows

      Please run the following command in Miniforge Prompt.

      .. code-block:: bash

         set no_proxy=localhost,127.0.0.1
         ollama create example -f Modelfile
         ollama run example

```

An example process of interacting with model with `ollama run example` looks like the following:

<a href="https://llm-assets.readthedocs.io/en/latest/_images/ollama_gguf_demo_image.png" target="_blank">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/ollama_gguf_demo_image.png" width=100%; />
</a>