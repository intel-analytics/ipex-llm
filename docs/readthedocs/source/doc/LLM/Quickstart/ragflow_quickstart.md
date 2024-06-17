# Run RAGFlow using Ollama with IPEX_LLM

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
.. important::

   If the `Ragflow` is not deployed on the same machine where Ollama is running (which means `Ragflow` needs to connect to a remote Ollama service), you must configure the Ollama service to accept connections from any IP address. To achieve this, set or export the environment variable `OLLAMA_HOST=0.0.0.0` before executing the command `ollama serve`.

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

### 3. Initialize Ragflow

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

### 4. Start up Ragflow server from Docker

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

### 5. Using the Ragflow

```eval_rst
.. note::

  For detailed information about how to use Open WebUI, visit the README of `open-webui official repository <https://github.com/open-webui/open-webui>`_.

```

#### Log-in

If this is your first time using it, you need to register. After registering, log in with the registered account to access the interface.

<a href="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-login.png" target="_blank">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-login.png" width="100%" />
</a>


<a href="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-login2.png" target="_blank">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-login2.png" width="100%" />
</a>

#### Configure `Ollama` service URL

Access the Ollama settings through **Settings -> Model Providers** in the menu. Fill out the  and **Base url**, and then hit the **OK** button at the bottom.


<a href="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-add-ollama.png" target="_blank">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-add-ollama.png" width="100%" />
</a>

If the connection is successful, you will see the model listed down **Show more models** as illustrated below.

<a href="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-add-ollama2.png" target="_blank">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-add-ollama2.png" width="100%" />
</a>

```eval_rst
.. note::

  If you want to use an Ollama server hosted at a different URL, simply update the **Ollama Base URL** to the new URL and press the **OK** button again to re-confirm the connection to Ollama. 
```

#### Create Knowledge Base
Go to **Knowledge Base** after clicking **Knowledge Base** at the top bar. Hit the **+Create knowledge base** button on the right. You will be prompted to input a name for the knowledge base. 

<a href="https://llm-assets.readthedocs.io/en/latest/_images/open_webui_settings.png" target="_blank">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/open_webui_settings.png" width="100%" />
</a>

#### Edit Knowledge Base
After inputting a name, you will be directed to edit the knowledge base. Hit the **Dataset** on the left, and then hit **+ Add file -> Local files**. Choose the file you want to train, and hit the green start button marked to start parsing the file. It will show **SUCCESS** when the parsing is completed. Then you can go to **Configuration** and hit **Save** at the bottom to save the changes.

#### Chat with the Model

Start new conversations with **Chat** at the top navbar. 

On the left-side, create a conversation by clicking **Create an Assistant**. Under **Assistant Setting**, give it a name and select your Knowledgebases. Then go to **Model Setting**, choose your model added by Ollama. Make sure disable the **Max Tokens** toggle and hit **OK** to start.

  <a href="https://llm-assets.readthedocs.io/en/latest/_images/open_webui_select_model.png" target="_blank">
    <img src="https://llm-assets.readthedocs.io/en/latest/_images/open_webui_select_model.png" width="100%" />
  </a> 


<br/>
Input your questions into the **Message Resume Assistant** textbox at the bottom, and click the button on the right to get responses.

<a href="https://llm-assets.readthedocs.io/en/latest/_images/open_webui_chat_2.png" target="_blank">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/open_webui_chat_2.png" width="100%" />
</a>

#### Exit Open-Webui

To shut down the open-webui server, use **Ctrl+C** in the terminal where the Ragflow server is runing, then close your browser tab.
