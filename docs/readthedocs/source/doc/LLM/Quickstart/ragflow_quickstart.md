# Run RAGFlow with IPEX_LLM on Intel GPU

[RAGFlow](https://github.com/infiniflow/ragflow) is an open-source RAG (Retrieval-Augmented Generation) engine based on deep document understanding; by integrating it with [`ipex-llm`](https://github.com/intel-analytics/ipex-llm), users can now easily leverage local LLMs running on Intel GPU (e.g., local PC with iGPU, discrete GPU such as Arc, Flex and Max).


*See the demo of ragflow running Qwen2:7B on Intel Arc A770 below.*

<video src="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-record.mp4" width="100%" controls></video>


## Quickstart

### 0 Prerequisites

- CPU >= 4 cores
- RAM >= 16 GB
- Disk >= 50 GB
- Docker >= 24.0.0 & Docker Compose >= v2.26.1


### 1. Install and Start `Ollama` Service on Intel GPU 

Follow the steps in [Run Ollama with IPEX-LLM on Intel GPU Guide](./ollama_quickstart.md) to install and run Ollama on Intel GPU. Ensure that `ollama serve` is running correctly and can be accessed through a local URL (e.g., `https://127.0.0.1:11434`) or a remote URL (e.g., `http://your_ip:11434`).



```eval_rst
.. important::

   If the `RAGFlow` is not deployed on the same machine where Ollama is running (which means `RAGFlow` needs to connect to a remote Ollama service), you must configure the Ollama service to accept connections from any IP address. To achieve this, set or export the environment variable `OLLAMA_HOST=0.0.0.0` before executing the command `ollama serve`.

.. tip::

  If your local LLM is running on Intel Arc™ A-Series Graphics with Linux OS (Kernel 6.2), it is recommended to additionaly set the following environment variable for optimal performance before executing `ollama serve`:

  .. code-block:: bash

      export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
```

### 2. Pull Model

Now we need to pull a model for RAG using Ollama. Here we use [Qwen/Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B) model as an example. Open a new terminal window, run the following command to pull [`qwen2:latest`](https://ollama.com/library/qwen2). 


```eval_rst
.. tabs::
   .. tab:: Linux

      .. code-block:: bash

         export no_proxy=localhost,127.0.0.1
         ./ollama pull qwen2:latest

   .. tab:: Windows

      Please run the following command in Miniforge or Anaconda Prompt.

      .. code-block:: cmd

         set no_proxy=localhost,127.0.0.1
         ollama pull qwen2:latest

.. seealso::

   Besides Qwen2, there are other LLM models you might want to explore, such as Llama3, Phi3, Mistral, etc. You can find all available models in the `Ollama model library <https://ollama.com/library>`_. Simply search for the model, pull it in a similar manner, and give it a try.
```

### 3. Start `RAGFlow` Service


#### 3.1 Download `RAGFlow`

You can either clone the repository or download the source zip from [github](https://github.com/infiniflow/ragflow/archive/refs/heads/main.zip):

```bash
$ git clone https://github.com/infiniflow/ragflow.git
```

#### 3.2 Environment Settings

Ensure `vm.max_map_count` is set to at least 262144. To check the current value of `vm.max_map_count`, use:

```bash
$ sysctl vm.max_map_count
```

##### Changing `vm.max_map_count`

To set the value temporarily, use:

```bash
$ sudo sysctl -w vm.max_map_count=262144
```

To make the change permanent and ensure it persists after a reboot, add or update the following line in `/etc/sysctl.conf`:

```bash
vm.max_map_count=262144
```

### 3.3 Start the `RAGFlow` server using Docker

Build the pre-built Docker images and start up the server:

```eval_rst
.. note::

  Running the following commands automatically downloads the *dev* version RAGFlow Docker image. To download and run a specified Docker version, update `RAGFLOW_VERSION` in **docker/.env** to the intended version, for example `RAGFLOW_VERSION=v0.7.0`, before running the following commands.
```

```bash
$ export no_proxy=localhost,127.0.0.1
$ cd ragflow/docker
$ chmod +x ./entrypoint.sh
$ docker compose up -d
```

```eval_rst
.. note::
  
  The core image is about 9 GB in size and may take a while to load.
```

Check the server status after having the server up and running:

```bash
$ docker logs -f ragflow-server
```

Upon successful deployment, you will see logs in the terminal similar to the following:

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


Open a browser and navigate to the URL displayed in the terminal logs. Look for messages like `Running on http://ip:port`. For local deployment, you can usually access the web portal at `http://127.0.0.1:9380`. For remote access, use `http://your_ip:9380`.


### 4. Using `RAGFlow`

```eval_rst
.. note::

  For detailed information about how to use RAGFlow, visit the README of `RAGFlow official repository <https://github.com/infiniflow/ragflow>`_.

```

#### Log-in

If this is your first time using RAGFlow, you will need to register. After registering, log in with your new account to access the portal.

<div style="display: flex; gap: 5px;">
  <a href="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-login.png" target="_blank" style="flex: 1;">
    <img src="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-login.png" style="width: 100%;" />
  </a>
  <a href="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-login2.png" target="_blank" style="flex: 1;">
    <img src="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-login2.png" style="width: 100%;" />
  </a>
</div>


#### Configure `Ollama` service URL

Access the Ollama settings through **Settings -> Model Providers** in the menu. Fill out the **Base URL**, and then click the **OK** button at the bottom.


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

Go to **Knowledge Base** by clicking on **Knowledge Base** in the top bar. Click the **+Create knowledge base** button on the right. You will be prompted to input a name for the knowledge base.


<a href="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-knowledgebase.png" target="_blank">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-knowledgebase.png" width="100%" />
</a>

#### Edit Knowledge Base

After entering a name, you will be directed to edit the knowledge base. Click on **Dataset** on the left, then click **+ Add file -> Local files**. Upload your file in the pop-up window and click **OK**.

<div style="display: flex; gap: 5px;">
  <a href="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-knowledgebase2.png" target="_blank" style="flex: 1;">
    <img src="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-knowledgebase2.png" style="width: 100%;" />
  </a>
  <a href="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-knowledgebase3.png" target="_blank" style="flex: 1;">
    <img src="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-knowledgebase3.png" style="width: 100%;" />
  </a>
</div>

After the upload is successful, you will see a new record in the dataset. The _**Parsing Status**_ column will show `UNSTARTED`. Click the green start button in the _**Action**_ column to begin file parsing. Once parsing is finished, the _**Parsing Status**_ column will change to **SUCCESS**.

<div style="display: flex; gap: 5px;">
  <a href="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-knowledgebase4.png" target="_blank" style="flex: 1;">
    <img src="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-knowledgebase4.png" style="width: 100%;" />
  </a>
  <a href="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-knowledgebase5.png" target="_blank" style="flex: 1;">
    <img src="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-knowledgebase5.png" style="width: 100%;" />
  </a>
</div>


Next, go to **Configuration** on the left menu and click **Save** at the bottom to save the changes.

<a href="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-knowledgebase6.png" target="_blank">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-knowledgebase6.png" width="100%" />
</a>

#### Chat with the Model

Start new conversations by clicking **Chat** in the top navbar.

On the left side, create a conversation by clicking **Create an Assistant**. Under **Assistant Setting**, give it a name and select your knowledge bases.


  <a href="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-chat.png" target="_blank">
    <img src="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-chat.png" width="100%" />
  </a>


Next, go to **Model Setting**, choose your model added by Ollama, and disable the **Max Tokens** toggle. Finally, click **OK** to start.

```eval_rst
.. tip::

  Enabling the **Max Tokens** toggle may result in very short answers.
```

  <a href="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-chat2.png" target="_blank">
    <img src="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-chat2.png" width="100%" />
  </a> 

<br/>

Input your questions into the **Message Resume Assistant** textbox at the bottom, and click the button on the right to get responses.

<a href="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-chat3.png" target="_blank">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-chat3.png" width="100%" />
</a>

#### Exit

To shut down the RAGFlow server, use **Ctrl+C** in the terminal where the Ragflow server is runing, then close your browser tab.
