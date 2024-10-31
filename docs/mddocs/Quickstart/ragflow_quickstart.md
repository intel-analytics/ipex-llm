# Run RAGFlow with IPEX-LLM on Intel GPU

[RAGFlow](https://github.com/infiniflow/ragflow) is an open-source RAG (Retrieval-Augmented Generation) engine based on deep document understanding; by integrating it with [`ipex-llm`](https://github.com/intel-analytics/ipex-llm), users can now easily leverage local LLMs running on Intel GPU (e.g., local PC with iGPU, discrete GPU such as Arc, Flex and Max).


*See the demo of ragflow running Qwen2:7B on Intel Arc A770 below.*

<table width="100%">
  <tr>
    <td><a href="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-record.mp4"><img src="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-record.png"/></a></td>
  </tr>
  <tr>
    <td align="center">You could also click <a href="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-record.mp4">here</a> to watch the demo video.</td>
  </tr>
</table>


## Table of Contents
- [Prerequisites](./ragflow_quickstart.md#0-prerequisites)
- [Install and Start Ollama Service on Intel GPU](./ragflow_quickstart.md#1-install-and-start-ollama-service-on-intel-gpu)
- [Pull Model](./ragflow_quickstart.md#2-pull-model)
- [Start `RAGFlow` Service](./ragflow_quickstart.md#3-start-ragflow-service)
- [Using `RAGFlow`](./ragflow_quickstart.md#4-using-ragflow)
- [Troubleshooting](./ragflow_quickstart.md#5-troubleshooting)

## Quickstart

### 0. Prerequisites

- CPU >= 4 cores
- RAM >= 16 GB
- Disk >= 50 GB
- Docker >= 24.0.0 & Docker Compose >= v2.26.1

### 1. Install and Start `Ollama` Service on Intel GPU 

Follow the steps in [Run Ollama with IPEX-LLM on Intel GPU Guide](./ollama_quickstart.md) to install and run Ollama on Intel GPU. Ensure that `ollama serve` is running correctly and can be accessed through a local URL (e.g., `https://127.0.0.1:11434`) or a remote URL (e.g., `http://your_ip:11434`).

> [!IMPORTANT]
> If the `RAGFlow` is not deployed on the same machine where Ollama is running (which means `RAGFlow` needs to connect to a remote Ollama service), you must configure the Ollama service to accept connections from any IP address. To achieve this, set or export the environment variable `OLLAMA_HOST=0.0.0.0` before executing the command `ollama serve`.

### 2. Pull Model

Now we need to pull a model for RAG using Ollama. Here we use [Qwen/Qwen2-7B](https://huggingface.co/Qwen/Qwen2-7B) model as an example. Open a new terminal window, run the following command to pull [`qwen2:latest`](https://ollama.com/library/qwen2). 

- For **Linux users**:

  ```bash
  export no_proxy=localhost,127.0.0.1
  ./ollama pull qwen2:latest
  ```

- For **Windows users**:

  Please run the following command in Miniforge or Anaconda Prompt.

  ```cmd
  set no_proxy=localhost,127.0.0.1
  ollama pull qwen2:latest
  ```

> [!TIP]
> Besides Qwen2, there are other LLM models you might want to explore, such as Llama3, Phi3, Mistral, etc. You can find all available models in the [Ollama model library](https://ollama.com/library). Simply search for the model, pull it in a similar manner, and give it a try.

### 3. Start `RAGFlow` Service

> [!NOTE]
> The steps in section 3 is verified on Linux system only. 

#### 3.1 Download `RAGFlow`

You can either clone the repository or download the source zip from [github](https://github.com/infiniflow/ragflow/archive/refs/heads/main.zip):

```bash
git clone https://github.com/infiniflow/ragflow.git
```

#### 3.2 Environment Settings

Ensure `vm.max_map_count` is set to at least 262144. To check the current value of `vm.max_map_count`, use:

```bash
sysctl vm.max_map_count
```

##### Changing `vm.max_map_count`

To set the value temporarily, use:

```bash
sudo sysctl -w vm.max_map_count=262144
```

To make the change permanent and ensure it persists after a reboot, add or update the following line in `/etc/sysctl.conf`:

```bash
vm.max_map_count=262144
```

#### 3.3 Start the `RAGFlow` server using Docker

Build the pre-built Docker images and start up the server:

> [!NOTE]
> Running the following commands automatically downloads the *dev* version RAGFlow Docker image. To download and run a specified Docker version, update `RAGFLOW_VERSION` in **docker/.env** to the intended version, for example `RAGFLOW_VERSION=v0.7.0`, before running the following commands.

```bash
export no_proxy=localhost,127.0.0.1
cd ragflow/docker
chmod +x ./entrypoint.sh
docker compose up -d
```

> [!NOTE]
> The core image is about 9 GB in size and may take a while to load.

Check the server status after having the server up and running:

```bash
docker logs -f ragflow-server
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

You can now open a browser and access the RAGflow web portal. With the default settings, simply enter `http://IP_OF_YOUR_MACHINE` (without the port number), as the default HTTP serving port `80` can be omitted. If RAGflow is deployed on the same machine as your browser, you can also access the web portal at `http://127.0.0.1` or `http://localhost`.


### 4. Using `RAGFlow`

> [!NOTE]
> For detailed information about how to use RAGFlow, visit the README of [RAGFlow official repository](https://github.com/infiniflow/ragflow).

#### Log-in

If this is your first time using RAGFlow, you will need to register. After registering, log in with your new account to access the portal.

<table width="100%">
  <tr>
    <td><a href="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-login.png"><img src="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-login.png"/></a></td>
    <td><a href="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-login2.png"><img src="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-login2.png"/></a></td>
</tr>
</table>

#### Configure `Ollama` service URL

Access the Ollama settings through **Settings -> Model Providers** in the menu. Fill out the **Base URL**, and then click the **OK** button at the bottom.


<a href="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-add-ollama.png" target="_blank">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-add-ollama.png" width="100%" />
</a>

If the connection is successful, you will see the model listed down **Show more models** as illustrated below.

<a href="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-add-ollama2.png" target="_blank">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-add-ollama2.png" width="100%" />
</a>

> [!NOTE]
> If you want to use an Ollama server hosted at a different URL, simply update the **Ollama Base URL** to the new URL and press the **OK** button again to re-confirm the connection to Ollama. 

#### Create Knowledge Base

Go to **Knowledge Base** by clicking on **Knowledge Base** in the top bar. Click the **+Create knowledge base** button on the right. You will be prompted to input a name for the knowledge base.


<a href="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-knowledgebase.png" target="_blank">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-knowledgebase.png" width="100%" />
</a>

#### Edit Knowledge Base

After entering a name, you will be directed to edit the knowledge base. Click on **Dataset** on the left, then click **+ Add file -> Local files**. Upload your file in the pop-up window and click **OK**.

<table width="100%">
  <tr>
    <td><a href="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-knowledgebase2.png"><img src="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-knowledgebase2.png"/></a></td>
    <td><a href="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-knowledgebase3.png"><img src="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-knowledgebase3.png"/></a></td>
</tr>
</table>

After the upload is successful, you will see a new record in the dataset. The _**Parsing Status**_ column will show `UNSTARTED`. Click the green start button in the _**Action**_ column to begin file parsing. Once parsing is finished, the _**Parsing Status**_ column will change to **SUCCESS**.

<table width="100%">
  <tr>
    <td><a href="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-knowledgebase4.pngg"><img src="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-knowledgebase4.png"/></a></td>
    <td><a href="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-knowledgebase5.png"><img src="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-knowledgebase5.png"/></a></td>
</tr>
</table>

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

> [!TIP]
> Enabling the **Max Tokens** toggle may result in very short answers.

<a href="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-chat2.png" target="_blank">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-chat2.png" width="100%" />
</a> 

Input your questions into the **Message Resume Assistant** textbox at the bottom, and click the button on the right to get responses.

<a href="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-chat3.png" target="_blank">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/ragflow-chat3.png" width="100%" />
</a>

#### Exit

To shut down the RAGFlow server, use **Ctrl+C** in the terminal where the Ragflow server is runing, then close your browser tab.

### 5. Troubleshooting

#### Stuck when parsing files `Node <Urllib3HttpNode(http://es01:9200)> has failed for xx times in a row, putting on 30 second timeout`

This is because there's no enough space on the disk and the Docker container stop working. Please left enough space on the disk and make sure the disk usage is below 90%.

#### `Max retries exceeded with url: /encodings/cl100k_base.tiktoken` while starting the RAGFlow service through Docker

This may caused by network problem. To resolve this, you could try to:

1. Attach to the Docker container by `docker exec -it ragflow-server /bin/bash`
2. Set environment variables like `HTTP_PROXY`, `HTTPS_PROXY`, and `NO_PROXY` at the beginning of the `/ragflow/entrypoint.sh`.
3. Stop the service by `docker compose stop`.
4. Restart the service by `docker compose start`.
