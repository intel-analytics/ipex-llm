# Run Open-Webui with Ollama on Linux with Intel GPU

The [open-webui](https://github.com/open-webui/open-webui) provides a user friendly GUI for anyone to run LLM locally; by porting it to [ollama](https://github.com/ollama/ollama) integrated with [`ipex-llm`](https://github.com/intel-analytics/ipex-llm), users can now easily run LLM in [open-webui](https://github.com/open-webui/open-webui) on Intel **GPU** *(e.g., local PC with iGPU, discrete GPU such as Arc, Flex and Max)*.

See the demo of running Mistral:7B on Intel Arc A770 below.

<video src="https://llm-assets.readthedocs.io/en/latest/_images/open_webui_demo.mp4" width="100%" controls></video>

## Quickstart
This quickstart guide walks you through setting up and using the [open-webui](https://github.com/open-webui/open-webui) with Ollama. 


### 1 Run Ollama on Linux with Intel GPU

To use the open-webui on Intel GPU, first ensure that you can run Ollama on Intel GPU. Follow the instructions on the [Run Ollama on Linux with Intel GPU](ollama_quickstart.md). Please keep ollama service running after completing the above steps.

### 2 Install and Run Open-Webui

**Requirements** 

- Node.js (>= 20.10) or Bun (>= 1.0.21)
- Python (>= 3.11)

**Installation Steps**

```sh
# Optional: Use Hugging Face mirror for restricted areas
export HF_ENDPOINT=https://hf-mirror.com

export no_proxy=localhost,127.0.0.1

git clone https://github.com/open-webui/open-webui.git
cd open-webui/
cp -RPp .env.example .env  # Copy required .env file

# Build Frontend
npm i
npm run build

# Serve Frontend with Backend
cd ./backend
pip install -r requirements.txt -U
bash start.sh
```

open-webui will be accessible at http://localhost:8080/.

> For detailed information, visit the [open-webui official repository](https://github.com/open-webui/open-webui).
>

### 3. Using Open-Webui

#### Log-in and Pull Model

If this is your first time using it, you need to register. After registering, log in with the registered account to access the interface.

<a href="https://llm-assets.readthedocs.io/en/latest/_images/open_webui_signup.png" target="_blank">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/open_webui_signup.png" width="100%" />
</a>


<a href="https://llm-assets.readthedocs.io/en/latest/_images/open_webui_login.png" target="_blank">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/open_webui_login.png" width="100%" />
</a>

Check your ollama service connection in `Settings`. The default Ollama Base URL is set to `https://localhost:11434`, you can also set your own url if you run Ollama service on another machine. Click this button to check if the Ollama service connection is functioning properly. If not, an alert will pop out as the below shows.

<a href="https://llm-assets.readthedocs.io/en/latest/_images/open_webui_settings_0.png" target="_blank">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/open_webui_settings_0.png" width="100%" />
</a>

If everything goes well, you will get a message as shown below.

<a href="https://llm-assets.readthedocs.io/en/latest/_images/open_webui_settings.png" target="_blank">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/open_webui_settings.png" width="100%" />
</a>

Pull model in `Settings/Models`, click the download button and ollama will download the model you select automatically.

<a href="https://llm-assets.readthedocs.io/en/latest/_images/open_webui_pull_models.png" target="_blank">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/open_webui_pull_models.png" width="100%" />
</a>

#### Chat with the Model

Start new conversations with **New chat**. Select a downloaded model here:

<a href="https://llm-assets.readthedocs.io/en/latest/_images/open_webui_select_model.png" target="_blank">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/open_webui_select_model.png" width="100%" />
</a>

Enter prompts into the textbox at the bottom and press the send button to receive responses.

<a href="https://llm-assets.readthedocs.io/en/latest/_images/open_webui_chat_1.png" target="_blank">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/open_webui_chat_1.png" width="100%" />
</a>

You can also drop files into the textbox for LLM to read.

<a href="https://llm-assets.readthedocs.io/en/latest/_images/open_webui_chat_2.png" target="_blank">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/open_webui_chat_2.png" width="100%" />
</a>

#### Exit Open-Webui

To shut down the open-webui server, use **Ctrl+C** in the terminal where the open-webui server is runing, then close your browser tab.
