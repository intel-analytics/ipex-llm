
# Run Coding Copilot on Windows with Intel GPU

[**Continue**](https://marketplace.visualstudio.com/items?itemName=Continue.continue) is a coding copilot extension in [Microsoft Visual Studio Code](https://code.visualstudio.com/); by porting it to [`ipex-llm`](https://github.com/intel-analytics/ipex-llm), users can now easily leverage local LLMs running on Intel GPU (e.g., local PC with iGPU, discrete GPU such as Arc, Flex and Max) for code explanation, code generation/completion, etc.

See the demos of using Continue with [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) running on Intel A770 GPU below.

<table border="1" width="100%">
  <tr>
    <td align="center">Code Generation</td>
    <td align="center">Code Explanation</td>
  </tr>
  <tr>
    <td><video src="https://llm-assets.readthedocs.io/en/latest/_images/code-gen.mp4" width="100%" controls></video></td>
    <td><video src="https://llm-assets.readthedocs.io/en/latest/_images/code-explanation.mp4" width="100%" controls></video></td>
</tr>
</table>




## Quickstart

This guide walks you through setting up and running **Continue** within _Visual Studio Code_, empowered by local large language models served via [Text Generation WebUI](https://github.com/intel-analytics/text-generation-webui/) with `ipex-llm` optimizations. 


### 1. Install and Run Text Generation WebUI

Visit [Run Text Generation WebUI Quickstart Guide](webui_quickstart.html), and follow the steps 1) [Install IPEX-LLM](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/webui_quickstart.html#install-ipex-llm), 2) [Install WebUI](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/webui_quickstart.html#install-the-webui) and 3) [Start the Server](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/webui_quickstart.html#start-the-webui-server) to install and start the Text Generation WebUI API Service. **Please pay attention to below items during installation:**

- The Text Generation WebUI API service requires Python version 3.10 or higher. We recommend use Python 3.11 as below:
  ```bash
  conda create -n llm python=3.11 libuv
  ```
- Remember to launch the server **with API service** as specified in [Launch the Server](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/webui_quickstart.html#launch-the-server)



### 2. Use WebUI to Load Model

#### Access the WebUI
Upon successful launch, URLs to access the WebUI will be displayed in the terminal as shown below. Open the provided local URL in your browser to interact with the WebUI. 

<a href="https://llm-assets.readthedocs.io/en/latest/_images/continue_quickstart_launch_server.jpeg" target="_blank">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/continue_quickstart_launch_server.jpeg" width=100%; />
</a>


#### Model Download and Loading

Here's a list of models that can be used for coding copilot on local PC. 
- Code Llama: 
- WizardCoder
- Mistral
- StarCoder
- DeepSeek Coder

Follow the steps in [Model Download](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/webui_quickstart.html#model-download) and [Load Model](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/webui_quickstart.html#load-model) to download and load your coding model. 

```eval_rst
.. note::

  If you don't need to use the API service anymore, you can follow the instructions in refer to `Exit WebUI <https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/webui_quickstart.html#exit-the-webui>`_  to stop the service.  
```



### 3. Install `Continue` Extension
1. Click `Install` on the [Continue extension in the Visual Studio Marketplace](https://marketplace.visualstudio.com/items?itemName=Continue.continue)
2. This will open the Continue extension page in VS Code, where you will need to click `Install` again
3. Once you do this, you will see the Continue logo show up on the left side bar. If you click it, the Continue extension will open up:

<a href="https://llm-assets.readthedocs.io/en/latest/_images/continue_quickstart_open_continue.png" target="_blank">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/continue_quickstart_open_continue.png" width=100%; />
</a>

```eval_rst
.. note::

   Note: We strongly recommend moving Continue to VS Code's right sidebar. This helps keep the file explorer open while using Continue, and the sidebar can be toggled with a simple keyboard shortcut.
```



### 4. Configure `Continue`

<a href="https://llm-assets.readthedocs.io/en/latest/_images/continue_quickstart_configuration.png" target="_blank">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/continue_quickstart_configuration.png" width=100%; />
</a>

Once you've started the API server, you can now use your local LLMs on Continue. After opening Continue(you can either click the extension icon on the left sidebar or press `Ctrl+Shift+L`), you can click the `+` button next to the model dropdown, and scroll down to the bottom and click `Open config.json`.

In `config.json`, you'll find the `models` property, a list of the models that you have saved to use with Continue. Please add the following configuration to `models`. Note that `model`, `apiKey`, `apiBase` should align with what you specified when starting the `Text Generation WebUI` server. Finally, remember to select this model in the model dropdown menu.
```json
{
  "models": [
    {
      "title": "Text Generation WebUI API Server",
      "provider": "openai",
      "model": "MODEL_NAME",
      "apiKey": "EMPTY",
      "apiBase": "http://localhost:5000/v1"
    }
  ]
}
```

### 5. How to Use `Continue`
For detailed tutorials please refer to [this link](https://continue.dev/docs/how-to-use-continue). Here we are only showing the most common scenarios.

#### Ask about highlighted code or an entire file
If you don't understand how some code works, highlight(press `Ctrl+Shift+L`) it and ask "how does this code work?"

<a href="https://llm-assets.readthedocs.io/en/latest/_images/continue_quickstart_sample_usage1.png" target="_blank">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/continue_quickstart_sample_usage1.png" width=100%; />
</a>

#### Editing existing code
You can ask Continue to edit your highlighted code with the command `/edit`.

<a href="https://llm-assets.readthedocs.io/en/latest/_images/continue_quickstart_sample_usage2.png" target="_blank">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/continue_quickstart_sample_usage2.png" width=100%; />
</a>


### Troubleshooting

#### Failed to load the extension `openai`

If you encounter `TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'` when you run `python server.py --load-in-4bit --api`, please make sure you are using `Python 3.11` instead of lower versions.




