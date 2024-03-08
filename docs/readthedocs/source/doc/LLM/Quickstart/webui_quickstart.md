
# Use Text Generation WebUI on Windows with Intel GPU

This quickstart guide walks you through setting up and using the [Text Generation WebUI](https://github.com/intel-analytics/text-generation-webui) (a Gradio WebUI for running Large Language Models) with `bigdl-llm`. 


A preview of the WebUI in action is shown below:

<img src="https://llm-assets.readthedocs.io/en/latest/_images/webui_quickstart_chat.png" width=80%; />



## 1 Install BigDL-LLM

To use the WebUI, first ensure that BigDL-LLM is installed. Follow the instructions on the [BigDL-LLM Installation Quickstart for Windows with Intel GPU](install_windows_gpu.html). 

**After the installation, you should have created a conda environment, named `llm` for instance, for running `bigdl-llm` applications.**

## 2 Install the WebUI


### Download the WebUI
Download the `text-generation-webui` with BigDL-LLM integrations from [this link](https://github.com/intel-analytics/text-generation-webui/archive/refs/heads/bigdl-llm.zip). Unzip the content into a directory, e.g.,`C:\text-generation-webui`. 
  
### Install Dependencies

Open **Anaconda Prompt** and activate the conda environment you have created in [section 1](#1-install-bigdl-llm), e.g., `llm`. 
```
conda activate llm
```
Then, change to the directory of WebUI (e.g.,`C:\text-generation-webui`) and install the necessary dependencies:
```cmd
cd C:\text-generation-webui
pip install -r requirements_cpu_only.txt
```

## 3 Start the WebUI Server

### Set Environment Variables
Configure oneAPI variables by running the following command in **Anaconda Prompt**:
> Note: For more details about runtime configurations, refer to [this guide](https://bigdl.readthedocs.io/en/latest/doc/LLM/Overview/install_gpu.html#runtime-configuration):
```cmd
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
```
If you're running on iGPU, set additional environment variables by running the following commands:
```cmd
set SYCL_CACHE_PERSISTENT=1
set BIGDL_LLM_XMX_DISABLED=1
```

### Launch the Server
In **Anaconda Prompt** with the conda environment `llm` activated, navigate to the text-generation-webui folder and start the server using the following command:
  > Note: with `--load-in-4bit` option, the models will be optimized and run at 4-bit precision. For configuration for other formats and precisions, refer to [this link](https://github.com/intel-analytics/text-generation-webui?tab=readme-ov-file#32-optimizations-for-other-percisions).
   ```cmd
   python server.py --load-in-4bit
   ```

### Access the WebUI
Upon successful launch, URLs to access the WebUI will be displayed in the terminal as shown below. Open the provided local URL in your browser to interact with the WebUI. 
  <!-- ```cmd
  Running on local URL:  http://127.0.0.1:7860
  ``` -->
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/webui_quickstart_launch_server.png" width=80%; />


## 4. Using the WebUI

### Model Download

Place Huggingface models in `C:\text-generation-webui\models` by either copying locally or downloading via the WebUI. To download, navigate to the **Model** tab, enter the model's huggingface id (for instance, `Qwen/Qwen-7B-Chat`) in the **Download model or LoRA** section, and click **Download**, as illustrated below. 

<img src="https://llm-assets.readthedocs.io/en/latest/_images/webui_quickstart_download_model.png" width=80%; />

After copying or downloading the models, click on the blue **refresh** button to update the **Model** drop-down menu. Then, choose your desired model from the newly updated list.  

<img src="https://llm-assets.readthedocs.io/en/latest/_images/webui_quickstart_select_model.png" width=80%; />


### Load Model

Default settings are recommended for most users. Click **Load** to activate the model. Address any errors by installing missing packages as prompted, and ensure compatibility with your version of the transformers package. Refer to [troubleshooting section](#troubleshooting) for more details.

If everything goes well, you will get a message as shown below.

<img src="https://llm-assets.readthedocs.io/en/latest/_images/webui_quickstart_load_model_success.png" width=80%; />



### Chat with the Model

In the **Chat** tab, start new conversations with **New chat**. 

Enter prompts into the textbox at the bottom and press the **Generate** button to receive responses.

<img src="https://llm-assets.readthedocs.io/en/latest/_images/webui_quickstart_chat.png" width=80%; />

<!-- Notes:
* Multi-turn conversations may consume GPU memory. You may specify the `Truncate the prompt up to this length` value in `Parameters` tab to reduce the GPU memory usage.

* You may switch to a single-turn conversation mode by turning off `Activate text streaming` in the Parameters tab.

* Please see [Chat-Tab Wiki](https://github.com/oobabooga/text-generation-webui/wiki/01-%E2%80%90-Chat-Tab) for more details. -->

### Exit the WebUI

To shut down the WebUI server, use **Ctrl+C** in the **Anaconda Prompt** terminal where the WebUI Server is runing, then close your browser tab.


## Troubleshooting

### Missing Required Dependencies

During model loading, you may encounter an **ImportError** like `ImportError: This modeling file requires the following packages that were not found in your environment`. This indicates certain packages required by the model are absent from your environment. Detailed instructions for installing these necessary packages can be found at the bottom of the error messages. Take the following steps to fix these errors:

- Exit the WebUI Server by pressing **Ctrl+C** in the **Anaconda Prompt** terminal.
- Install the missing pip packages as specified in the error message
- Restart the WebUI Server.

If there are still errors on missing packages, repeat the installation process for any additional required packages.


### Compatiblity issues
If you encounter **AttributeError** errors like shown below, it may be due to some models being incompatible with the current version of the transformers package because they are outdated. In such instances, using a more recent model is recommended.

<img src="https://llm-assets.readthedocs.io/en/latest/_images/webui_quickstart_load_model_error.png" width=80%; />
