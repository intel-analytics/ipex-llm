
This tutorial provides a step-by-step guide on how to use Text-Generation-WebUI to run Hugging Face transformers-based applications on BigDL-LLM.

The WebUI is ported from [Text-Generation-WebUI](https://github.com/oobabooga/text-generation-webui).

## 1. Prepare the environment on Windows

Please use a python environment management tool (we recommend using Conda) to create a python enviroment and install necessary libs.

### 1.1 Install BigDL-LLM

Please see [BigDL-LLm Installation on Windows](https://bigdl.readthedocs.io/en/latest/doc/LLM/Overview/install_gpu.html#windows) for more details to install BigDL-LLM on your Client.

### 1.2 Install other required dependencies

```bash
pip install -r requirements.txt gradio==3.50.*
```
Note: Text-Generation-WebUI requires transformers version >= 4.36.0


## 2. Start the WebUI Server

### 2.1 For INT4 Optimizations

For a quick start, you may run the script as below to start WebUI directly, it will automatically optimize and accelerate LLMs using INT4 optimizations.
```bash
python server.py
```

### 2.2 Optimizations for Other Percisions

To enable optimizations for more precisions (`sym_int4`, `asym_int4`, `sym_int8`, `fp4`, `fp8`, `fp16`, `mixed_fp4`, `mixed_fp8`, etc.), you may run the command as below:
```bash
python server.py --load-in-low-bit
```

### 2.3 Access the WebUI

After the successful startup of the WebUI server, it will provide links to access the WebUI as below. Please open the public URL in your browser to access the full functionality of the WebUI.

```bash
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://your_tokens_here.gradio.live

This share link expires in 72 hours. For free permanent hosting and GPU upgrades, run `gradio deploy` from Terminal to deploy to Spaces (https://huggingface.co/spaces)
```


## 3. Run Models

### 3.1 Select the Model
First, place your local model in `Text-Generation-WebUI/models` directory, you may also choose to download a model from Hugging Face.

Next, please click the `Model` button to choose your model.

![Image text](https://github.com/intel-analytics/BigDL/blob/1df67d7927ebea0af570b09f36ce76efbf9b8bad/python/llm/example/Text-Generation-WebUI/readme_folder/image.png)


### 3.2 Enable BigDL-LLM Optimizations
Text-Generation-WebUI supports multiple backends, including `Transformers`, `llama.cpp`, `BigDL-LLM`, etc. Please select the BigDL-LLM backend as below to enable low-bit optimizations.

![Image text](https://github.com/intel-analytics/BigDL/blob/1df67d7927ebea0af570b09f36ce76efbf9b8bad/python/llm/example/Text-Generation-WebUI/readme_folder/image-1.png)

Then please select the device according to your device.

![Image text](https://github.com/intel-analytics/BigDL/blob/1df67d7927ebea0af570b09f36ce76efbf9b8bad/python/llm/example/Text-Generation-WebUI/readme_folder/image-2.png)


### 3.3 Load Model in Low Precision 

One common use case of BigDL-LLM is to load a Hugging Face transformers model in low precision.

Notes:

-  When you start the web UI with `--load-in-4bit`, you will not be allowed to choose the quantization precision in `load-in-low-bit`. The model will be loaded with the INT4 precision as default.

-  When you want to load model in other precisions, please run server.py with `--load-in-low-bit` parameter. You may choose the precision from the list of `load-in-low-bit` option, and the `load-in-4bit` option will be disabled.

-  Please select the `optimize-model` and `use_cache` options to accelerate the model.


Now you may click the `Load` button to load the model with BigDL-LLM optimizations.

![Image text](https://github.com/intel-analytics/BigDL/blob/1df67d7927ebea0af570b09f36ce76efbf9b8bad/python/llm/example/Text-Generation-WebUI/readme_folder/image-3.png)


### 3.4 Run the Model on WebUI

Now you can do model inference on Text-Generation-WebUI with BigDL-LLM optimizations, including `Chat`, `Default` and `Notebook` Tabs. Please see [Chat-Tab Wiki](https://github.com/oobabooga/text-generation-webui/wiki/01-%E2%80%90-Chat-Tab) and [Default and Notebook Tabs Wiki](https://github.com/oobabooga/text-generation-webui/wiki/02-%E2%80%90-Default-and-Notebook-Tabs) for more details.

![Image text](https://github.com/intel-analytics/BigDL/blob/1df67d7927ebea0af570b09f36ce76efbf9b8bad/python/llm/example/Text-Generation-WebUI/readme_folder/image-4.png)
