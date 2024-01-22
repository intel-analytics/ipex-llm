
This tutorial will show you how to use text generation webui to conduct a multi-turn chat task with bigdl-llm backend.

## 1. Prepare the environment

This webui has already support running inference on both CPU and GPU device.

1. Install BigDL-LLM

    ```bash
    # on cpu device
    pip install --pre --upgrade bigdl-llm[all]

    # on gpu device
    pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
    ```

2. Install other required dependencies

    ```bash
    pip install -r requirements.txt
    ```

## 2. Start the webui

For a quick start, you can run the script as below to start the webui directly.
```
python server.py --load-in-4bit
```

To enable more relevant low bit optimizations, run the command as below:
```
python server.py
```

After the successful startup of the WebUI server, it will provide the link to access the WebUI as below. Open this link in your browser to access the full functionality of the WebUI.


## 3. Multi-turn chat

### 3.1. Select the Model
First, place your local model in `Text-Generation-WebUI/models` directory, or you can choose to download a model from Hugging Face.

Next, please choose the model you want to use.

![Image text](https://github.com/intel-analytics/BigDL/blob/1df67d7927ebea0af570b09f36ce76efbf9b8bad/python/llm/example/Text-Generation-WebUI/readme_folder/image.png)


### 3.2. Select the Backend
Text-Generation-WebUI supports multiple backends, including `Transformers`, `llama.cpp`, `BigDL-LLM`, etc. Please select the BigDL-LLM backend as below to enable low-bit optimization for the models.

![Image text](https://github.com/intel-analytics/BigDL/blob/1df67d7927ebea0af570b09f36ce76efbf9b8bad/python/llm/example/Text-Generation-WebUI/readme_folder/image-1.png)

Then you could select the device.

![Image text](https://github.com/intel-analytics/BigDL/blob/1df67d7927ebea0af570b09f36ce76efbf9b8bad/python/llm/example/Text-Generation-WebUI/readme_folder/image-2.png)

Notes:

1. When you start the web UI with `--load-in-4bit`, you will not be allowed to choose the quantization precision in `load-in-low-bit`. The model will be imported with the default precision of int4.

2. When you want to use other precisions, including `fp16`, `nf4` and `int8`, use `--load-in-low-bit` to start server.py. You can choose the precision you need from the list of `load-in-low-bit` options, and the `load-in-4bit` option will be disabled.

3. Please select the `optimize` and `use_cache` options to accelerate the model.


### 3.3. Load Model
Then you may click the `Load` button to load the model with BigDL-LLM optimizations.

![Image text](https://github.com/intel-analytics/BigDL/blob/1df67d7927ebea0af570b09f36ce76efbf9b8bad/python/llm/example/Text-Generation-WebUI/readme_folder/image-3.png)

### 3.4. Start Multi-turn Chat
Now you can use your model for multi-turn generates.

![Image text](https://github.com/intel-analytics/BigDL/blob/1df67d7927ebea0af570b09f36ce76efbf9b8bad/python/llm/example/Text-Generation-WebUI/readme_folder/image-4.png)
