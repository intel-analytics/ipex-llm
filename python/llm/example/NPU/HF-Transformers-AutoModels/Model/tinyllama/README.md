# Run LLama2 on Intel NPU
In this directory, you will find examples on how you could apply IPEX-LLM INT4 optimizations on Llama2 models on [Intel NPUs](../../../README.md). For illustration purposes, we utilize the [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) as reference Llama2 models.

## 0. Requirements
To run these examples with IPEX-LLM on Intel NPUs, make sure to install the newest driver version of Intel NPU.
Go to https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html to download and unzip the driver.
Then go to **Device Manager**, find **Neural Processors** -> **Intel(R) AI Boost**.
Right click and select **Update Driver**. And then manually select the folder unzipped from the driver.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a Llama2 model to predict the next N tokens using `generate()` API, with IPEX-LLM INT4 optimizations on Intel NPUs.
### 1. Install
#### 1.1 Installation on Windows
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.10 libuv
conda activate llm
pip install --pre --upgrade ipex-llm
pip install openvino
pip install onnx
pip install torch
pip install accelerate
pip install transformers==4.35.1
```

### 2. Runtime Configurations
For optimal performance, it is recommended to set several environment variables. Please check out the suggestions based on your device.
#### 2.1 Configurations for Windows
<details>

```cmd
set BIGDL_USE_NPU=1
```

</details>

### 3. Running examples

```
python ./generate.py
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the TinyLlama model (e.g. `TinyLlama/TinyLlama-1.1B-Chat-v1.0`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'TinyLlama/TinyLlama-1.1B-Chat-v1.0'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'Once upon a time, there is a little girl named Lily who lives in a small village.'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

#### Sample Output
#### [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)

```log
Inference time: xxxx s
-------------------- Output --------------------
<s> Once upon a time, there is a little girl named Lily who lives in a small village. She loves to play with her friends and spend time with her family.<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk>
--------------------------------------------------------------------------------
done
```
