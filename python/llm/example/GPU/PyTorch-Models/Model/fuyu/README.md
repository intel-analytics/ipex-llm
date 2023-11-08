# Fuyu
In this directory, you will find examples on how you could apply BigDL-LLM INT4 optimizations on Fuyu models on [Intel GPUs](../README.md). For illustration purposes, we utilize the [adept/fuyu-8b](https://huggingface.co/adept/fuyu-8b) as a reference Fuyu model.

## Requirements
To run these examples with BigDL-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a Fuyu model to predict the next N tokens using `generate()` API, with BigDL-LLM INT4 optimizations on Intel GPUs.
### 1. Install
We suggest using conda to manage the Python environment. For more information about conda installation, please refer to [here](https://docs.conda.io/en/latest/miniconda.html#).

After installing conda, create a Python environment for BigDL-LLM:
```bash
conda create -n llm python=3.9 # recommend to use Python 3.9
conda activate llm
# below command will install intel_extension_for_pytorch==2.0.110+xpu as default
# you can install specific ipex/torch version for your need
pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu

pip install transformers==4.35 pillow # additional package required for Fuyu to conduct generation
```

### 2. Configures OneAPI environment variables
```bash
source /opt/intel/oneapi/setvars.sh
```

### 3. Run

For optimal performance on Arc, it is recommended to set several environment variables.

```bash
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
```
```
python ./generate.py --image-path demo.jpg
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Fuyu model (e.g `adept/fuyu-8b`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'adept/fuyu-8b'`.
- `--prompt PROMPT`: argument defining the prompt to be inferred (with the image for chat). It is default to be `'Generate a coco-style caption.'`.
- `--image-path IMAGE_PATH`: argument defining the input image that the chat will focus on. It is required and should be a local path (not url).
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `512`.

#### Sample Chat
#### [adept/fuyu-8b](https://huggingface.co/adept/fuyu-8b)


```log
Inference time: 3.150639772415161 s
-------------------- Prompt --------------------
Generate a coco-style caption.
-------------------- Output --------------------
An orange bus parked on the side of a road.
```
The sample input image is (which is fetched from [COCO dataset](https://cocodataset.org/#explore?id=178242)):

[demo.jpg](https://cocodataset.org/#explore?id=178242)

<a href="http://farm6.staticflickr.com/5331/8954873157_539393fece_z.jpg"><img width=400px src="http://farm6.staticflickr.com/5331/8954873157_539393fece_z.jpg" ></a>