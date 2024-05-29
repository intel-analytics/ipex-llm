# phi-3-vision

In this directory, you will find examples on how you could apply IPEX-LLM INT8 optimizations on phi-3-vision models. For illustration purposes, we utilize the [microsoft/Phi-3-vision-128k-instruct](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct) as a reference phi-3-vision model.

> **Note**: If you want to download the Hugging Face *Transformers* model, please refer to [here](https://huggingface.co/docs/hub/models-downloading#using-git).
>
> IPEX-LLM optimizes the *Transformers* model in INT8 precision at runtime, and thus no explicit conversion is needed.

## Requirements
To run these examples with IPEX-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a phi-3-vision model to predict the next N tokens using `generate()` API, with IPEX-LLM INT8 optimizations.
### 1. Install
We suggest using conda to manage the Python environment. For more information about conda installation, please refer to [here](https://docs.conda.io/en/latest/miniconda.html#).

After installing conda, create a Python environment for IPEX-LLM:

On Linux:

```bash
conda create -n llm python=3.11 # recommend to use Python 3.11
conda activate llm

pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu

pip install pillow torchvision
pip install transformers==4.37.0
```

On Windows:

```cmd
conda create -n llm python=3.11
conda activate llm

pip install --pre --upgrade ipex-llm[all]

pip install pillow torchvision
pip install transformers==4.37.0
```

### 2. Run
After setting up the Python environment, you could run the example by following steps.

> **Note**: When loading the model in 8-bit, IPEX-LLM converts linear layers in the model into INT8 format. In theory, a *X*B model saved in 16-bit will requires approximately 2*X* GB of memory for loading, and ~0.5*X* GB memory for further inference.
>
> Please select the appropriate size of the phi-3-vision model based on the capabilities of your machine.

#### 2.1 Client
On client Windows machines, it is recommended to run directly with full utilization of all cores:
```cmd
python ./generate.py --prompt 'What is in the image?'
```
More information about arguments can be found in [Arguments Info](#23-arguments-info) section. The expected output can be found in [Sample Output](#24-sample-output) section.

#### 2.2 Server
For optimal performance on server, it is recommended to set several environment variables (refer to [here](../README.md#best-known-configuration-on-linux) for more information), and run the example with all the physical cores of a single socket.

E.g. on Linux,
```bash
# set IPEX-LLM env variables
source ipex-llm-init

# e.g. for a server with 48 cores per socket
export OMP_NUM_THREADS=48
numactl -C 0-47 -m 0 python ./generate.py --prompt 'What is in the image?'
```
More information about arguments can be found in [Arguments Info](#23-arguments-info) section. The expected output can be found in [Sample Output](#24-sample-output) section.

#### 2.3 Arguments Info
In the example, several arguments can be passed to satisfy your requirements:

- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the phi-3-vision model (e.g. `microsoft/Phi-3-vision-128k-instruct`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'microsoft/Phi-3-vision-128k-instruct'`.
- `--image-url-or-path PROMPT`: argument defining the image to be infered. It is default to be `'http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'What is in the image?'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

#### 2.4 Sample Output
#### [microsoft/Phi-3-vision-128k-instruct](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct)

```log
Inference time: xxxx s
-------------------- Prompt --------------------
[{'role': 'user', 'content': '<|image_1|>\nWhat is in the image?'}]
-------------------- Output --------------------
Image link/path: http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg


What is in the image?
 The image shows a child holding a white teddy bear dressed in a pink dress.
```

The sample input image is (which is fetched from [COCO dataset](https://cocodataset.org/#explore?id=264959)):

<a href="http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg"><img width=400px src="http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg" ></a>
