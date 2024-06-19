# GLM-4V

In this directory, you will find examples on how you could apply IPEX-LLM INT4 optimizations on glm-4v models. For illustration purposes, we utilize the [THUDM/glm-4v-9b](https://huggingface.co/THUDM/glm-4v-9b) as a reference glm-4v model.

> **Note**: If you want to download the Hugging Face *Transformers* model, please refer to [here](https://huggingface.co/docs/hub/models-downloading#using-git).
>
> IPEX-LLM optimizes the *Transformers* model in INT4 precision at runtime, and thus no explicit conversion is needed.

## 0. Requirements
To run these examples with IPEX-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a glm-4v model to predict the next N tokens using `generate()` API, with IPEX-LLM INT8 optimizations.
### 1. Install
We suggest using conda to manage environment:

On Linux:

```bash
conda create -n llm python=3.11 # recommend to use Python 3.11
conda activate llm

pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu
pip install torchvision tiktoken
```

On Windows:

```cmd
conda create -n llm python=3.11 # recommend to use Python 3.11
conda activate llm

pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu
pip install torchvision tiktoken
```

### 2. Run
```
python ./generate.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --image-url-or-path IMAGE_URL_OR_PATH --prompt PROMPT --n-predict N_PREDICT
```

Arguments Info:

- `--repo-id-or-model-path`: str, argument defining the huggingface repo id for the glm-4v model to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'glm-4v-9b'`.
- `--image-url-or-path`: str, the URL or path to the image to infer. It is default to be `'http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg'`.
- `--prompt`: str, argument defining the prompt to be inferred (with integrated prompt format for chat). It is default to be `What is in the image?`.
- `--n-predict`: int, argument defining the max number of tokens to predict. It is default to be `32`.

> **Note**: When loading the model in 8-bit, IPEX-LLM converts linear layers in the model into INT8 format. In theory, a *X*B model saved in 16-bit will requires approximately 2*X* GB of memory for loading, and ~0.5*X* GB memory for further inference.
>
> Please select the appropriate size of the phi-3-vision model based on the capabilities of your machine.

#### 2.1 Client
On client Windows machines, it is recommended to run directly with full utilization of all cores:
```cmd
python ./generate.py
```

#### 2.2 Server
For optimal performance on server, it is recommended to set several environment variables (refer to [here](../README.md#best-known-configuration-on-linux) for more information), and run the example with all the physical cores of a single socket.

E.g. on Linux,
```bash
# set IPEX-LLM env variables
source ipex-llm-init

# e.g. for a server with 48 cores per socket
export OMP_NUM_THREADS=48
numactl -C 0-47 -m 0 python ./generate.py
```

#### 2.3 Sample Output
#### [THUDM/glm-4v-9b](https://huggingface.co/THUDM/glm-4v-9b)
```log
Inference time: xxxx s
-------------------- Prompt --------------------
What is in the image?
-------------------- Output --------------------
The image shows a young child holding up a small white teddy bear dressed in a pink
```
The sample input image is shown below. 

<a href="http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg"><img width=400px src="http://farm6.staticflickr.com/5268/5602445367_3504763978_z.jpg" ></a>