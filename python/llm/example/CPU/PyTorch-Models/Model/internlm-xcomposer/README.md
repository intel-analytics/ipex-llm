# InternLM_XComposer
In this directory, you will find examples on how you could use IPEX-LLM `optimize_model` API to accelerate InternLM_XComposer models. For illustration purposes, we utilize the [internlm/internlm-xcomposer-vl-7b](https://huggingface.co/internlm/internlm-xcomposer-vl-7b) as a reference InternLM_XComposer model.

## Requirements
To run these examples with IPEX-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Multi-turn chat centered around an image using `chat()` API
In the example [chat.py](./chat.py), we show a basic use case for an InternLM_XComposer model to start a multi-turn chat centered around an image using `chat()` API, with IPEX-LLM 'optimize_model' API.
### 1. Install
We suggest using conda to manage the Python environment. For more information about conda installation, please refer to [here](https://conda-forge.org/download/).

After installing conda, create a Python environment for IPEX-LLM:

On Linux:

```bash
conda create -n llm python=3.11 # recommend to use Python 3.11
conda activate llm

# install the latest ipex-llm nightly build with 'all' option
pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu

pip install accelerate timm==0.4.12 sentencepiece==0.1.99 gradio==3.44.4 markdown2==2.4.10 xlsxwriter==3.1.2 einops # additional package required for InternLM_XComposer to conduct generation
```

On Windows:

```cmd
conda create -n llm python=3.11
conda activate llm

pip install --pre --upgrade ipex-llm[all]

pip install accelerate timm==0.4.12 sentencepiece==0.1.99 gradio==3.44.4 markdown2==2.4.10 xlsxwriter==3.1.2 einops
```

### 2. Download Model and Replace File
If you select the InternLM_XComposer model ([internlm/internlm-xcomposer-vl-7b](https://huggingface.co/internlm/internlm-xcomposer-vl-7b)), please note that their code (`modeling_InternLM_XComposer.py`) does not support inference on CPU. To address this issue, we have provided the updated file ([internlm-xcomposer-vl-7b/modeling_InternLM_XComposer.py](./internlm-xcomposer-vl-7b/modeling_InternLM_XComposer.py), which can be used to conduct inference on CPU.

#### 2.1 Download Model
You could use the following code to download [internlm/internlm-xcomposer-vl-7b](https://huggingface.co/internlm/internlm-xcomposer-vl-7b) with a specific snapshot id. Please note that the `modeling_InternLM_XComposer.py` file that we provide are based on these specific commits.

```
from huggingface_hub import snapshot_download

# for internlm/internlm-xcomposer-vl-7b
model_path = snapshot_download(repo_id='internlm/internlm-xcomposer-vl-7b',
                               revision="b06eb0c11653fe1568b6c5614b6b7be407ef8660",
                               cache_dir="dir/path/where/model/files/are/downloaded")
print(f'internlm/internlm-xcomposer-vl-7b checkpoint is downloaded to {model_path}')
```

#### 2.2 Replace `modeling_InternLM_XComposer.py`
For `internlm/internlm-xcomposer-vl-7b`, you should replace the `modeling_InternLM_XComposer.py` with [internlm-xcomposer-vl-7b/modeling_InternLM_XComposer.py](./internlm-xcomposer-vl-7b/modeling_InternLM_XComposer.py).


### 3. Run
After setting up the Python environment, you could run the example by following steps.

> **Note**: When loading the model in 4-bit, IPEX-LLM converts linear layers in the model into INT4 format. In theory, a *X*B model saved in 16-bit will requires approximately 2*X* GB of memory for loading, and ~0.5*X* GB memory for further inference.
>
> Please select the appropriate size of the LLaVA model based on the capabilities of your machine.

#### 3.1 Client
On client Windows machines, it is recommended to run directly with full utilization of all cores:
```cmd
python ./chat.py --image-path demo.jpg
```
More information about arguments can be found in [Arguments Info](#33-arguments-info) section. The expected output can be found in [Sample Output](#34-sample-output) section.

#### 3.2 Server
For optimal performance on server, it is recommended to set several environment variables (refer to [here](../README.md#best-known-configuration-on-linux) for more information), and run the example with all the physical cores of a single socket.

E.g. on Linux,
```bash
# set IPEX-LLM env variables
source ipex-llm-init

# e.g. for a server with 48 cores per socket
export OMP_NUM_THREADS=48
numactl -C 0-47 -m 0 python ./chat.py --image-path demo.jpg
```
More information about arguments can be found in [Arguments Info](#33-arguments-info) section. The expected output can be found in [Sample Output](#34-sample-output) section.

#### 3.3 Arguments Info
In the example, several arguments can be passed to satisfy your requirements:

- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the LLaVA model (e.g. `internlm/internlm-xcomposer-vl-7b`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'internlm/internlm-xcomposer-vl-7b'`.
- `--image-path IMAGE_PATH`: argument defining the input image that the chat will focus on. It is required and should be a local path (not url).
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `512`.


#### 3.4 Sample Chat
#### [internlm/internlm-xcomposer-vl-7b](https://huggingface.co/internlm/internlm-xcomposer-vl-7b)

```log
User: 这是什么？
Bot: bus
User: 它可以用来干什么
Bot: transport people
```

The sample input image is (which is fetched from [COCO dataset](https://cocodataset.org/#explore?id=178242)):

[demo.jpg](https://cocodataset.org/#explore?id=178242)

<a href="http://farm6.staticflickr.com/5331/8954873157_539393fece_z.jpg"><img width=400px src="http://farm6.staticflickr.com/5331/8954873157_539393fece_z.jpg" ></a>