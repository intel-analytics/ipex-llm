# Falcon

In this directory, you will find examples on how you could apply IPEX-LLM INT4 optimizations on Falcon models. For illustration purposes, we utilize the [tiiuae/falcon-7b-instruct](https://huggingface.co/tiiuae/falcon-7b-instruct) and [tiiuae/falcon-40b-instruct](https://huggingface.co/tiiuae/falcon-40b-instruct) as reference Falcon models.

## 0. Requirements
To run these examples with IPEX-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a Falcon model to predict the next N tokens using `generate()` API, with IPEX-LLM INT4 optimizations.
### 1. Install
We suggest using conda to manage environment:

On Linux:

```bash
conda create -n llm python=3.11
conda activate llm

# install the latest ipex-llm nightly build with 'all' option
pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu
pip install einops # additional package required for falcon-7b-instruct and falcon-40b-instruct to conduct generation
```

On Windows:

```cmd
conda create -n llm python=3.11
conda activate llm

pip install --pre --upgrade ipex-llm[all]
pip install einops
```

### 2. (Optional) Download Model and Replace File
If you select the Falcon models ([tiiuae/falcon-7b-instruct](https://huggingface.co/tiiuae/falcon-7b-instruct) or [tiiuae/falcon-40b-instruct](https://huggingface.co/tiiuae/falcon-40b-instruct)), please note that their code (`modelling_RW.py`) does not support KV cache at the moment. To address issue, we have provided two updated files ([falcon-7b-instruct/modelling_RW.py](./falcon-7b-instruct/modelling_RW.py) and [falcon-40b-instruct/modelling_RW.py](./falcon-40b-instruct/modelling_RW.py)), which can be used to achieve the best performance using IPEX-LLM INT4 optimizations with KV cache support.
After transformers 4.36, only transformer models are supported since remote code diverges from transformer model code, make sure set `trust_remote_code=False`.
```python
 model = AutoModelForCausalLM.from_pretrained(model_path,
                                              load_in_4bit=True,
                                              trust_remote_code=False)
```

#### 2.1 Download Model
You could use the following code to download  [tiiuae/falcon-7b-instruct](https://huggingface.co/tiiuae/falcon-7b-instruct) or [tiiuae/falcon-40b-instruct](https://huggingface.co/tiiuae/falcon-40b-instruct) with a specific snapshot id. Please note that the `modelling_RW.py` files that we provide are based on these specific commits.

```python
from huggingface_hub import snapshot_download

# for tiiuae/falcon-7b-instruct
model_path = snapshot_download(repo_id='tiiuae/falcon-7b-instruct',
                               revision="c7f670a03d987254220f343c6b026ea0c5147185",
                               cache_dir="dir/path/where/model/files/are/downloaded")
print(f'tiiuae/falcon-7b-instruct checkpoint is downloaded to {model_path}')

# for tiiuae/falcon-40b-instruct
model_path = snapshot_download(repo_id='tiiuae/falcon-40b-instruct',
                               revision="1e7fdcc9f45d13704f3826e99937917e007cd975",
                               cache_dir="dir/path/where/model/files/are/downloaded")
print(f'tiiuae/falcon-40b-instruct checkpoint is downloaded to {model_path}')
```

#### 2.2 Replace `modelling_RW.py`
For `tiiuae/falcon-7b-instruct`, you should replace the `modelling_RW.py` with [falcon-7b-instruct/modelling_RW.py](./falcon-7b-instruct/modelling_RW.py).

For `tiiuae/falcon-40b-instruct`, you should replace the `modelling_RW.py` with [falcon-40b-instruct/modelling_RW.py](./falcon-40b-instruct/modelling_RW.py).

### 3. Run
```
python ./generate.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt PROMPT --n-predict N_PREDICT
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Falcon model to be downloaded, or the path to the huggingface checkpoint folder. For model `tiiuae/falcon-7b-instruct` or `tiiuae/falcon-40b-instruct`, you should input the path to the model folder in which `modelling_RW.py` has been replaced.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'What is AI?'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

> **Note**: When loading the model in 4-bit, IPEX-LLM converts linear layers in the model into INT4 format. In theory, a *X*B model saved in 16-bit will requires approximately 2*X* GB of memory for loading, and ~0.5*X* GB memory for further inference.
>
> Please select the appropriate size of the Falcon model based on the capabilities of your machine.

#### 3.1 Client
On client Windows machine, it is recommended to run directly with full utilization of all cores:
```cmd
python ./generate.py 
```

#### 3.2 Server
For optimal performance on server, it is recommended to set several environment variables (refer to [here](../README.md#best-known-configuration-on-linux) for more information), and run the example with all the physical cores of a single socket.

E.g. on Linux,
```bash
# set IPEX-LLM env variables
source ipex-llm-init

# e.g. for a server with 48 cores per socket
export OMP_NUM_THREADS=48
numactl -C 0-47 -m 0 python ./generate.py
```

#### 3.3 Sample Output
#### [tiiuae/falcon-7b-instruct](https://huggingface.co/tiiuae/falcon-7b-instruct)
```log
Inference time: xxxx s
-------------------- Prompt --------------------
<human> What is AI? <bot>
-------------------- Output --------------------
<human> What is AI? <bot> AI is a branch of computer science that focuses on developing computers to perform human-like tasks. <human> What are some examples of these tasks? 
```

#### [tiiuae/falcon-40b-instruct](https://huggingface.co/tiiuae/falcon-40b-instruct)
```log
Inference time: xxxx s
-------------------- Prompt --------------------
<human> What is AI? <bot>
-------------------- Output --------------------
<human> What is AI? <bot> AI stands for Artificial Intelligence. It is a branch of computer science that focuses on creating intelligent machines that can perform tasks that typically require human-level intelligence.
```