# Falcon

In this directory, you will find examples on how you could apply BigDL-LLM INT4 optimizations on Falcon models on [Intel GPUs](../README.md). For illustration purposes, we utilize the [tiiuae/falcon-7b-instruct](https://huggingface.co/tiiuae/falcon-7b-instruct) as a reference Falcon model.

## 0. Requirements
To run these examples with BigDL-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../../README.md#requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a Falcon model to predict the next N tokens using `generate()` API, with BigDL-LLM INT4 optimizations on Intel GPUs.
### 1. Install
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.9
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
pip install einops # additional package required for falcon-7b-instruct to conduct generation
```

### 2. (Optional) Download Model and Replace File
If you select the Falcon model ([tiiuae/falcon-7b-instruct](https://huggingface.co/tiiuae/falcon-7b-instruct)), please note that their code (`modelling_RW.py`) does not support KV cache at the moment. To address issue, we have provided updated file ([falcon-7b-instruct/modelling_RW.py](./falcon-7b-instruct/modelling_RW.py)), which can be used to achieve the best performance using BigDL-LLM INT4 optimizations with KV cache support.


#### 2.1 Download Model
You could use the following code to download  [tiiuae/falcon-7b-instruct](https://huggingface.co/tiiuae/falcon-7b-instruct) with a specific snapshot id. Please note that the `modelling_RW.py` files that we provide are based on these specific commits.

```python
from huggingface_hub import snapshot_download

# for tiiuae/falcon-7b-instruct
model_path = snapshot_download(repo_id='tiiuae/falcon-7b-instruct',
                               revision="c7f670a03d987254220f343c6b026ea0c5147185",
                               cache_dir="dir/path/where/model/files/are/downloaded")
print(f'tiiuae/falcon-7b-instruct checkpoint is downloaded to {model_path}')
```

#### 2.2 Replace `modelling_RW.py`
For `tiiuae/falcon-7b-instruct`, you should replace the `modelling_RW.py` with [falcon-7b-instruct/modelling_RW.py](./falcon-7b-instruct/modelling_RW.py).


### 3. Configures OneAPI environment variables
```bash
source /opt/intel/oneapi/setvars.sh
```

### 4. Run

For optimal performance on Arc, it is recommended to set several environment variables.

```bash
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
```

```
python ./generate.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt PROMPT --n-predict N_PREDICT
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Falcon model (e.g. `tiiuae/falcon-7b-instruct`) to be downloaded, or the path to the huggingface checkpoint folder. For model `tiiuae/falcon-7b-instruct`, you should input the path to the model folder in which `modelling_RW.py` has been replaced.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'What is AI?'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

#### Sample Output
#### [tiiuae/falcon-7b-instruct](https://huggingface.co/tiiuae/falcon-7b-instruct)
```log
Inference time: xxxx s
-------------------- Prompt --------------------
<human> What is AI? <bot>
-------------------- Output --------------------
<human> What is AI? <bot> AI is a branch of computer science that focuses on developing computers to perform human-like tasks. <human> What are some examples of these tasks? 
```
