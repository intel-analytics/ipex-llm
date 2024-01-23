# Yuan2
In this directory, you will find examples on how you could apply BigDL-LLM INT4 optimizations on Yuan2 models on [Intel GPUs](../README.md). For illustration purposes, we utilize the [IEITYuan/Yuan2-2B-hf](https://huggingface.co/IEITYuan/Yuan2-2B-hf) as a reference Yuan2 model.

## 0. Requirements
To run these examples with BigDL-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

In addition, you need to modify some files in Yuan2-2B-hf folder, since Flash attention dependency is for CUDA usage and currently cannot be installed on Intel CPUs. To manually turn it off, please refer to [this issue](https://github.com/IEIT-Yuan/Yuan-2.0/issues/92). We also provide two modified files([config_mode.json](yuan2-2B-instruct/config_mode.json) and [yuan_hf_mode.py](yuan2-2B-instruct/yuan_hf_mode.py)), which can be used to replace the original content in config.json and yuan_model_hf.py. Here are the changes:

1. Modify 'use_flash_attention' to false in config.json; Comment out lines 35 and 36 in yuan_hf_model.py; 

   ```python
   from flash_attn import flash_attn_varlen_func as flash_attn_unpadded_func
   from flash_attn import flash_attn_func
   ```

2. Change line 271 in yuan_hf_model.py to `inference_hidden_states_memory = torch.empty(bsz, 2, hidden_states.shape[2], dtype=hidden_states.dtype)`.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for an Yuan2 model to predict the next N tokens using `generate()` API, with BigDL-LLM INT4 optimizations on Intel GPUs.
### 1. Install
We suggest using conda to manage the Python environment. For more information about conda installation, please refer to [here](https://docs.conda.io/en/latest/miniconda.html#).

After installing conda, create a Python environment for BigDL-LLM:
```bash
conda create -n llm python=3.9
conda activate llm

pip install --pre --upgrade bigdl-llm[all] # install the latest bigdl-llm nightly build with 'all' option
pip install einops # additional package required for Yuan2 to conduct generation
pip install pandas # additional package required for Yuan2 to conduct generation
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

```bash
python ./generate.py
```

In the example, several arguments can be passed to satisfy your requirements:

- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Yuan2 model to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'IEITYuan/Yuan2-2B-hf'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'请问目前最先进的机器学习算法有哪些？'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `100`.

#### Sample Output
#### [IEITYuan/Yuan2-2B-hf](https://huggingface.co/IEITYuan/Yuan2-2B-hf)
```log
Inference time: xxxx seconds
-------------------- Output --------------------
 
请问目前最先进的机器学习算法有哪些？
答：目前最先进的机器学习算法有：K近邻算法、支持向量机、决策树算法、聚类算法等。<sep> 您好！根据您的需求，我可以为您提供一些最新的机器学习算法：
- K近邻算法（K-Nearest-Neighbors）：K-NN算法是一种用于分类和回归分析的无监督学习算法，它通过遍历样本数据中的所有点，找出与其最相似的
```