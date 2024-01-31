# InternLM2
In this directory, you will find examples on how you could apply BigDL-LLM INT4 optimizations on InternLM2 models on [Intel GPUs](../../../README.md). For illustration purposes, we utilize the [internlm/internlm2-chat-7b](https://huggingface.co/internlm/internlm2-chat-7b) as a reference InternLM2 model.

## 0. Requirements
To run these examples with BigDL-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../../README.md#requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a InternLM2 model to predict the next N tokens using `generate()` API, with BigDL-LLM INT4 optimizations on Intel GPUs.
### 1. Install
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.9
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
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
python ./generate.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt PROMPT --n-predict N_PREDICT
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the InternLM2 model (e.g. `internlm/internlm2-chat-7b`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'internlm/internlm2-chat-7b'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'AI是什么？'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

#### Sample Output
#### [internlm/internlm2-chat-7b](https://huggingface.co/internlm/internlm2-chat-7b)
```log
Inference time: 2.16 s
-------------------- Prompt --------------------
<|User|>:解释一种机器学习算法
<|Bot|>:
-------------------- Output --------------------
<|User|>:解释一种机器学习算法
<|Bot|>:好的，我可以解释一种常见的机器学习算法——决策树。

决策树是一种监督学习算法，用于分类和回归问题。它的主要思想是将数据集分成更小的子集，直到每个子集只包含一个类别的数据。

决策树的构建过程可以分为以下几个步骤：

1. **数据准备**：
   - **数据清洗**：处理缺失值、异常值等。
   - **特征选择**：选择对目标变量有较大影响的特征。
2. **特征处理**：
   - **离散化**：将连续的特征离散化，以便于决策
```