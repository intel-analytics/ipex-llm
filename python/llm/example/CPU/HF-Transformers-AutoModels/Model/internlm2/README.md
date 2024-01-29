# InternLM2

In this directory, you will find examples on how you could apply BigDL-LLM INT4 optimizations on InternLM2 models. For illustration purposes, we utilize the [internlm/internlm2-chat-7b](https://huggingface.co/internlm/internlm2-chat-7b) as a reference InternLM2 model.

## 0. Requirements
To run these examples with BigDL-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a InternLM2 model to predict the next N tokens using `generate()` API, with BigDL-LLM INT4 optimizations.
### 1. Install
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.9
conda activate llm

pip install bigdl-llm[all] # install bigdl-llm with 'all' option
```

### 2. Run
```
python ./generate.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt PROMPT --n-predict N_PREDICT
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the InternLM2 model to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'internlm/internlm2-chat-7b'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'AI是什么？'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

> **Note**: When loading the model in 4-bit, BigDL-LLM converts linear layers in the model into INT4 format. In theory, a *X*B model saved in 16-bit will requires approximately 2*X* GB of memory for loading, and ~0.5*X* GB memory for further inference.
>
> Please select the appropriate size of the InternLM2 model based on the capabilities of your machine.

#### 2.1 Client
On client Windows machine, it is recommended to run directly with full utilization of all cores:
```powershell
python ./generate.py 
```

#### 2.2 Server
For optimal performance on server, it is recommended to set several environment variables (refer to [here](../README.md#best-known-configuration-on-linux) for more information), and run the example with all the physical cores of a single socket.

E.g. on Linux,
```bash
# set BigDL-LLM env variables
source bigdl-llm-init

# e.g. for a server with 48 cores per socket
export OMP_NUM_THREADS=48
numactl -C 0-47 -m 0 python ./generate.py
```

#### 2.3 Sample Output
#### [internlm/internlm2-chat-7b](https://huggingface.co/internlm/internlm2-chat-7b)
```log
Inference time: xxxx s
-------------------- Prompt --------------------
<|User|>:解释一种机器学习算法
<|Bot|>:
-------------------- Output --------------------
<|User|>:解释一种机器学习算法
<|Bot|>:一种常见的机器学习算法是决策树。决策树是一种基于树形结构的分类算法，它通过一系列的判断条件，将数据集分成不同的类别。
决策树的构建过程如下：
1.选择一个最佳特征，将数据集分成不同的子集。
2.对于每个子集，递归地构建决策树。
3.对于叶子节点，将其标记为相应的类别。
4.对于非叶子节点，将其标记为相应的特征。
决策树的优点包括：
1.易于理解和解释。
2.可以处理多分类问题。
3.可以处理缺失值。
```