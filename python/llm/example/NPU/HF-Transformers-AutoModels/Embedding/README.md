# Run Embedding Model on Intel NPU
In this directory, you will find examples on how you could apply IPEX-LLM low-bit optimizations on embedding models on [Intel NPUs](../../../README.md). See the table blow for verified models.

## Verified Models

| Model      | Model Link                                                    |
|------------|----------------------------------------------------------------|
| Bce-Embedding-Base-V1 | [maidalun1020/bce-embedding-base_v1](https://huggingface.co/maidalun1020/bce-embedding-base_v1) |

Please refer to [Quickstart](../../../../../../docs/mddocs/Quickstart/npu_quickstart.md#python-api) for details about verified platforms.

## 0. Prerequisites
For `ipex-llm` NPU support, please refer to [Quickstart](../../../../../../docs/mddocs/Quickstart/npu_quickstart.md#install-prerequisites) for details about the required preparations.

## 1. Install
### 1.1 Installation on Windows
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.11
conda activate llm

# install ipex-llm with 'npu' option
pip install --pre --upgrade ipex-llm[npu]

# [optional] for Bce-Embedding-Base-V1
pip install BCEmbedding==0.1.5 transformers==4.40.0
```
Please refer to [Quickstart](../../../../../../docs/mddocs/Quickstart/npu_quickstart.md#install-ipex-llm-with-npu-support) for more details about `ipex-llm` installation on Intel NPU.

### 1.2 Runtime Configurations
Please refer to [Quickstart](../../../../../../docs/mddocs/Quickstart/npu_quickstart.md#runtime-configurations) for environment variables setting based on your device.

## 2. Run Optimized Models
The examples below show how to run the **_optimized HuggingFace model implementations_** on Intel NPU, including
- [Bce-Embedding-Base-V1 ](./bce-embedding.py)

### 2.1 Run Bce-Embedding-Base-V1
```bash
# to run Bce-Embedding-Base-V1
python bce-embedding.py --save-directory <converted_model_path>
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the model (i.e. `maidalun1020/bce-embedding-base_v1`) to be downloaded, or the path to the huggingface checkpoint folder.
- `--prompt PROMPT`: argument defining the sentences to encode.
- `--max-context-len MAX_CONTEXT_LEN`: argument defining the maximum sequence length for both input and output tokens. It is default to be `1024`.
- `--max-prompt-len MAX_PROMPT_LEN`: argument defining the maximum number of tokens that the input prompt can contain. It is default to be `512`.
- `--save-directory SAVE_DIRECTORY`: argument defining the path to save converted model. If it is a non-existing path, the original pretrained model specified by `REPO_ID_OR_MODEL_PATH` will be loaded, otherwise the lowbit model in `SAVE_DIRECTORY` will be loaded.

#### Sample Output
##### [maidalun1020/bce-embedding-base_v1](https://huggingface.co/maidalun1020/bce-embedding-base_v1)

```log
Inference time: xxxx s
[[-0.00674987 -0.01700369 -0.0028928  ... -0.05296675 -0.00352772
   0.00827096]
 [-0.04398304  0.00023038  0.00643183 ... -0.02717186  0.00483789
   0.02298774]]
```
