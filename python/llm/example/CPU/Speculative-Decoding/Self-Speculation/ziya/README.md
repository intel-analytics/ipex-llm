# Ziya
In this directory, you will find examples on how you could run Ziya BF16 inference with self-speculative decoding using IPEX-LLM on [Intel CPUs](../README.md). For illustration purposes,we utilize the [IDEA-CCNL/Ziya-Coding-34B-v1.0](https://huggingface.co/IDEA-CCNL/Ziya-Coding-34B-v1.0) as reference Ziya model.

## 0. Requirements
To run the example with IPEX-LLM on Intel CPUs, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [speculative.py](speculative.py), we show a basic use case for a Ziya model to predict the next N tokens using `generate()` API, with IPEX-LLM speculative decoding optimizations on Intel CPUs.
### 1. Install
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.11
conda activate llm
pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu
pip install intel_extension_for_pytorch==2.1.0
pip install transformers==4.35.2
```
### 2. Configures high-performing processor environment variables
```bash
source ipex-llm-init -t
export OMP_NUM_THREADS=48 # you can change 48 here to #cores of one processor socket
```
### 3. Run

We recommend to use `numactl` to bind the program to a specified processor socket:

```bash
numactl -C 0-47 -m 0 python ./speculative.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt PROMPT --n-predict N_PREDICT
```

For example, 0-47 means bind the python program to core list 0-47 for a 48-core socket.

Arguments info:

- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Ziya model (e.g. `IDEA-CCNL/Ziya-Coding-34B-v1.0`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `IDEA-CCNL/Ziya-Coding-34B-v1.0`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). A default prompt is provided.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `128`.

#### Sample Output
#### [IDEA-CCNL/Ziya-Coding-34B-v1.0](https://huggingface.co/IDEA-CCNL/Ziya-Coding-34B-v1.0)

```log
<human>: 
写一段快速排序
<bot>: 
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
Tokens generated 100
E2E Generation time xx.xxxxs
First token latency xx.xxxxs
```
