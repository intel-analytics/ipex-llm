# Ziya
In this directory, you will find examples on how you could run Ziya BF16 inference with self-speculative decoding using BigDL-LLM on [Intel CPUs](../README.md). For illustration purposes,we utilize the [IDEA-CCNL/Ziya-Coding-34B-v1.0](https://huggingface.co/IDEA-CCNL/Ziya-Coding-34B-v1.0) as reference Ziya model.

## 0. Requirements
To run the example with BigDL-LLM on Intel CPUs, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [speculative.py](./speculative.py), we show a basic use case for a Ziya model to predict the next N tokens using `generate()` API, with BigDL-LLM speculative decoding optimizations on Intel CPUs.
### 1. Install
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.9
conda activate llm
pip install --pre --upgrade bigdl-llm[all]
pip install intel_extension_for_pytorch==2.1.0
pip install transformers==4.35.2
```
### 2. Configures high-performing processor environment variables
```bash
source bigdl-llm-init -t
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
Please Complete the given function below according to the docstring: 
写一段快速排序
<bot>: 
def quick_sort(arr):
    """
    This function takes in a list of integers and returns a new list that is sorted in ascending order.
    """
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[0]
        less_than_pivot = [x for x in arr[1:] if x <= pivot]
        greater_than_pivot = [x for x in arr[1:] if x > pivot]
        return quick_sort(less_than_pivot) + [pivot]
Tokens generated 128
E2E Generation time xx.xxxxs
First token latency xx.xxxxs
```


### 4. Accelerate with BIGDL_OPT_IPEX

BIGDL_OPT_IPEX can help to accelerate speculative decoding on Ziya, and please refer to [here](https://github.com/intel-analytics/BigDL/blob/main/python/llm/example/CPU/Speculative-Decoding/baichuan2/README.md#4-accelerate-with-bigdl_opt_ipex) for a try.
