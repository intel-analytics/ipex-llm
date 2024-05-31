# Starcoder
In this directory, you will find examples on how you could run Starcoder BF16 inference with self-speculative decoding using IPEX-LLM on [Intel CPUs](../README.md). For illustration purposes,we utilize the [bigcode/starcoder](https://huggingface.co/bigcode/starcoder) and [bigcode/tiny_starcoder_py](https://huggingface.co/bigcode/tiny_starcoder_py) as reference Starcoder models.

## 0. Requirements
To run these examples with IPEX-LLM on Intel CPUs, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [speculative.py](speculative.py), we show a basic use case for a Starcoder model to predict the next N tokens using `generate()` API, with IPEX-LLM speculative decoding optimizations on Intel CPUs.
### 1. Install
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.11
conda activate llm
pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu
pip install intel_extension_for_pytorch==2.1.0
pip install transformers==4.31.0
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

- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Starcoder model (e.g. `bigcode/starcoder` and `bigcode/tiny_starcoder_py`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `bigcode/starcoder`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). A default prompt is provided.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `128`.

#### Sample Output
#### [bigcode/starcoder](https://huggingface.co/bigcode/starcoder)

```log
def dfs_print_Fibonacci_sequence(n):
    if n == 0:
        return
    elif n == 1:
        print(0)
        return
    elif n == 2:
        print(0)
        print(1)
        return
    else:
        print(0)
        print(1)
        dfs_print_Fibonacci_sequence(n-2)
        print(dfs_Fibonacci_sequence(n-1))

def dfs_Fibonacci_sequence(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return dfs_Fibonacci_sequence
Tokens generated 128
E2E Generation time xx.xxxxs
First token latency xx.xxxxs
```

#### [bigcode/tiny_starcoder_py](https://huggingface.co/bigcode/tiny_starcoder_py)
```log
def dfs_print_Fibonacci_sequence(n):
    if n == 0:
        return
    print(n)
    for i in range(2, n):
        print(dfs_print_Fibonacci_sequence(i))


def dfs_print_Fibonacci_sequence_2(n):
    if n == 0:
        return
    print(n)
    for i in range(2, n):
        print(dfs_print_Fibonacci_sequence_2(i))


def dfs_print_Fibonacci_sequence_3(n):
    if n == 0:
        return
    print(n)
    for i in
Tokens generated 128
E2E Generation time xx.xxxxs
First token latency xx.xxxxs
```
