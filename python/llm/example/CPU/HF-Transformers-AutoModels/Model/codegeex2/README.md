# CodeGeeX2

In this directory, you will find examples on how you could apply IPEX-LLM INT4 optimizations on CodeGeex2 models which is implemented based on the ChatGLM2 architecture trained on more code data. We utilize the [THUDM/codegeex2-6b](https://huggingface.co/THUDM/codegeex2-6b) as a reference CodeGeeX2 model.

## 0. Requirements
To run these examples with IPEX-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example 1: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a CodeGeeX2 model to predict the next N tokens using `generate()` API, with IPEX-LLM INT4 optimizations.
### 1. Install
We suggest using conda to manage environment:

On Linux:

```bash
conda create -n llm python=3.11 # recommend to use Python 3.11
conda activate llm

# install the latest ipex-llm nightly build with 'all' option
pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.31.0
```

On Windows:

```cmd
conda create -n llm python=3.11
conda activate llm

pip install --pre --upgrade ipex-llm[all]
pip install transformers==4.31.0
```

### 2. Run
```
python ./generate.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt PROMPT --n-predict N_PREDICT
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the CodeGeex2 model to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'THUDM/codegeex2-6b'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'# language: Python\n# write a bubble sort function\n'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `128`.

#### 2.1 Client
On client Windows machine, it is recommended to run directly with full utilization of all cores:
```cmd
python ./generate.py 
```

#### 2.2 Server
For optimal performance on server, it is recommended to set several environment variables (refer to [here](../README.md#best-known-configuration-on-linux) for more information), and run the example with all the physical cores of a single socket.

E.g. on Linux,
```bash
# set IPEX-LLM env variables
source ipex-llm-init -t

# e.g. for a server with 48 cores per socket
export OMP_NUM_THREADS=48
numactl -C 0-47 -m 0 python ./generate.py
```

#### 2.3 Sample Output
#### [THUDM/codegeex2-6b](https://huggingface.co/THUDM/codegeex2-6b)
```log
Inference time: xxxx s
-------------------- Prompt --------------------
# language: Python
# write a bubble sort function

-------------------- Output --------------------
# language: Python
# write a bubble sort function


def bubble_sort(lst):
    for i in range(len(lst) - 1):
        for j in range(len(lst) - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
    return lst


print(bubble_sort([1, 2, 3, 4, 5, 6, 7, 8,
```