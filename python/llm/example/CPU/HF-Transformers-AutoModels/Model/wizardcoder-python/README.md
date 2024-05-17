# WizardCoder-Python
In this directory, you will find examples on how you could apply IPEX-LLM INT4 optimizations on WizardCoder-Python models. For illustration purposes, we utilize the [WizardLM/WizardCoder-Python-7B-V1.0](https://huggingface.co/WizardLM/WizardCoder-Python-7B-V1.0) as a reference WizardCoder-Python model.

## 0. Requirements
To run these examples with IPEX-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a WizardCoder-Python model to predict the next N tokens using `generate()` API, with IPEX-LLM INT4 optimizations.
### 1. Install
We suggest using conda to manage environment:

On Linux:

```bash
conda create -n llm python=3.11 # recommend to use Python 3.11
conda activate llm

# install the latest ipex-llm nightly build with 'all' option
pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu
```

On Windows:

```cmd
conda create -n llm python=3.11
conda activate llm

pip install --pre --upgrade ipex-llm[all]
```

### 2. Run
```
python ./generate.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt PROMPT --n-predict N_PREDICT
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the WizardCoder-Python model to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'WizardLM/WizardCoder-Python-7B-V1.0'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'def print_hello_world():'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `64`.

> **Note**: When loading the model in 4-bit, IPEX-LLM converts linear layers in the model into INT4 format. In theory, a *X*B model saved in 16-bit will requires approximately 2*X* GB of memory for loading, and ~0.5*X* GB memory for further inference.
>
> Please select the appropriate size of the WizardCoder-Python model based on the capabilities of your machine.

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
source ipex-llm-init

# e.g. for a server with 48 cores per socket
export OMP_NUM_THREADS=48
numactl -C 0-47 -m 0 python ./generate.py
```

#### 2.3 Sample Output
#### [WizardLM/WizardCoder-Python-7B-V1.0](https://huggingface.co/WizardLM/WizardCoder-Python-7B-V1.0)
````log
Inference time: xxxx s
-------------------- Prompt --------------------
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
def print_hello_world():

### Response:
-------------------- Output --------------------
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
def print_hello_world():

### Response:Here's the code for the `print_hello_world()` function:

```python
def print_hello_world():
    print("Hello, World!")
```

This function simply prints the string "Hello, World!" to the console. You
````
