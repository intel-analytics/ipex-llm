# Yuan2
In this directory, you will find examples on how you could apply IPEX-LLM `optimize_model` API to accelerate Yuan2 models. For illustration purposes, we utilize the [IEITYuan/Yuan2-2B-hf](https://huggingface.co/IEITYuan/Yuan2-2B-hf) as a reference Yuan2 model.

## 0. Requirements
To run these examples with IPEX-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

In addition, you need to modify some files in Yuan2-2B-hf folder, since Flash attention dependency is for CUDA usage and currently cannot be installed on Intel CPUs. To manually turn it off, please refer to [this issue](https://github.com/IEIT-Yuan/Yuan-2.0/issues/92). We also provide two modified files([config.json](yuan2-2B-instruct/config.json) and [yuan_hf_model.py](yuan2-2B-instruct/yuan_hf_model.py)), which can be used to replace the original content in config.json and yuan_hf_model.py.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for an Yuan2 model to predict the next N tokens using `generate()` API, with IPEX-LLM INT4 optimizations.
### 1. Install
We suggest using conda to manage the Python environment. For more information about conda installation, please refer to [here](https://conda-forge.org/download/).

After installing conda, create a Python environment for IPEX-LLM:

On Linux:

```bash
conda create -n llm python=3.11 # recommend to use Python 3.11
conda activate llm

# install the latest ipex-llm nightly build with 'all' option
pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu
pip install einops # additional package required for Yuan2 to conduct generation
pip install pandas # additional package required for Yuan2 to conduct generation
```

On Windows:

```cmd
conda create -n llm python=3.11
conda activate llm

pip install --pre --upgrade ipex-llm[all]
pip install einops
pip install pandas
```

### 2. Run
```
python ./generate.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt PROMPT --n-predict N_PREDICT
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Yuan2 model to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'IEITYuan/Yuan2-2B-hf'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'IEITYuan/Yuan2-2B-hf'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `100`.

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
#### [IEITYuan/Yuan2-2B-hf](https://huggingface.co/IEITYuan/Yuan2-2B-hf)
```log
Inference time: xxxx seconds
-------------------- Output --------------------
 
What is AI?
The term "AI" refers to a process that involves creating machines or devices that can perform tasks that typically require human intelligence, such as AI-based decision-making and machine learning. AI is rapidly advancing in the fields of machine learning, computer science, and artificial intelligence, and has been used in various fields to achieve various goals, such as improving accuracy, efficiency, and complexity. However, the
```