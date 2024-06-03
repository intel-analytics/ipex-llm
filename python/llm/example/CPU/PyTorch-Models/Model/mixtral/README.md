# Mixtral
In this directory, you will find examples on how you could use IPEX-LLM `optimize_model` API to accelerate Mixtral models. For illustration purposes, we utilize the [mistralai/Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) as a reference Mixtral model.

## Requirements
To run these examples with IPEX-LLM on Intel CPUs, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

**Important: Please make sure you have installed `transformers==4.36.0` to run the example.**

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a Mixtral model to predict the next N tokens using `generate()` API, with IPEX-LLM INT4 optimizations on Intel CPUs.
### 1. Install
We suggest using conda to manage the Python environment. For more information about conda installation, please refer to [here](https://conda-forge.org/download/).

After installing conda, create a Python environment for IPEX-LLM:

On Linux:

```bash
conda create -n llm python=3.11 # recommend to use Python 3.11
conda activate llm

pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu


# Please make sure you are using a stable version of Transformers, 4.36.0 or newer.
pip install transformers==4.36.0
```

On Windows:

```cmd
conda create -n llm python=3.11
conda activate llm

pip install --pre --upgrade ipex-llm[all]

pip install transformers==4.36.0
```

### 2. Run

```bash
python ./generate.py --prompt 'What is AI?'
```

In the example, several arguments can be passed to satisfy your requirements:

- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Mixtral model (e.g. `mistralai/Mixtral-8x7B-Instruct-v0.1`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'mistralai/Mixtral-8x7B-Instruct-v0.1'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'What is AI?'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

#### Sample Output
#### [mistralai/Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
```log
Inference time: xxxx s 
-------------------- Output --------------------
[INST] What is AI? [/INST] AI, or Artificial Intelligence, refers to the development of computer systems that can perform tasks that would normally require human intelligence to accomplish. These tasks can include things
```
