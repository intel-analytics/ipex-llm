# BERT
In this directory, you will find examples on how you could use IPEX-LLM `optimize_model` API to accelerate BERT models. For illustration purposes, we utilize the [bert-large-uncased](https://huggingface.co/bert-large-uncased) as reference BERT models.

## Requirements
To run these examples with IPEX-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Extract the feature of given text
In the example [extract_feature.py](./extract_feature.py), we show a basic use case for a BERT model to extract the feature of given text, with IPEX-LLM INT4 optimizations.
### 1. Install
We suggest using conda to manage the Python environment. For more information about conda installation, please refer to [here](https://conda-forge.org/download/).

After installing conda, create a Python environment for IPEX-LLM:

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
After setting up the Python environment, you could run the example by following steps.

#### 2.1 Client
On client Windows machines, it is recommended to run directly with full utilization of all cores:
```cmd
python ./extract_feature.py --text 'This is an example text for feature extraction.'
```
More information about arguments can be found in [Arguments Info](#23-arguments-info) section.

#### 2.2 Server
For optimal performance on server, it is recommended to set several environment variables (refer to [here](../README.md#best-known-configuration-on-linux) for more information), and run the example with all the physical cores of a single socket.

E.g. on Linux,
```bash
# set IPEX-LLM env variables
source ipex-llm-init

# e.g. for a server with 48 cores per socket
export OMP_NUM_THREADS=48
numactl -C 0-47 -m 0 python ./extract_feature.py --text 'This is an example text for feature extraction.'
```
More information about arguments can be found in [Arguments Info](#23-arguments-info) section.

#### 2.3 Arguments Info
In the example, several arguments can be passed to satisfy your requirements:

- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the BERT model (e.g. `bert-large-uncased`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'bert-large-uncased'`.
- `--text TEXT`: argument defining the text to be extracted features. It is default to be `'This is an example text for feature extraction.'`.
