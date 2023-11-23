# Mistral
This example shows how to directly run 4-bit AWQ models using BigDL-LLM on Intel CPU. For illustration purposes, we utilize the [TheBloke/Mistral-7B-Instruct-v0.1-AWQ](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-AWQ) and [TheBloke/Mistral-7B-v0.1-AWQ](https://huggingface.co/TheBloke/Mistral-7B-v0.1-AWQ) as reference models.

## 0. Requirements
To run these examples with BigDL-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a Mistral model to predict the next N tokens using `generate()` API, with BigDL-LLM INT4 optimizations.
### 1. Install
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.9
conda activate llm

pip install autoawq==0.1.6 --no-deps
pip install --pre --upgrade bigdl-llm[all] # install bigdl-llm with 'all' option
pip install transformers==4.35.0
pip install accelerate==0.24.1
```

### 2. Run
After setting up the Python environment, you could run the example by following steps.

> **Note**: When loading the model in 4-bit, BigDL-LLM converts linear layers in the model into INT4 format. In theory, a *X*B model saved in 16-bit will requires approximately 2*X* GB of memory for loading, and ~0.5*X* GB memory for further inference.
>
> Please select the appropriate size of the Mistral model based on the capabilities of your machine.

#### 2.1 Client
On client Windows machines, it is recommended to run directly with full utilization of all cores:
```powershell
python ./generate.py --prompt 'What is AI?'
```
More information about arguments can be found in [Arguments Info](#23-arguments-info) section. The expected output can be found in [Sample Output](#24-sample-output) section.

#### 2.2 Server
For optimal performance on server, it is recommended to set several environment variables (refer to [here](../README.md#best-known-configuration-on-linux) for more information), and run the example with all the physical cores of a single socket.

E.g. on Linux,
```bash
# set BigDL-LLM env variables
source bigdl-llm-init

# e.g. for a server with 48 cores per socket
export OMP_NUM_THREADS=48
numactl -C 0-47 -m 0 python ./generate.py --prompt 'What is AI?'
```
More information about arguments can be found in [Arguments Info](#23-arguments-info) section. The expected output can be found in [Sample Output](#24-sample-output) section.

#### 2.3 Arguments Info
In the example, several arguments can be passed to satisfy your requirements:

- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Mistral model (e.g. `TheBloke/Mistral-7B-Instruct-v0.1-AWQ` and `TheBloke/Mistral-7B-v0.1-AWQ`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'TheBloke/Mistral-7B-Instruct-v0.1-AWQ'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'What is AI?'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

#### 2.3 Sample Output
#### [TheBloke/Mistral-7B-Instruct-v0.1-AWQ](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-AWQ)
```log
Inference time: xxxx s
-------------------- Output --------------------
[INST] What is AI? [/INST] AI stands for Artificial Intelligence, which is the simulation of human intelligence in machines that are programmed to think and act like humans. AI technology includes machine learning
```

#### [TheBloke/Mistral-7B-v0.1-AWQ](https://huggingface.co/TheBloke/Mistral-7B-v0.1-AWQ)
```log
Inference time: xxxx s
-------------------- Output --------------------
[INST] What is AI? [/INST]

Artificial Intelligence (AI) is a branch of computer science that deals with the creation of intelligent machines that work and react like humans.

[
```
