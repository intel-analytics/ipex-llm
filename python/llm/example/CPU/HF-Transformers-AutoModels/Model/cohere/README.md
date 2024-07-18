# CoHere/command-r
In this directory, you will find examples on how you could apply IPEX-LLM INT4 optimizations on cohere models. For illustration purposes, we utilize the [CohereForAI/c4ai-command-r-v01](https://huggingface.co/CohereForAI/c4ai-command-r-v01) as reference model.

## 0. Requirements
To run these examples with IPEX-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a cohere model to predict the next N tokens using `generate()` API, with IPEX-LLM INT4 optimizations.
### 1. Install
We suggest using conda to manage environment:

On Linux:

```bash
conda create -n llm python=3.11
conda activate llm

# install ipex-llm with 'all' option
pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu
pip install "transformers>=4.40.0"
```

On Windows:

```cmd
conda create -n llm python=3.11
conda activate llm

pip install --pre --upgrade ipex-llm[all]
pip install "transformers>=4.40.0"
```

### 2. Run
```
python ./generate.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt PROMPT --n-predict N_PREDICT
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the cohere model to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'CohereForAI/c4ai-command-r-v01'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'What is AI?'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

> **Note**: When loading the model in 4-bit, IPEX-LLM converts linear layers in the model into INT4 format. In theory, a *X*B model saved in 16-bit will requires approximately 2*X* GB of memory for loading, and ~0.5*X* GB memory for further inference.
>
> Please select the appropriate size of the cohere model based on the capabilities of your machine.

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
#### [CohereForAI/c4ai-command-r-v01](https://huggingface.co/CohereForAI/c4ai-command-r-v01)
```log
Inference time: xxxxx s
-------------------- Prompt --------------------

<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>What is AI?<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>

-------------------- Output --------------------

<|START_OF_TURN_TOKEN|><|USER_TOKEN|>What is AI?<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>
Artificial Intelligence, or AI, is a fascinating field of study that aims to create intelligent machines that can mimic human cognitive functions and perform complex tasks. AI strives to
```
