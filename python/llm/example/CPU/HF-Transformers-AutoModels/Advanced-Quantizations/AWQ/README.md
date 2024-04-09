# AWQ

This example shows how to directly run 4-bit AWQ models using IPEX-LLM on Intel CPU.

## Verified Models

### Auto-AWQ Backend

- [Llama-2-7B-Chat-AWQ](https://huggingface.co/TheBloke/Llama-2-7B-Chat-AWQ)
- [CodeLlama-7B-AWQ](https://huggingface.co/TheBloke/CodeLlama-7B-AWQ)
- [Mistral-7B-Instruct-v0.1-AWQ](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-AWQ)
- [Mistral-7B-v0.1-AWQ](https://huggingface.co/TheBloke/Mistral-7B-v0.1-AWQ)
- [vicuna-7B-v1.5-AWQ](https://huggingface.co/TheBloke/vicuna-7B-v1.5-AWQ)
- [vicuna-13B-v1.5-AWQ](https://huggingface.co/TheBloke/vicuna-13B-v1.5-AWQ)
- [llava-v1.5-13B-AWQ](https://huggingface.co/TheBloke/llava-v1.5-13B-AWQ)
- [Yi-6B-AWQ](https://huggingface.co/TheBloke/Yi-6B-AWQ)
- [Yi-34B-AWQ](https://huggingface.co/TheBloke/Yi-34B-AWQ)
- [Mixtral-8x7B-Instruct-v0.1-AWQ](https://huggingface.co/ybelkada/Mixtral-8x7B-Instruct-v0.1-AWQ)

### llm-AWQ Backend

- [vicuna-7b-1.5-awq](https://huggingface.co/ybelkada/vicuna-7b-1.5-awq)

## Requirements

To run these examples with IPEX-LLM, we have some recommended requirements for your machine, please refer to [here](../../../README.md#system-support) for more information.

## Example: Predict Tokens using `generate()` API

In the example [generate.py](./generate.py), we show a basic use case for a AWQ model to predict the next N tokens using `generate()` API, with IPEX-LLM INT4 optimizations.

### 1. Install

We suggest using conda to manage environment:

```bash
conda create -n llm python=3.11
conda activate llm

pip install autoawq==0.1.8 --no-deps
pip install --pre --upgrade ipex-llm[all] # install ipex-llm with 'all' option
pip install transformers==4.35.0
pip install accelerate==0.25.0
pip install einops
```
**Note: For Mixtral model, please use transformers 4.36.0:**
```bash
pip install transformers==4.36.0
```

### 2. Run

```
python ./generate.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt PROMPT --n-predict N_PREDICT
```

Arguments info:

- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the AWQ model (e.g. `TheBloke/Llama-2-7B-Chat-AWQ`, `TheBloke/Mistral-7B-Instruct-v0.1-AWQ`, `TheBloke/Mistral-7B-v0.1-AWQ`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'TheBloke/Llama-2-7B-Chat-AWQ'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'What is AI?'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

> **Note**: When loading the model in 4-bit, IPEX-LLM converts linear layers in the model into INT4 format. In theory, a *X*B model saved in 16-bit will requires approximately 2*X* GB of memory for loading, and ~0.5*X* GB memory for further inference.
>
> Please select the appropriate size of the model based on the capabilities of your machine.

#### 2.1 Client

On client Windows machine, it is recommended to run directly with full utilization of all cores:

```powershell
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

#### [TheBloke/Llama-2-7B-Chat-AWQ](https://huggingface.co/TheBloke/Llama-2-7B-Chat-AWQ)

```log
Inference time: xxxx s
-------------------- Prompt --------------------
### HUMAN:
What is AI?

### RESPONSE:

-------------------- Output --------------------
### HUMAN:
What is AI?

### RESPONSE:

Artificial intelligence (AI) is the ability of machines to perform tasks that typically require human intelligence, such as learning, problem-solving, decision
```
