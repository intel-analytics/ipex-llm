# BigDL-LLM More Data Types for Low-Bit Inference on Large Language Model

In this example, we show how to apply BigDL-LLM low-bit optimizations (including INT8/INT5/INT4) to Llama2 model, and then run inference on the optimized low-bit model.

## 0. Requirements
To run this example with BigDL-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case of low-bit optimizations on a Llama2 model to predict the next N tokens using `generate()` API, with BigDL-LLM low-bit optimizations.
### 1. Install
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.9
conda activate llm

pip install bigdl-llm[all] # install bigdl-llm with 'all' option
```

### 2. Run
```
python ./generate.py
```
In the example, several arguments can be passed to satisfy your requirements:

- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Llama2 model (e.g. `meta-llama/Llama-2-7b-chat-hf` and `meta-llama/Llama-2-13b-chat-hf`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'meta-llama/Llama-2-7b-chat-hf'`.
- `--low-bit`: argument defining the low-bit optimization data type, options are sym_int4, asym_int4, sym_int5, asym_int5 or sym_int8. (sym_int4 means symmetric int 4, asym_int4 means asymmetric int 4, etc.). Relevant low bit optimizations will be applied to the model.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'What is AI?'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

#### 2.1 Client
On client Windows machine, it is recommended to run directly with full utilization of all cores:
```powershell
python ./generate.py
```

#### 2.2 Server
For optimal performance on server, it is recommended to set several environment variables (refer to [here](../README.md#best-known-configuration-on-linux) for more information), and run the example with all the physical cores of a single socket.

E.g. on Linux,
```bash
# set BigDL-Nano env variables
source bigdl-nano-init

# e.g. for a server with 48 cores per socket
export OMP_NUM_THREADS=48
numactl -C 0-47 -m 0 python ./generate.py
```

#### 2.3 Sample Output
#### [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
```log
Inference time: xxxx s
-------------------- Output --------------------
### HUMAN:
What is AI?

### RESPONSE:

AI is a term used to describe the development of computer systems that can perform tasks that typically require human intelligence, such as understanding natural language, recognizing images
```