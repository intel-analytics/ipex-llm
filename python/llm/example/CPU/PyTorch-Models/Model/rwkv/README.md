# RWKV

In this directory, you will find examples on how you could apply BigDL-LLM INT4 optimizations on RWKV models. For illustration purposes, we utilize the [RWKV/rwkv-4-world-7b](https://huggingface.co/RWKV/rwkv-4-world-7b) as a reference RWKV model.

> **Note**: If you want to download the Hugging Face *Transformers* model, please refer to [here](https://huggingface.co/docs/hub/models-downloading#using-git).
>
> BigDL-LLM optimizes the *Transformers* model in INT4 precision at runtime, and thus no explicit conversion is needed.

## Requirements
To run these examples with BigDL-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a RWKV model to predict the next N tokens using `generate()` API, with BigDL-LLM INT4 optimizations.
### 1. Install
We suggest using conda to manage the Python environment. For more information about conda installation, please refer to [here](https://docs.conda.io/en/latest/miniconda.html#).

After installing conda, create a Python environment for BigDL-LLM:
```bash
conda create -n llm python=3.9 # recommend to use Python 3.9
conda activate llm

pip install --pre --upgrade bigdl-llm[all] # install the latest bigdl-llm nightly build with 'all' option
```

### 2. Run
After setting up the Python environment, you could run the example by following steps.

> **Note**: When loading the model in 4-bit, BigDL-LLM converts linear layers in the model into INT4 format. In theory, a *X*B model saved in 16-bit will requires approximately 2*X* GB of memory for loading, and ~0.5*X* GB memory for further inference.
>
> Please select the appropriate size of the RWKV model based on the capabilities of your machine.

#### 2.1 Client
On client Windows machines, it is recommended to run directly with full utilization of all cores:
```powershell
python ./generate.py --prompt "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley"
```
More information about arguments can be found in [Arguments Info](#23-arguments-info) section. The expected output can be found in [Sample Output](#24-sample-output) section.

#### 2.2 Server
For optimal performance on server, it is recommended to set several environment variables (refer to [here](../README.md#best-known-configuration-on-linux) for more information), and run the example with all the physical cores of a single socket.

E.g. on Linux,
```bash
# set BigDL-Nano env variables
source bigdl-nano-init

# e.g. for a server with 48 cores per socket
export OMP_NUM_THREADS=48
numactl -C 0-47 -m 0 python ./generate.py --prompt "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley"
```
More information about arguments can be found in [Arguments Info](#23-arguments-info) section. The expected output can be found in [Sample Output](#24-sample-output) section.

#### 2.3 Arguments Info
In the example, several arguments can be passed to satisfy your requirements:

- `--repo-id-or-model-path`: str, argument defining the huggingface repo id for the RWKV model to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'RWKV/rwkv-4-world-7b'`.
- `--prompt`: str, argument defining the prompt to be inferred (with integrated prompt format for chat). It is default to be "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley".
- `--n-predict`: int, argument defining the max number of tokens to predict. It is default to be `40`.

#### 2.4 Sample Output
#### [RWKV/rwkv-4-world-7b](https://huggingface.co/RWKV/rwkv-4-world-7b)
```log
Inference time: xxxx s
-------------------- Prompt --------------------
Question: 
In a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese.

Answer:
-------------------- Output --------------------
Question: 
In a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese.

Answer: 科学家在一个不为人知的谷地发现一群能说中文的龙。科学家惊讶地发现这些龙是中国的

```
