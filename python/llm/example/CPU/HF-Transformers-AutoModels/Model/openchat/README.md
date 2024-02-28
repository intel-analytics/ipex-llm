# OpenChat

In this directory, you will find examples on how you could apply BigDL-LLM INT4 optimizations on OpenChat models. For illustration purposes, we utilize the [OpenChat-3.5-0106](https://huggingface.co/openchat/openchat-3.5-0106) as a reference OpenChat model.

## Requirements

To run these examples with BigDL-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Predict Tokens using `generate()` API

In the example [generate.py](./generate.py), we show a basic use case for a OpenChat model to predict the next N tokens using `generate()` API, with BigDL-LLM INT4 optimizations.

### 1. Install

We suggest using conda to manage environment:

```bash
conda create -n llm python=3.9
conda activate llm

pip install bigdl-llm[all] # install bigdl-llm with 'all' option
```

### 2. Run
```
python ./generate.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt PROMPT --n-predict N_PREDICT
```

Arguments info:

- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the OpenChat model to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `OpenChat/OpenChat-3.5-0106`.
- `--prompt PROMPT`: argument defining the prompt to be inferred (with integrated prompt format for chat). It is default to be `'AI是什么？'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

> **Note**: When loading the model in 4-bit, BigDL-LLM converts linear layers in the model into INT4 format. In theory, a *X*B model saved in 16-bit will requires approximately 2*X* GB of memory for loading, and ~0.5*X* GB memory for further inference.
>
> Please select the appropriate size of the OpenChat model based on the capabilities of your machine.

#### 2.1 Client

On client Windows machine, it is recommended to run directly with full utilization of all cores:

```powershell
python ./generate.py 
```

#### 2.2 Server

For optimal performance on server, it is recommended to set several environment variables (refer to [here](../README.md#best-known-configuration-on-linux) for more information), and run the example with all the physical cores of a single socket.

E.g. on Linux,

```bash
# set BigDL-LLM env variables
source bigdl-llm-init

# e.g. for a server with 48 cores per socket
export OMP_NUM_THREADS=48
numactl -C 0-47 -m 0 python ./generate.py
```

#### 2.3 Sample Output

#### [OpenChat-3.5-0106](https://huggingface.co/openchat/openchat-3.5-0106)

```log
Inference time: xxxx s
-------------------- Output --------------------
AI是什么？ AI是人工智能的缩写，是指计算机科学家和工程师通过模拟人类思维和行为的方式来创造出能够自主地解决问题、学习和适应环境的计算机系统。
```

```log
Inference time: xxxx s
-------------------- Output --------------------
What is AI? AI is an abbreviation for Artificial Intelligence. It is a branch of computer science that deals with the creation of intelligent machines that work and react like humans. AI focuses on the development of computer systems that can perform tasks which would require intelligence if done by humans.
```
