# Phoenix

In this directory, you will find examples on how you could apply BigDL-LLM INT4 optimizations on Phoenix models. For illustration purposes, we utilize the [FreedomIntelligence/phoenix-inst-chat-7b](https://huggingface.co/FreedomIntelligence/phoenix-inst-chat-7b) as a reference Phoenix model.

## 0. Requirements
To run these examples with BigDL-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [generate.py](./generate.py), we show a basic use case for a Phoenix model to predict the next N tokens using `generate()` API, with BigDL-LLM INT4 optimizations.
### 1. Install
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.9
conda activate llm

pip install bigdl-llm[all] # install bigdl-llm with 'all' option
```

### 2. Config
It is recommended to set several environment variables for better performance. Please refer to [here](../README.md#best-known-configuration) for more information.

### 3. Run
```
python ./generate.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt PROMPT --n-predict N_PREDICT
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Phoenix model to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `"FreedomIntelligence/phoenix-inst-chat-7b"`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). It is default to be `'What is AI?'`.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `32`.

> **Note**: When loading the model in 4-bit, BigDL-LLM converts linear layers in the model into INT4 format. In theory, a *X*B model saved in 16-bit will requires approximately 2*X* GB of memory for loading, and ~0.5*X* GB memory for further inference.
>
> Please select the appropriate size of the Phoenix model based on the capabilities of your machine.

#### 3.1 Client
For better utilization of multiple cores on the client machine, it is recommended to use all the performance-cores along with their hyperthreads.

E.g. on Windows,
```powershell
# for a client machine with 8 Performance-cores
$env:OMP_NUM_THREADS=16
python ./generate.py
```

#### 3.2 Server
On server, it is recommended to run the example with all the physical cores of a single socket.

E.g. on Linux,
```bash
# for a server with 48 cores per socket
export OMP_NUM_THREADS=48
numactl -C 0-47 -m 0 python ./generate.py
```

#### 3.3 Sample Output
#### [FreedomIntelligence/phoenix-inst-chat-7b](https://huggingface.co/FreedomIntelligence/phoenix-inst-chat-7b)
```log
Inference time: xxxx s
-------------------- Prompt --------------------
<human>What is AI?<bot>
-------------------- Output --------------------
<human>What is AI?<bot> AI stands for Artificial Intelligence. It is a branch of computer science that focuses on creating intelligent machines that can perform tasks that typically require human intelligence, such as visual
```

```log
Inference time: xxxx s
-------------------- Prompt --------------------
<human>什么是人工智能？<bot>
-------------------- Output --------------------
<human>什么是人工智能？<bot> 人工智能（Artificial Intelligence，简称AI），是指计算机系统能够执行通常需要人类智能才能完成的任务。这些任务包括语言理解、视觉
```
