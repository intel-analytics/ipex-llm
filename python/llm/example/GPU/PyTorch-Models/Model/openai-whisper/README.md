# Whisper

In this directory, you will find examples of how to use IPEX-LLM to optimize OpenAI Whisper models within the `openai-whisper` Python library. For illustration purposes, we utilize the [whisper-tiny](https://github.com/openai/whisper/blob/main/model-card.md) as a reference Whisper model.

## Requirements
To run these examples with IPEX-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../../../README.md#requirements) for more information.

## Example: Recognize Tokens using `transcribe()` API
In the example [recognize.py](./recognize.py), we show a basic use case for a Whisper model to conduct transcription using `transcribe()` API, with IPEX-LLM INT4 optimizations on Intel GPUs.
### 1. Install
#### 1.1 Installation on Linux
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.11
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
pip install -U openai-whisper
pip install librosa # required by audio processing 
```

#### 1.2 Installation on Windows
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.11 libuv
conda activate llm

# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
pip install -U openai-whisper
pip install librosa
```

### 2. Configures OneAPI environment variables for Linux

> [!NOTE]
> Skip this step if you are running on Windows.

This is a required step on Linux for APT or offline installed oneAPI. Skip this step for PIP-installed oneAPI.

```bash
source /opt/intel/oneapi/setvars.sh
```

### 3. Runtime Configurations
For optimal performance, it is recommended to set several environment variables. Please check out the suggestions based on your device.
#### 3.1 Configurations for Linux
<details>

<summary>For Intel Arc™ A-Series Graphics and Intel Data Center GPU Flex Series</summary>

```bash
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export SYCL_CACHE_PERSISTENT=1
```

</details>

<details>

<summary>For Intel Data Center GPU Max Series</summary>

```bash
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export SYCL_CACHE_PERSISTENT=1
export ENABLE_SDP_FUSION=1
```
> Note: Please note that `libtcmalloc.so` can be installed by `conda install -c conda-forge -y gperftools=2.10`.
</details>

<details>

<summary>For Intel iGPU</summary>

```bash
export SYCL_CACHE_PERSISTENT=1
```

</details>

#### 3.2 Configurations for Windows
<details>

<summary>For Intel iGPU and Intel Arc™ A-Series Graphics</summary>

```cmd
set SYCL_CACHE_PERSISTENT=1
```

</details>


> [!NOTE]
> For the first time that each model runs on Intel iGPU/Intel Arc™ A300-Series or Pro A60, it may take several minutes to compile.
### 4. Running examples

```bash
python ./recognize.py --audio-file AUDIO_FILE
```

Arguments info:
- `--model-name MODEL_NAME`: argument defining the model name(tiny, medium, base, etc.) for the Whisper model to be downloaded. It is one of the official model names listed by `whisper.available_models()`, or path to a model checkpoint containing the model dimensions and the model state_dict. It is default to be `'tiny'`.
- `--audio-file AUDIO_FILE`: argument defining the path of the audio file to be recognized.
- `--language LANGUAGE`: argument defining language to be transcribed. It is default to be `english`.

> **Note**: When loading the model in 4-bit, IPEX-LLM converts linear layers in the model into INT4 format. In theory, a *X*B model saved in 16-bit will requires approximately 2*X* GB of memory for loading, and ~0.5*X* GB memory for further inference.
>
> Please select the appropriate size of the Whisper model based on the capabilities of your machine.

#### Sample Output
#### [whisper-tiny](https://github.com/openai/whisper/blob/main/model-card.md)

For audio file(.wav) download from https://www.youtube.com/watch?v=-LIIf7E-qFI, it should be extracted as:
```log
[00:00.000 --> 00:10.000]  I don't know who you are.
[00:10.000 --> 00:15.000]  I don't know what you want.
[00:15.000 --> 00:21.000]  If you're looking for ransom, I can tell you I don't know money, but what I do have.
[00:21.000 --> 00:24.000]  I'm a very particular set of skills.
[00:24.000 --> 00:27.000]  The skills I have acquired are very long career.
[00:27.000 --> 00:31.000]  The skills that make me a nightmare for people like you.
[00:31.000 --> 00:35.000]  If you let my daughter go now, that'll be the end of it.
[00:35.000 --> 00:39.000]  I will not look for you. I will not pursue you.
[00:39.000 --> 00:45.000]  But if you don't, I will look for you. I will find you.
[00:45.000 --> 00:48.000]  And I will kill you.
[00:48.000 --> 00:53.000]  Good luck.
Inference time: xxxx s
-------------------- Output --------------------
 I don't know who you are. I don't know what you want. If you're looking for ransom, I can tell you I don't know money, but what I do have. I'm a very particular set of skills. The skills I have acquired are very long career. The skills that make me a nightmare for people like you. If you let my daughter go now, that'll be the end of it. I will not look for you. I will not pursue you. But if you don't, I will look for you. I will find you. And I will kill you. Good luck.
```
