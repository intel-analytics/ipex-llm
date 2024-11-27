# SpeechT5
In this directory, you will find examples on how you could use IPEX-LLM `optimize_model` API to accelerate SpeechT5 models. For illustration purposes, we utilize the [microsoft/speecht5_tts](https://huggingface.co/microsoft/speecht5_tts) as reference SpeechT5 models.

## Requirements
To run these examples with IPEX-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Synthesize speech with the given input text
In the example [synthesize_speech.py](./synthesize_speech.py), we show a basic use case for SpeechT5 model to synthesize speech based on the given text, with IPEX-LLM INT4 optimizations.
### 1. Install
#### 1.1 Installation on Linux
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.11
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

pip install "datasets==2.16.1" soundfile # additional package required for SpeechT5 to conduct generation
```

#### 1.2 Installation on Windows
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.11 libuv
conda activate llm

# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

pip install "datasets==2.16.1" soundfile # additional package required for SpeechT5 to conduct generation
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
python ./synthesize_speech.py --text 'Artificial intelligence refers to the development of computer systems that can perform tasks that typically require human intelligence.'
```

In the example, several arguments can be passed to satisfy your requirements:

- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the SpeechT5 model (e.g `microsoft/speecht5_tts`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'microsoft/speecht5_tts'`.
- `--repo-id-or-vocoder-path REPO_ID_OR_VOCODER_PATH`: argument defining the huggingface repo id for the SpeechT5 vocoder (e.g `microsoft/speecht5_hifigan`, which generates audio from a spectrogram) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'microsoft/speecht5_hifigan'`.
- `--repo-id-or-data-path REPO_ID_OR_DATA_PATH`: argument defining the huggingface repo id for the audio dataset (e.g. `Matthijs/cmu-arctic-xvectors`, which decides voice characteristics) to be downloaded, or the path to the huggingface dataset folder. It is default to be `'Matthijs/cmu-arctic-xvectors'`.
- `--text TEXT`: argument defining the text to synthesize speech. It is default to be `"Artificial intelligence refers to the development of computer systems that can perform tasks that typically require human intelligence."`.

#### 4.1 Sample Output

#### [microsoft/speecht5_tts](https://huggingface.co/microsoft/speecht5_tts)

Text: Artificial intelligence refers to the development of computer systems that can perform tasks that typically require human intelligence.

[Click here to hear sample output.](https://llm-assets.readthedocs.io/en/latest/_downloads/f0bebfbe8c350b71fe565a82192c079b/speech-t5-example-output.wav)
