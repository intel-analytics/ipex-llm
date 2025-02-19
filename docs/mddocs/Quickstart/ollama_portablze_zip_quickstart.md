# Run Ollama Portable Zip on Intel GPU with IPEX-LLM
<p>
  <b>< English</b> | <a href='./ollama_portablze_zip_quickstart.zh-CN.md'>中文</a> >
</p>

This guide demonstrates how to use [Ollama portable zip](https://github.com/intel/ipex-llm/releases/tag/v2.2.0-nightly) to directly run Ollama on Intel GPU with `ipex-llm` (without the need of manual installations).

## Table of Contents
- [Windows Quickstart](#windows-quickstart)
  - [Prerequisites](#prerequisites)
  - [Step 1: Download and Unzip](#step-1-download-and-unzip)
  - [Step 2: Start Ollama Serve](#step-2-start-ollama-serve)
  - [Step 3: Run Ollama](#step-3-run-ollama)
- [Linux Quickstart](#linux-quickstart)
  - [Prerequisites](#prerequisites-1)
  - [Step 1: Download and Extract](#step-1-download-and-extract)
  - [Step 2: Start Ollama Serve](#step-2-start-ollama-serve-1)
  - [Step 3: Run Ollama](#step-3-run-ollama-1)
- [Tips & Troubleshooting](#tips--troubleshooting)

## Windows Quickstart

### Prerequisites

Check your GPU driver version, and update it if needed:

- For Intel Core Ultra processors (Series 2) or Intel Arc B-Series GPU, we recommend updating your GPU driver to the [latest](https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html)

- For other Intel iGPU/dGPU, we recommend using GPU driver version [32.0.101.6078](https://www.intel.com/content/www/us/en/download/785597/834050/intel-arc-iris-xe-graphics-windows.html)

### Step 1: Download and Unzip

Download IPEX-LLM Ollama portable zip for Windows users from the [link](https://github.com/intel/ipex-llm/releases/tag/v2.2.0-nightly).

Then, extract the zip file to a folder.

### Step 2: Start Ollama Serve

Double-click `start-ollama.bat` in the extracted folder to start the Ollama service. A window will then pop up as shown below:

<div align="center">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/ollama_portable_start_ollama.png"  width=80%/>
</div>

### Step 3: Run Ollama

You could then use Ollama to run LLMs on Intel GPUs as follows:

- Open "Command Prompt" (cmd), and enter the extracted folder through `cd /d PATH\TO\EXTRACTED\FOLDER`
- Run `ollama run deepseek-r1:7b` in the "Command Prompt" (you may use any other model)

<div align="center">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/ollama_portable_run_ollama.png"  width=80%/>
</div>

## Linux Quickstart

### Prerequisites

Check your GPU driver version, and update it if needed:

- For client GPU, like A-series, B-series and integrated GPU, we recommend following [Intel client GPU driver installing guide](https://dgpu-docs.intel.com/driver/client/overview.html) to install your GPU driver.

### Step 1: Download and Extract

Download IPEX-LLM Ollama portable tgz for Ubuntu users from the [link](https://github.com/intel/ipex-llm/releases/tag/v2.2.0-nightly).

Then open a terminal, extract the tgz file to a folder.

```bash
cd PATH/TO/DOWNLOADED/TGZ
tar xvf [Downloaded tgz file]
```

### Step 2: Start Ollama Serve

Enter the extracted folder, and run `start-ollama.sh` to start Ollama service.  

```bash
cd PATH/TO/EXTRACTED/FOLDER
./start-ollama.sh
```

<div align="center">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/ollama_portable_start_ollama_ubuntu.png"  width=80%/>
</div>


### Step 3: Run Ollama

You could then use Ollama to run LLMs on Intel GPUs as follows:

- Open another ternimal, and enter the extracted folder through `cd PATH/TO/EXTRACTED/FOLDER`
- Run `./ollama run deepseek-r1:7b` (you may use any other model)

<div align="center">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/ollama_portable_run_ollama_ubuntu.png"  width=80%/>
</div>


## Tips & Troubleshooting

### Speed up model download using alternative sources

Ollama by default downloads model from [Ollama library](https://ollama.com/library). By setting the environment variable `IPEX_LLM_MODEL_SOURCE` to `modelscope`/`ollama` **before Run Ollama**, you could switch the source from which the model is downloaded first.

For example, if you would like to run `deepseek-r1:7b` but the download speed from Ollama library is quite slow, you could use [its model source](https://www.modelscope.cn/models/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF) from [ModelScope](https://www.modelscope.cn/models) instead, through:

- For **Windows** users:

  - Open "Command Prompt", and navigate to the extracted folder by `cd /d PATH\TO\EXTRACTED\FOLDER`
  - Run `set IPEX_LLM_MODEL_SOURCE=modelscope` in "Command Prompt"
  - Run `ollama run deepseek-r1:7b`

- For **Linux** users:

  - In a terminal other than the one for Ollama serve, navigate to the extracted folder by `cd PATH\TO\EXTRACTED\FOLDER`
  - Run `export IPEX_LLM_MODEL_SOURCE=modelscope` in the terminal
  - Run `./ollama run deepseek-r1:7b`

> [!TIP]
> Model downloaded with `set IPEX_LLM_MODEL_SOURCE=modelscope` will still show actual model id in `ollama list`, e.g.
> ```
> NAME                                                             ID              SIZE      MODIFIED
> modelscope.cn/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF:Q4_K_M    f482d5af6aec    4.7 GB    About a minute ago
> ```
> Except for `ollama run` and `ollama pull`, the model should be identified through its actual id, e.g. `ollama rm modelscope.cn/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF:Q4_K_M`

### Increase context length in Ollama

By default, Ollama runs model with a context window of 2048 tokens. That is, the model can "remember" at most 2048 tokens of context.

To increase the context length, you could set environment variable `IPEX_LLM_NUM_CTX` **before Start Ollama Serve**, as shwon below (if Ollama serve is already running, please make sure to stop it first):

- For **Windows** users:

  - Open "Command Prompt", and navigate to the extracted folder through `cd /d PATH\TO\EXTRACTED\FOLDER`
  - Set `IPEX_LLM_NUM_CTX` to the desired length in the "Command Prompt, e.g. `set IPEX_LLM_NUM_CTX=16384`
  - Start Ollama serve through `start-ollama.bat`

- For **Linux** users:

  - In a terminal, navigate to the extracted folder through `cd PATH\TO\EXTRACTED\FOLDER`
  - Set `IPEX_LLM_NUM_CTX` to the desired length in the terminal, e.g. `export IPEX_LLM_NUM_CTX=16384`
  - Start Ollama serve through `./start-ollama.sh`

> [!TIP]
> `IPEX_LLM_NUM_CTX` has a higher priority than the `num_ctx` settings in a models' `Modelfile`.

### Select Intel GPU for running Ollama when multiple Intel GPUs are available

If your machine has multiple Intel GPUs, Ollama will by default runs on all of them.

To specify which Intel GPU you would like Ollama to use, you could set environment variable `ONEAPI_DEVICE_SELECTOR` **before Start Ollama Serve**, as follows (if Ollama serve is already running, please make sure to stop it first):

- Identify the id (e.g. 0, 1, etc.) for your multiple GPUs. You could find them in the logs of Ollama serve when loading any models, e.g.:

  <div align="center">
    <img src="https://llm-assets.readthedocs.io/en/latest/_images/ollama_portable_start_ollama_multi_gpus.pn"  width=80%/>
  </div>

- For **Windows** users:

  - Open "Command Prompt", and navigate to the extracted folder by `cd /d PATH\TO\EXTRACTED\FOLDER`
  - In the "Command Prompt", set `ONEAPI_DEVICE_SELECTOR` to define the Intel GPU you want to use, e.g. `set ONEAPI_DEVICE_SELECTOR=level_zero:0`, in which `0` should be changed to your desired GPU id
  - Start Ollama serve through `start-ollama.bat`

- For **Linux** users:

  - In a terminal, navigate to the extracted folder by `cd PATH\TO\EXTRACTED\FOLDER`
  - Set `ONEAPI_DEVICE_SELECTOR` to define the Intel GPU you want to use, e.g. `export ONEAPI_DEVICE_SELECTOR=level_zero:0`, in which `0` should be changed to your desired GPU id
  - Start Ollama serve through `./start-ollama.sh`

### Additional models supported after Ollama v0.5.4

The currently Ollama Portable Zip is based on Ollama v0.5.4; in addition, the following new models have also been supported in the Ollama Portable Zip:

  | Model  | Download | Model Link |
  | - | - | - |
  | DeepSeek-R1 | `ollama run deepseek-r1` | [deepseek-r1](https://ollama.com/library/deepseek-r1) |
  | Openthinker | `ollama run openthinker` | [openthinker](https://ollama.com/library/openthinker) |
  | DeepScaleR | `ollama run deepscaler` | [deepscaler](https://ollama.com/library/deepscaler) |
  | Phi-4 | `ollama run phi4` | [phi4](https://ollama.com/library/phi4) |
  | Dolphin 3.0 | `ollama run dolphin3` | [dolphin3](https://ollama.com/library/dolphin3) |
  | Smallthinker | `ollama run smallthinker` | [smallthinker](https://ollama.com/library/smallthinker) |
  | Granite3.1-Dense |  `ollama run granite3-dense` | [granite3.1-dense](https://ollama.com/library/granite3.1-dense) |
  | Granite3.1-Moe-3B | `ollama run granite3-moe` | [granite3.1-moe](https://ollama.com/library/granite3.1-moe) |