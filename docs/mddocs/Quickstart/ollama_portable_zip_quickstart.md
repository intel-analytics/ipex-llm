# Run Ollama Portable Zip on Intel GPU with IPEX-LLM
<p>
  <b>< English</b> | <a href='./ollama_portable_zip_quickstart.zh-CN.md'>中文</a> >
</p>

This guide demonstrates how to use [Ollama portable zip](https://github.com/intel/ipex-llm/releases/tag/v2.2.0-nightly) to directly run Ollama on Intel GPU with `ipex-llm` (without the need of manual installations).

> [!NOTE]
> Ollama portable zip has been verified on:
> - Intel Core Ultra processors
> - Intel Core 11th - 14th gen processors
> - Intel Arc A-Series GPU
> - Intel Arc B-Series GPU

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
  - [Speed up model download using alternative sources](#speed-up-model-download-using-alternative-sources)
  - [Increase context length in Ollama](#increase-context-length-in-ollama)
  - [Select specific GPU(s) to run Ollama when multiple ones are available](#select-specific-gpus-to-run-ollama-when-multiple-ones-are-available)
  - [Tune performance](#tune-performance)
  - [Additional models supported after Ollama v0.5.4](#additional-models-supported-after-ollama-v054)
- [More details](ollama_quickstart.md)

## Windows Quickstart

> [!NOTE]
> We recommand using Windows 11 for Windows users.

### Prerequisites

Check your GPU driver version, and update it if needed:

- For Intel Core Ultra processors (Series 2) or Intel Arc B-Series GPU, we recommend updating your GPU driver to the [latest](https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html)

- For other Intel iGPU/dGPU, we recommend using GPU driver version [32.0.101.6078](https://www.intel.com/content/www/us/en/download/785597/834050/intel-arc-iris-xe-graphics-windows.html)

### Step 1: Download and Unzip

Download IPEX-LLM Ollama portable zip for Windows users from the [link](https://github.com/intel/ipex-llm/releases/tag/v2.2.0-nightly).

Then, extract the zip file to a folder.

### Step 2: Start Ollama Serve

Start Ollama serve as follows:

- Open "Command Prompt" (cmd), and enter the extracted folder through `cd /d PATH\TO\EXTRACTED\FOLDER`
- Run `start-ollama.bat` in the "Command Prompt. A window will then pop up as shown below:

<div align="center">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/ollama_portable_start_ollama_new.png"  width=80%/>
</div>

### Step 3: Run Ollama

You could then use Ollama to run LLMs on Intel GPUs through running `ollama run deepseek-r1:7b` in the same "Command Prompt" (not the pop-up window). You may use any other model.

<div align="center">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/ollama_portable_run_ollama_new.png"  width=80%/>
</div>

## Linux Quickstart

### Prerequisites

Check your GPU driver version, and update it if needed; we recommend following [Intel client GPU driver installation guide](https://dgpu-docs.intel.com/driver/client/overview.html) to install your GPU driver.

### Step 1: Download and Extract

Download IPEX-LLM Ollama portable tgz for Ubuntu users from the [link](https://github.com/intel/ipex-llm/releases/tag/v2.2.0-nightly).

Then open a terminal, extract the tgz file to a folder.

```bash
tar -xvf [Downloaded tgz file path]
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

Ollama by default downloads model from the Ollama library. By setting the environment variable `IPEX_LLM_MODEL_SOURCE` to `modelscope` or `ollama` **before running Ollama**, you could switch the source where the model is downloaded.

For example, if you would like to run `deepseek-r1:7b` but the download speed from the Ollama library is slow, you could download the model from ModelScope as follows:

- For **Windows** users:

  - In the "Command Prompt", navigate to the extracted folder by `cd /d PATH\TO\EXTRACTED\FOLDER`
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

To increase the context length, you could set environment variable `IPEX_LLM_NUM_CTX` **before staring Ollama Serve**, as shwon below (if Ollama serve is already running, please make sure to stop it first):

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

### Select specific GPU(s) to run Ollama when multiple ones are available

If your machine has multiple Intel GPUs, Ollama will by default runs on all of them.

To specify which Intel GPU(s) you would like Ollama to use, you could set environment variable `ONEAPI_DEVICE_SELECTOR` **before starting Ollama Serve**, as follows (if Ollama serve is already running, please make sure to stop it first):

- Identify the id (e.g. 0, 1, etc.) for your multiple GPUs. You could find them in the logs of Ollama serve when loading any models, e.g.:

  <div align="center">
    <img src="https://llm-assets.readthedocs.io/en/latest/_images/ollama_portable_multi_gpus.png"  width=80%/>
  </div>

- For **Windows** users:

  - Open "Command Prompt", and navigate to the extracted folder by `cd /d PATH\TO\EXTRACTED\FOLDER`
  - In the "Command Prompt", set `ONEAPI_DEVICE_SELECTOR` to define the Intel GPU(s) you want to use, e.g. `set ONEAPI_DEVICE_SELECTOR=level_zero:0` (on single Intel GPU), or `set ONEAPI_DEVICE_SELECTOR=level_zero:0;level_zero:1` (on multiple Intel GPUs), in which `0`, `1` should be changed to your desired GPU id
  - Start Ollama serve through `start-ollama.bat`

- For **Linux** users:

  - In a terminal, navigate to the extracted folder by `cd PATH\TO\EXTRACTED\FOLDER`
  - Set `ONEAPI_DEVICE_SELECTOR` to define the Intel GPU(s) you want to use, e.g. `export ONEAPI_DEVICE_SELECTOR=level_zero:0` (on single Intel GPU), or `export ONEAPI_DEVICE_SELECTOR="level_zero:0;level_zero:1"` (on multiple Intel GPUs), in which `0`, `1` should be changed to your desired GPU id
  - Start Ollama serve through `./start-ollama.sh`

### Tune performance

Here are some settings you could try to tune the performance:

#### Environment variable `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS`

The environment variable `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS` determines the usage of immediate command lists for task submission to the GPU. You could experiment with `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1` or `0` for best performance.

To enable `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS`, set it **before starting Ollama Serve**, as shown below (if Ollama serve is already running, please make sure to stop it first):

- For **Windows** users:

  - Open "Command Prompt", and navigate to the extracted folder through `cd /d PATH\TO\EXTRACTED\FOLDER`
  - Run `set SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1` in "Command Prompt"
  - Start Ollama serve through `start-ollama.bat`

- For **Linux** users:

  - In a terminal, navigate to the extracted folder through `cd PATH\TO\EXTRACTED\FOLDER`
  - Run `export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1` in the terminal
  - Start Ollama serve through `./start-ollama.sh`

> [!TIP]
> You could refer to [here](https://www.intel.com/content/www/us/en/developer/articles/guide/level-zero-immediate-command-lists.html) regarding more information about Level Zero Immediate Command Lists.

### Additional models supported after Ollama v0.5.4

The currently Ollama Portable Zip is based on Ollama v0.5.4; in addition, the following new models have also been supported in the Ollama Portable Zip:

| Model  | Download (Windows) | Download (Linux) | Model Link |
| - | - | - | - |
| DeepSeek-R1 | `ollama run deepseek-r1` | `./ollama run deepseek-r1` | [deepseek-r1](https://ollama.com/library/deepseek-r1) |
| Openthinker | `ollama run openthinker` | `./ollama run openthinker` | [openthinker](https://ollama.com/library/openthinker) |
| DeepScaleR | `ollama run deepscaler` | `./ollama run deepscaler` | [deepscaler](https://ollama.com/library/deepscaler) |
| Phi-4 | `ollama run phi4` | `./ollama run phi4` | [phi4](https://ollama.com/library/phi4) |
| Dolphin 3.0 | `ollama run dolphin3` | `./ollama run dolphin3` | [dolphin3](https://ollama.com/library/dolphin3) |
| Smallthinker | `ollama run smallthinker` |`./ollama run smallthinker` | [smallthinker](https://ollama.com/library/smallthinker) |
| Granite3.1-Dense |  `ollama run granite3-dense` | `./ollama run granite3-dense` | [granite3.1-dense](https://ollama.com/library/granite3.1-dense) |
| Granite3.1-Moe-3B | `ollama run granite3-moe` | `./ollama run granite3-moe` | [granite3.1-moe](https://ollama.com/library/granite3.1-moe) |
