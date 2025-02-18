# Run Ollama Portable Zip on Intel GPU with IPEX-LLM
<p>
  <b>< English</b> | <a href='./ollama_portablze_zip_quickstart.zh-CN.md'>中文</a> >
</p>

This guide demonstrates how to use [Ollama portable zip](https://github.com/intel/ipex-llm/releases/tag/v2.2.0-nightly) to directly run Ollama on Intel GPU with `ipex-llm` (without the need of manual installations).

> [!NOTE]
> Currently, IPEX-LLM only provides Ollama portable zip on Windows.

## Table of Contents
- [Prerequisites](#prerequisitesa)
- [Step 1: Download and Unzip](#step-1-download-and-unzip)
- [Step 2: Start Ollama Serve](#step-2-start-ollama-serve)
- [Step 3: Run Ollama](#step-3-run-ollama)
- [Runtime Configurations](#runtime-configurations)

## Prerequisites

Check your GPU driver version, and update it if needed:

- For Intel Core Ultra processors (Series 2) or Intel Arc B-Series GPU, we recommend updating your GPU driver to the [latest](https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html)

- For other Intel iGPU/dGPU, we recommend using GPU driver version [32.0.101.6078](https://www.intel.com/content/www/us/en/download/785597/834050/intel-arc-iris-xe-graphics-windows.html)

## Step 1: Download and Unzip

Download IPEX-LLM Ollama portable zip from the [link](https://github.com/intel/ipex-llm/releases/tag/v2.2.0-nightly).

Then, extract the zip file to a folder.

## Step 2: Start Ollama Serve

Double-click `start-ollama.bat` in the extracted folder to start the Ollama service. A window will then pop up as shown below:

<div align="center">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/ollama_portable_start_ollama.png"  width=80%/>
</div>

## Step 3: Run Ollama

You could then use Ollama to run LLMs on Intel GPUs as follows:

- Open "Command Prompt" (cmd), and enter the extracted folder through `cd /d PATH\TO\EXTRACTED\FOLDER`
- Run `ollama run deepseek-r1:7b` in the "Command Prompt" (you may use any other model)

<div align="center">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/ollama_portable_run_ollama.png"  width=80%/>
</div>

## Runtime Configurations

IPEX-LLM provides several environment variables settings to customize the Ollama runtime experience on Intel GPU:

### `IPEX_LLM_NUM_CTX`

By default, Ollama runs model with a context window of 2048 tokens. That is, the model can "remember" at most 2048 tokens of context.

To increase the context windows, you could set environment variable `IPEX_LLM_NUM_CTX` before [starting Ollama serve](#sta), as shwon below:

- Open "Command Prompt" (cmd), and navigate to the extracted folder through `cd /d PATH\TO\EXTRACTED\FOLDER`
- Set `IPEX_LLM_NUM_CTX` in the "Command Prompt, e.g. `set IPEX_LLM_NUM_CTX=16384`
- Start Ollama serve through `start-ollama.bat`

> [!TIP]
> `IPEX_LLM_NUM_CTX` has a higher priority than the `num_ctx` settings in a models' `Modelfile`.

### `IPEX_LLM_MODEL_SOURCE`

Ollama by default downloads model from [Ollama library](https://ollama.com/library). By setting the environment variable `IPEX_LLM_MODEL_SOURCE` to `modelscope`/`ollama` before [run Ollama](#step-3-run-ollama), you could switch the source from which the model is downloaded first.

For example, if you would like to run `deepseek-r1:7b` but the download speed from Ollama library is quite slow, you could use [model source](https://www.modelscope.cn/models/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF) from [ModelScope](https://www.modelscope.cn/models) instead, through:

- Open "Command Prompt" (cmd), and navigate to the extracted folder by `cd /d PATH\TO\EXTRACTED\FOLDER`
- Run `set IPEX_LLM_MODEL_SOURCE=modelscope` in "Command Prompt"
- Run `ollama run deepseek-r1:7b`

> [!TIP]
> Model downloaded with `set IPEX_LLM_MODEL_SOURCE=modelscope` will still show actual model id in `ollama list`, e.g.
> ```
> NAME                                                             ID              SIZE      MODIFIED
> modelscope.cn/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF:Q4_K_M    f482d5af6aec    4.7 GB    About a minute ago
> ```
> Except for `ollama run` and `ollama pull`, the model should be identified through its actual id, e.g. `ollama rm modelscope.cn/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF:Q4_K_M`