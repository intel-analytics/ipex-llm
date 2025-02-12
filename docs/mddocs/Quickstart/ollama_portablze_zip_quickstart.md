# Run Ollama Portable Zip on Intel GPU with IPEX-LLM

This guide demonstrates how to use **Ollama portable zip** to directly run Ollama on Intel GPU with `ipex-llm` (without the need of manual installations).

> [!NOTE]
> Currently, IPEX-LLM only provides Ollama portable zip on Windows.

## Table of Contents
- [Prerequisites](#prerequisitesa)
- [Step 1: Download and Unzip](#step-1-download-and-unzip)
- [Step 2: Start Ollama Serve](#step-2-start-ollama-serve)
- [Step 3: Run Ollama](#step-3-run-ollama)

## Prerequisites

Check your GPU driver version, and update it if needed:

- For Intel Core Ultra processors (Series 2) or Intel Arc B-Series GPU, we recommend updating your GPU driver to the [latest](https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html)

- For other Intel iGPU/dGPU, we recommend using GPU driver version [32.0.101.6078](https://www.intel.com/content/www/us/en/download/785597/834050/intel-arc-iris-xe-graphics-windows.html)

## Step 1: Download and Unzip

Download IPEX-LLM Ollama portable zip from the [link](https://github.com/intel/ipex-llm/releases/download/v2.2.0-rc/ollama-ipex-llm-0.5.4-20250211.zip).

Then, extract the zip file to a folder.

## Step 2: Start Ollama Serve

Double-click `start-ollama.bat` in the extracted folder to start the Ollama service. A window will then pop up as shown below:

<div align="center">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/start_ollama.png"  width=80%/>
</div>


## Step 3: Run Ollama

You could then use Ollama to run LLMs on Intel GPUs as follows:

- Open "Command Prompt" (cmd), and enter the extracted folder through `cd /d PATH\TO\EXTRACTED\FOLDER`
- Run `ollama run deepseek-r1:7b` in the "Command Prompt" (you may use any other model)