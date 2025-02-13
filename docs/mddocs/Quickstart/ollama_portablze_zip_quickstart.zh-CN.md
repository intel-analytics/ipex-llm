# 使用 IPEX-LLM 在 Intel GPU 上运行 Ollama Portable Zip
<p>
   < <a href='./ollama_portablze_zip_quickstart.md'>English</a> | <b>中文</b> >
</p>

本指南演示如何使用 [Ollama portable zip](https://github.com/intel/ipex-llm/releases/download/v2.2.0-nightly/ollama-0.5.4-ipex-llm-2.2.0b20250211.zip) 通过 `ipex-llm` 在 Intel GPU 上直接免安装运行 Ollama。

> [!NOTE]
> 目前，IPEX-LLM 仅在 Windows 上提供 Ollama portable zip。

## 目录
- [系统环境安装](#系统环境准备)
- [步骤 1：下载和解压](#步骤-1下载和解压)
- [步骤 2：启动 Ollama Serve](#步骤-2启动-ollama-serve)
- [步骤 3：运行 Ollama](#步骤-3运行-ollama)

## 系统环境准备

检查你的 GPU 驱动程序版本，并根据需要进行更新：

- 对于 Intel Core Ultra processors（系列 2）或 Intel Arc B系列 GPU，建议将 GPU 驱动程序更新至[最新版本](https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html)

- 对于其他 Intel iGPU/dGPU，建议使用 GPU 驱动程序的版本是[32.0.101.6078](https://www.intel.com/content/www/us/en/download/785597/834050/intel-arc-iris-xe-graphics-windows.html)

## 步骤 1：下载和解压

从此[链接](https://github.com/intel/ipex-llm/releases/download/v2.2.0-nightly/ollama-0.5.4-ipex-llm-2.2.0b20250211.zip)下载 IPEX-LLM Ollama portable zip。

然后，将 zip 文件解压到一个文件夹中。

## 步骤 2：启动 Ollama Serve

在解压后的文件夹中双击 `start-ollama.bat`，启动 Ollama Serve，会弹出如下图所示的窗口：

<div align="center">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/ollama_portable_start_ollama.png"  width=80%/>
</div>

## 步骤 3：运行 Ollama

在 Intel GPUs 上使用 Ollama 运行 LLMs，如下所示：

- 打开 "Command Prompt" (cmd)，通过此路径 `cd /d PATH\TO\EXTRACTED\FOLDER` 进入解压后的文件夹
- 在 "Command Prompt" 中运行 `ollama run deepseek-r1:7（可以将当前模型替换为你需要的模型）

<div align="center">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/ollama_portable_run_ollama.png"  width=80%/>
</div>
