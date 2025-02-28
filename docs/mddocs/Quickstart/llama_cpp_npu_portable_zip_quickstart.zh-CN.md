# 使用 IPEX-LLM 在 Intel NPU 上运行 llama.cpp Portable Zip
<p>
   < <a href='./llama_cpp_npu_portable_zip_quickstart.md'>English</a> | <b>中文</b> >
</p>

IPEX-LLM 提供了 llama.cpp 的相关支持以在 Intel NPU 上运行 GGUF 模型。本指南演示如何使用 [llama.cpp NPU portable zip](https://github.com/intel/ipex-llm/releases/tag/v2.2.0-nightly) 在 Intel NPU 上直接免安装运行。

> [!IMPORTANT]
> 
> - IPEX-LLM 在 Intel NPU 上暂时只支持 Windows。
> - 目前支持的模型有 `meta-llama/Llama-3.2-3B-Instruct`, `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` 和 `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`。


## 目录
- [系统环境安装](#系统环境准备)
- [步骤 1：下载和解压](#步骤-1下载和解压)
- [步骤 2：启动](#步骤-2启动)
- [步骤 3：运行 GGUF 模型](#步骤-3运行-gguf-模型)
- [更多信息](npu_quickstart.md)


## 系统环境准备

检查你的 NPU 驱动程序版本，并根据需要进行更新：

- 请使用 NPU 驱动版本 [32.0.100.3104](https://www.intel.com/content/www/us/en/download/794734/838895/intel-npu-driver-windows.html)
- 你也可以参考这里 (https://github.com/intel/ipex-llm/blob/main/docs/mddocs/Quickstart/npu_quickstart.md#update-npu-driver) 了解更多关于 NPU 驱动程序更新的细节

## 步骤 1：下载和解压

从此[链接](https://github.com/intel/ipex-llm/releases/tag/v2.2.0-nightly)下载 IPEX-LLM llama.cpp NPU portable zip。

然后，将 zip 文件解压到一个文件夹中。

## 步骤 2：启动

- 打开命令提示符（cmd），并通过在命令行输入指令 "cd /d PATH\TO\EXTRACTED\FOLDER" 进入解压缩后的文件夹
- 根据你的设备完成运行配置：
  - 对于 **处理器为 2xxV 的 Intel Core™ Ultra Processors (Series 2) (代号 Lunar Lake)**:

    - 对于 Intel Core™ Ultra 7 Processor 258V:
        不需要额外的配置

    - 对于 Intel Core™ Ultra 5 Processor 228V & 226V:
        ```cmd
        set IPEX_LLM_NPU_DISABLE_COMPILE_OPT=1
        ```

  - 对于 **处理器为 2xxK 或者 2xxH 的 Intel Core™ Ultra Processors (Series 2) (代号 Arrow Lake)**:
    ```cmd
    set IPEX_LLM_NPU_ARL=1
    ```

  - 对于 **处理器为 1xxH 的 Intel Core™ Ultra Processors (Series 1) (代号 Meteor Lake)**:
    ```cmd
    set IPEX_LLM_NPU_MTL=1
    ```

## 步骤 3：运行 GGUF 模型

你可以在命令行中使用 cli 工具 `llama-cli-npu.exe` 以在 Intel NPU 上运行 GGUF 模型:

```cmd
llama-cli-npu.exe -m DeepSeek-R1-Distill-Qwen-7B-Q6_K.gguf -n 32 --prompt "What is AI?"
```
