# 使用 IPEX-LLM 在 Intel GPU 上运行 Ollama Portable Zip
<p>
   < <a href='./ollama_portable_zip_quickstart.md'>English</a> | <b>中文</b> >
</p>

本指南演示如何使用 [Ollama portable zip](https://github.com/intel/ipex-llm/releases/tag/v2.2.0-nightly) 通过 `ipex-llm` 在 Intel GPU 上直接免安装运行 Ollama。

> [!NOTE]
> Ollama portable zip 在如下设备上进行了验证:
> - Intel Core Ultra processors
> - Intel Core 11th - 14th gen processors
> - Intel Arc A-Series GPU
> - Intel Arc B-Series GPU

## 目录
- [Windows用户指南](#windows用户指南)
  - [系统环境安装](#系统环境准备)
  - [步骤 1：下载和解压](#步骤-1下载和解压)
  - [步骤 2：启动 Ollama Serve](#步骤-2启动-ollama-serve)
  - [步骤 3：运行 Ollama](#步骤-3运行-ollama)
- [Linux用户指南](#linux用户指南)
  - [系统环境安装](#系统环境准备-1)
  - [步骤 1：下载和解压](#步骤-1下载和解压-1)
  - [步骤 2：启动 Ollama Serve](#步骤-2启动-ollama-serve-1)
  - [步骤 3：运行 Ollama](#步骤-3运行-ollama-1)
- [提示和故障排除](#提示和故障排除)
  - [通过切换源提升模型下载速度](#通过切换源提升模型下载速度)
  - [在 Ollama 中增加上下文长度](#在-ollama-中增加上下文长度)
  - [在多块 GPU 可用时选择特定的 GPU 来运行 Ollama](#在多块-gpu-可用时选择特定的-gpu-来运行-ollama)
  - [性能调优](#性能调优)
  - [Ollama v0.5.4 之后新增模型支持](#ollama-v054-之后新增模型支持)
- [更多信息](ollama_quickstart.zh-CN.md)

## Windows用户指南

> [!NOTE]
> 对于 Windows 用户，我们推荐使用 Windows 11。

### 系统环境准备

检查你的 GPU 驱动程序版本，并根据需要进行更新：

- 对于 Intel Core Ultra processors (Series 2) 或者 Intel Arc B-Series GPU，我们推荐将你的 GPU 驱动版本升级到[最新版本](https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html)

- 对于其他的 Intel 核显和独显，我们推荐使用 GPU 驱动版本 [32.0.101.6078](https://www.intel.com/content/www/us/en/download/785597/834050/intel-arc-iris-xe-graphics-windows.html)

### 步骤 1：下载和解压

从此[链接](https://github.com/intel/ipex-llm/releases/tag/v2.2.0-nightly)下载 IPEX-LLM Ollama portable zip。

然后，将 zip 文件解压到一个文件夹中。

### 步骤 2：启动 Ollama Serve

根据如下步骤启动 Ollama serve:

- 打开命令提示符（cmd），并通过在命令行输入指令 `cd /d PATH\TO\EXTRACTED\FOLDER` 进入解压缩后的文件夹
- 在命令提示符中运行 `start-ollama.bat` 即可启动 Ollama Serve。随后会弹出一个窗口，如下所示：

<div align="center">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/ollama_portable_start_ollama_new.png"  width=80%/>
</div>

### 步骤 3：运行 Ollama

接下来通过在相同的命令提示符（非弹出的窗口）中运行 `ollama run deepseek-r1:7b`（可以将当前模型替换为你需要的模型），即可在 Intel GPUs 上使用 Ollama 运行 LLMs：


<div align="center">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/ollama_portable_run_ollama_new.png"  width=80%/>
</div>

## Linux用户指南

### 系统环境准备

检查你的 GPU 驱动程序版本，并根据需要进行更新；我们推荐用户按照[消费级显卡驱动安装指南](https://dgpu-docs.intel.com/driver/client/overview.html)来安装 GPU 驱动。


### 步骤 1：下载和解压

从此[链接](https://github.com/intel/ipex-llm/releases/tag/v2.2.0-nightly)下载 IPEX-LLM Ollama portable tgz。

然后，开启一个终端，输入如下命令将 tgz 文件解压到一个文件夹中。
```bash
tar -xvf [Downloaded tgz file path]
```

### 步骤 2：启动 Ollama Serve

进入解压后的文件夹，执行`./start-ollama.sh`启动 Ollama Serve： 

```bash
cd PATH/TO/EXTRACTED/FOLDER
./start-ollama.sh
```

<div align="center">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/ollama_portable_start_ollama_ubuntu.png"  width=80%/>
</div>

### 步骤 3：运行 Ollama

在 Intel GPUs 上使用 Ollama 运行大语言模型，如下所示：

- 打开另外一个终端，并输入指令 `cd PATH/TO/EXTRACTED/FOLDER` 进入解压后的文件夹
- 在终端中运行 `./ollama run deepseek-r1:7b`（可以将当前模型替换为你需要的模型）

<div align="center">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/ollama_portable_run_ollama_ubuntu.png"  width=80%/>
</div>

## 提示和故障排除

### 通过切换源提升模型下载速度

Ollama 默认从 Ollama 库下载模型。通过在**运行 Ollama 之前**设置环境变量 `IPEX_LLM_MODEL_SOURCE` 为 `modelscope` 或 `ollama`，你可以切换模型的下载源。

例如，如果你想运行 `deepseek-r1:7b` 但从 Ollama 库的下载速度较慢，可以通过如下方式改用 ModelScope 上的模型源：

- 对于 **Windows** 用户：

  - 打开命令提示符通过 `cd /d PATH\TO\EXTRACTED\FOLDER` 命令进入解压后的文件夹
  - 在命令提示符中运行 `set IPEX_LLM_MODEL_SOURCE=modelscope`
  - 运行 `ollama run deepseek-r1:7b`

- 对于 **Linux** 用户：

  - 在另一个终端（不同于运行 Ollama serve 的终端）中，输入指令 `cd PATH/TO/EXTRACTED/FOLDER` 进入解压后的文件夹
  - 在终端中运行 `export IPEX_LLM_MODEL_SOURCE=modelscope`
  - 运行 `./ollama run deepseek-r1:7b`

> [!Tip]
> 使用 `set IPEX_LLM_MODEL_SOURCE=modelscope` 下载的模型，在执行 `ollama list` 时仍会显示实际的模型 ID，例如：
> ```
> NAME                                                             ID              SIZE      MODIFIED
> modelscope.cn/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF:Q4_K_M    f482d5af6aec    4.7 GB    About a minute ago
> ```
> 除了 `ollama run` 和 `ollama pull`，其他操作中模型应通过其实际 ID 进行识别，例如： `ollama rm modelscope.cn/unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF:Q4_K_M`

### 在 Ollama 中增加上下文长度

默认情况下，Ollama 使用 2048 个 token 的上下文窗口运行模型。也就是说，模型最多能 “记住” 2048 个 token 的上下文。

要增加上下文长度，可以在**启动 Ollama serve 之前**设置环境变量 `IPEX_LLM_NUM_CTX`，步骤如下（如果 Ollama serve 已经在运行，请确保先将其停止）：

- 对于 **Windows** 用户：

  - 打开命令提示符，并通过 `cd /d PATH\TO\EXTRACTED\FOLDER` 命令进入解压后的文件夹
  - 在命令提示符中将 `IPEX_LLM_NUM_CTX` 设置为所需长度，例如：`set IPEX_LLM_NUM_CTX=16384`
  - 通过运行 `start-ollama.bat` 启动 Ollama serve

- 对于 **Linux** 用户：

  - 在终端中输入指令 `cd PATH/TO/EXTRACTED/FOLDER` 进入解压后的文件夹
  - 在终端中将 `IPEX_LLM_NUM_CTX` 设置为所需长度，例如：`export IPEX_LLM_NUM_CTX=16384`
  - 通过运行 `./start-ollama.sh` 启动 Ollama serve

> [!Tip]
> `IPEX_LLM_NUM_CTX` 的优先级高于模型 `Modelfile` 中设置的 `num_ctx`。

### 在多块 GPU 可用时选择特定的 GPU 来运行 Ollama

如果你的机器上有多块 GPU，Ollama 默认会在所有 GPU 上运行。

你可以通过在**启动 Ollama serve 之前**设置环境变量 `ONEAPI_DEVICE_SELECTOR` 来指定在特定的 Intel GPU 上运行 Ollama，步骤如下（如果 Ollama serve 已经在运行，请确保先将其停止）：

- 确认多块 GPU 对应的 id (例如0，1等)。你可以通过在加载任何模型时查看 Ollama serve 的日志来找到它们，例如：

  <div align="center">
    <img src="https://llm-assets.readthedocs.io/en/latest/_images/ollama_portable_multi_gpus.png"  width=80%/>
  </div>

- 对于 **Windows** 用户：

  - 打开命令提示符，并通过 `cd /d PATH\TO\EXTRACTED\FOLDER` 命令进入解压后的文件夹
  - 在命令提示符中设置 `ONEAPI_DEVICE_SELECTOR` 来定义你想使用的 Intel GPU，例如 `set ONEAPI_DEVICE_SELECTOR=level_zero:0`（使用单块 GPU）或 `set ONEAPI_DEVICE_SELECTOR=level_zero:0;level_zero:1`（使用多块 GPU），其中 `0`、`1` 应该替换成你期望的 GPU id
  - 通过运行 `start-ollama.bat` 启动 Ollama serve

- 对于 **Linux** 用户：

  - 在终端中输入指令 `cd PATH/TO/EXTRACTED/FOLDER` 进入解压后的文件夹
  - 在终端中设置 `ONEAPI_DEVICE_SELECTOR` 来定义你想使用的 Intel GPU，例如 `export ONEAPI_DEVICE_SELECTOR=level_zero:0`（使用单块 GPU）或 `export ONEAPI_DEVICE_SELECTOR="level_zero:0;level_zero:1"`（使用多块 GPU），其中 `0`、`1` 应该替换成你期望的 GPU id
  - 通过运行 `./start-ollama.sh` 启动 Ollama serve

### 性能调优

你可以尝试如下设置来进行性能调优：

#### 环境变量 `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS`

环境变量 `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS` 用于控制是否使用 immediate command lists 将任务提交到 GPU。你可以尝试将 `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS` 设为 `1` 或 `0` 以找到最佳性能配置。

你可以通过如下步骤，在**启动 Ollama serve 之前**启用 `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS`（如果 Ollama serve 已经在运行，请确保先将其停止）：

- 对于 **Windows** 用户：

  - 打开命令提示符，并通过 `cd /d PATH\TO\EXTRACTED\FOLDER` 命令进入解压后的文件夹
  - 在命令提示符中设置 `set SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1`
  - 通过运行 `start-ollama.bat` 启动 Ollama serve

- 对于 **Linux** 用户：

  - 在终端中输入指令 `cd PATH/TO/EXTRACTED/FOLDER` 进入解压后的文件夹
  - 在终端中设置 `export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1`
  - 通过运行 `./start-ollama.sh` 启动 Ollama serve

> [!TIP]
> 参考[此处文档](https://www.intel.com/content/www/us/en/developer/articles/guide/level-zero-immediate-command-lists.html)以获取更多 Level Zero Immediate Command Lists 相关信息。

### Ollama v0.5.4 之后新增模型支持

当前的 Ollama Portable Zip 基于 Ollama v0.5.4；此外，以下新模型也已在 Ollama Portable Zip 中得到支持：

| 模型  | 下载（Windows）| 下载（Linux）| 模型链接 |
| - | - | - | - |
| DeepSeek-R1 | `ollama run deepseek-r1` | `./ollama run deepseek-r1` | [deepseek-r1](https://ollama.com/library/deepseek-r1) |
| Openthinker | `ollama run openthinker` | `./ollama run openthinker` | [openthinker](https://ollama.com/library/openthinker) |
| DeepScaleR | `ollama run deepscaler` | `./ollama run deepscaler` | [deepscaler](https://ollama.com/library/deepscaler) |
| Phi-4 | `ollama run phi4` | `./ollama run phi4` | [phi4](https://ollama.com/library/phi4) |
| Dolphin 3.0 | `ollama run dolphin3` | `./ollama run dolphin3` | [dolphin3](https://ollama.com/library/dolphin3) |
| Smallthinker | `ollama run smallthinker` |`./ollama run smallthinker` | [smallthinker](https://ollama.com/library/smallthinker) |
| Granite3.1-Dense |  `ollama run granite3-dense` | `./ollama run granite3-dense` | [granite3.1-dense](https://ollama.com/library/granite3.1-dense) |
| Granite3.1-Moe-3B | `ollama run granite3-moe` | `./ollama run granite3-moe` | [granite3.1-moe](https://ollama.com/library/granite3.1-moe) |
