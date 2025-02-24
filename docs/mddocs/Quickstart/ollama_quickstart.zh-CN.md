# 在 Intel GPU 上使用 IPEX-LLM 运行 Ollama 
<p>
  < <a href='./ollama_quickstart.md'>English</a> | <b>中文</b> >
</p>

[ollama/ollama](https://github.com/ollama/ollama) 是一个轻量级、可扩展的框架，用于在本地机器上构建和运行大型语言模型。现在，借助 [`ipex-llm`](https://github.com/intel-analytics/ipex-llm) 的 C++ 接口作为其加速后端，你可以在 Intel **GPU** *(如配有集成显卡，以及 Arc，Flex 和 Max 等独立显卡的本地 PC)* 上，轻松部署并运行 `ollama`。

> [!Important]
> 现在可使用 [Ollama Portable Zip](./ollama_portable_zip_quickstart.zh-CN.md) 在 Intel GPU 上直接***免安装运行 Ollama***.

> [!NOTE]
> 如果是在 Intel Arc B 系列 GPU 上安装(例如 **B580**)，请参阅本[指南](./bmg_quickstart.md)。

> [!NOTE]
>  `ipex-llm[cpp]` 的最新版本与官方 ollama 的 [v0.5.4](https://github.com/ollama/ollama/releases/tag/v0.5.4) 版本保持一致。
>
> `ipex-llm[cpp]==2.2.0b20250123` 与官方 ollama 的 [v0.5.1](https://github.com/ollama/ollama/releases/tag/v0.5.1) 版本保持一致。

以下是在 Intel Arc GPU 上运行 LLaMA2-7B 的 DEMO 演示。

<table width="100%">
  <tr>
    <td><a href="https://llm-assets.readthedocs.io/en/latest/_images/ollama-linux-arc.mp4"><img src="https://llm-assets.readthedocs.io/en/latest/_images/ollama-linux-arc.png"/></a></td>
  </tr>
  <tr>
    <td align="center">你也可以点击<a href="https://llm-assets.readthedocs.io/en/latest/_images/ollama-linux-arc.mp4">这里</a>观看 DEMO 视频。</td>
  </tr>
</table>

> [!NOTE]
> 从 `ipex-llm[cpp]==2.2.0b20250207` 版本开始，Windows 上 `ipex-llm[cpp]` 依赖的 oneAPI 版本已从 `2024.2.1` 更新到 `2025.0.1`。
> 
> 如果要将 `ipex-llm[cpp]` 升级到 `2.2.0b20250207` 或更高版本，在Windows环境下，你需要新建一个干净的 conda 环境来安装新版本。如果直接在旧的 conda 环境中卸载旧版本并升级，可能会遇到 `找不到 sycl8.dll` 的错误。

## 目录
- [安装 IPEX-LLM 来使用 Ollama](./ollama_quickstart.zh-CN.md#1-安装-ipex-llm-来使用-Ollama)
- [初始化 Ollama](./ollama_quickstart.zh-CN.md#2-初始化-ollama)
- [运行 Ollama 服务](./ollama_quickstart.zh-CN.md#3-运行-ollama-服务)
- [拉模型](./ollama_quickstart.zh-CN.md#4-拉模型)
- [使用 Ollama](./ollama_quickstart.zh-CN.md#5-使用-ollama)

## 快速入门

### 1. 安装 IPEX-LLM 来使用 Ollama

IPEX-LLM 现在已支持在 Linux 和 Windows 系统上运行 `Ollama`。

请仔细参阅网页[在 Intel GPU 中使用 IPEX-LLM 运行 llama.cpp 指南](./llama_cpp_quickstart.zh-CN.md)，首先按照 [系统环境准备](./llama_cpp_quickstart.zh-CN.md#0-系统环境准备) 步骤进行设置，再参考 [llama.cpp 中安装 IPEX-LLM](./llama_cpp_quickstart.zh-CN.md#1-为-llamacpp-安装-IPEX-LLM) 步骤用 Ollama 可执行文件安装 IPEX-LLM。 

**完成上述步骤后，你应该已经创建了一个名为 `llm-cpp` 的新 conda 环境。该 conda 环境将用于在 Intel GPU 上使用 IPEX-LLM 运行 ollama。**

### 2. 初始化 Ollama

然后，运行下列命令进行 `llm-cpp` conda 环境激活和初始化 Ollama。在你的当前目录中会出现一个指向 `ollama` 的符号链接。

- **Linux 用户**:
  
  ```bash
  conda activate llm-cpp
  init-ollama
  ```

- **Windows 用户**:

  请**在 Miniforge Prompt 中使用管理员权限** 运行以下命令。

  ```cmd
  conda activate llm-cpp
  init-ollama.bat
  ```

> [!NOTE]
> 如果你已经安装了更高版本的 `ipex-llm[cpp]`，并希望同时升级 ollama 可执行文件，请先删除目录下旧文件，然后使用 `init-ollama`（Linux）或 `init-ollama.bat`（Windows）重新初始化。

**现在，你可以按照 ollama 的官方用法来执行 ollama 的命令了。**

### 3. 运行 Ollama 服务

请根据你的操作系统选择以下对应的步骤启动 Ollama 服务:

- **Linux 用户**:

  ```bash
  export OLLAMA_NUM_GPU=999
  export no_proxy=localhost,127.0.0.1
  export ZES_ENABLE_SYSMAN=1
  
  source /opt/intel/oneapi/setvars.sh
  export SYCL_CACHE_PERSISTENT=1
  # [optional] under most circumstances, the following environment variable may improve performance, but sometimes this may also cause performance degradation
  export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
  # [optional] if you want to run on single GPU, use below command to limit GPU may improve performance
  export ONEAPI_DEVICE_SELECTOR=level_zero:0

  ./ollama serve
  ```

- **Windows 用户**:

  请在 Miniforge Prompt 中运行以下命令。

  ```cmd
  set OLLAMA_NUM_GPU=999
  set no_proxy=localhost,127.0.0.1
  set ZES_ENABLE_SYSMAN=1
  set SYCL_CACHE_PERSISTENT=1
  rem under most circumstances, the following environment variable may improve performance, but sometimes this may also cause performance degradation
  set SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1

  ollama serve
  ```

> [!NOTE]
> 请设置环境变量 `OLLAMA_NUM_GPU` 为 `999` 确保模型的所有层都在 Intel GPU 上运行，否则某些层可能会在 CPU 上运行。

> [!NOTE]
> 为了允许服务器接受来自所有 IP 地址的连接，请使用 `OLLAMA_HOST=0.0.0.0 ./ollama serve` 代替仅使用 `./ollama serve`。

> [!TIP]
> 如果你的设备配备了多个 GPU，而你只想在其中一个 GPU 上运行 ollama 时，就需要设置环境变量 `ONEAPI_DEVICE_SELECTOR=level_zero:[gpu_id]`，其中 `[gpu_id]` 是指定运行 ollama 的 GPU 设备 ID。相关详情请参阅[多 GPU 选择指南](../Overview/KeyFeatures/multi_gpus_selection.md#2-oneapi-device-selector)。

> [!NOTE]
> 环境变量 `SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS` 用于控制是否使用*即时命令列表*将任务提交到 GPU。启动此变量通常可以提高性能，但也有例外情况。因此，建议你在启用和禁用该环境变量的情况下进行测试，以找到最佳的性能设置。更多相关细节请参考[此处文档](https://www.intel.com/content/www/us/en/developer/articles/guide/level-zero-immediate-command-lists.html)。

控制台将显示类似以下内容的消息:

<a href="https://llm-assets.readthedocs.io/en/latest/_images/ollama_serve.png" target="_blank">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/ollama_serve.png" width=100%; />
</a>

### 4. 拉模型
保持 Ollama 服务开启并打开另一个终端，然后使用 `./ollama pull <model_name>`（Linux）或 `ollama.exe pull <model_name>`（Windows）自动拉一个模型。例如，`dolphin-phi:latest`:

<a href="https://llm-assets.readthedocs.io/en/latest/_images/ollama_pull.png" target="_blank">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/ollama_pull.png" width=100%; />
</a>

### 5. 使用 Ollama

#### 使用 Curl 

使用 `curl` 是验证 API 服务和模型最简单的方法。在终端中执行以下命令。你可以**将 <model_name> 替换成你使用的模型**，例如，`dolphin-phi`。

- **Linux 用户**:
  
   ```bash
   curl http://localhost:11434/api/generate -d '
   { 
      "model": "<model_name>", 
      "prompt": "Why is the sky blue?", 
      "stream": false
   }'
   ```

- **Windows 用户**:

  请在 Miniforge Prompt 中运行下列命令。

   ```cmd
   curl http://localhost:11434/api/generate -d "
   {
      \"model\": \"<model_name>\",
      \"prompt\": \"Why is the sky blue?\",
      \"stream\": false
   }"
   ```

#### 使用 Ollama 运行 GGUF models

Ollama 支持在 Modelfile 中导入 GGUF 模型，例如，假设你已经从 [Mistral-7B-Instruct-v0.1-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/tree/main) 下载了 `mistral-7b-instruct-v0.1.Q4_K_M.gguf`，那么你可以创建一个名为 `Modelfile` 的文件:

```bash
FROM ./mistral-7b-instruct-v0.1.Q4_K_M.gguf
TEMPLATE [INST] {{ .Prompt }} [/INST]
PARAMETER num_predict 64
```

然后，你可以在 Ollama 中通过 `ollama create example -f Modelfile` 创建模型，并使用 `ollama run` 直接在控制台运行该模型。

- **Linux 用户**:
  
  ```bash
  export no_proxy=localhost,127.0.0.1
  source /opt/intel/oneapi/setvars.sh
  ./ollama create example -f Modelfile
  ./ollama run example
  ```

- **Windows 用户**:

  请在 Miniforge Prompt 中运行下列命令。

  ```cmd
  set no_proxy=localhost,127.0.0.1
  ollama create example -f Modelfile
  ollama run example
  ```

使用 `ollama run example` 与模型交互的示例过程，如下所示:

<a href="https://llm-assets.readthedocs.io/en/latest/_images/ollama_gguf_demo_image.png" target="_blank">
  <img src="https://llm-assets.readthedocs.io/en/latest/_images/ollama_gguf_demo_image.png" width=100%; />
</a>

### 故障排除
#### 1. 无法运行初始化脚本
如果你无法运行 `init-ollama.bat`，请确保你已经在 conda 环境中安装了 `ipex-llm[cpp]`。如果你已安装，请检查你是否已激活正确的 conda 环境。此外，如果你使用的是 Windows，请确保你已在提示终端中以管理员权限运行该脚本。

#### 2. 为什么模型总是几分钟后再次加载
Ollama 默认每 5 分钟从 GPU 内存卸载一次模型。针对 ollama 的最新版本，你可以设置 `OLLAMA_KEEP_ALIVE=-1` 来将模型保持在显存上。请参阅此问题：https://github.com/intel-analytics/ipex-llm/issues/11608

#### 3. 执行 `ollama serve`时报 `exit status 0xc0000135` 错误
执行 `ollama serve`时，如果你在 Windows 中遇到 `llama runner process has terminated: exit status 0xc0000135` 或者在 Linux 中遇到 `ollama_llama_server: error while loading shared libraries: libmkl_core.so.2: cannot open shared object file`，这很可能是由于缺少 sycl 依赖导致的。请检查：

1. Windows：是否已经安装了 conda 并激活了正确的 conda 环境，环境中是否已经使用 pip 安装了 oneAPI 依赖项
2. Linux：是否已经在运行 ollama 命令前执行了 `source /opt/intel/oneapi/setvars.sh`。执行此 source 命令只在当前会话有效。

#### 4. 初始模型加载阶段程序挂起
在 Windows 中首次启动 `ollama serve` 时，可能会在模型加载阶段卡住。如果你在首次运行时发现程序长时间挂起，可以手动在服务器端输入空格或其他字符以确保程序正在运行。

#### 5. 如何区分社区版 Ollama 和 IPEX-LLM 版 Ollama
在社区版 Ollama 的服务器日志中，你可能会看到 `source=payload_common.go:139 msg="Dynamic LLM libraries [rocm_v60000 cpu_avx2 cuda_v11 cpu cpu_avx]"`。而在 IPEX-LLM 版 Ollama 的服务器日志中，你应该仅看到 `source=common.go:49 msg="Dynamic LLM libraries" runners=[ipex_llm]`。

#### 6. 当询问多个不同的问题或上下文很长时，Ollama 会挂起
如果你在询问多个不同问题或上下文很长时，发现 ollama 挂起，并且在服务器日志中看到 `update_slots : failed to free spaces in the KV cache`，这可能是因为 LLM 上下文大于默认 `n_ctx` 值导致的，你可以尝试增加 `n_ctx` 值后重试。

#### 7. `signal: bus error (core dumped)` 错误
如果你遇到此错误，请先检查你的 Linux 内核版本。较高版本的内核（例如 6.15）可能会导致此问题。你也可以参考[此问题](https://github.com/intel-analytics/ipex-llm/issues/10955)来查看是否有帮助。

#### 8. 通过设置`OLLAMA_NUM_PARALLEL=1`节省GPU内存
如果你的GPU内存较小，可以通过在运行`ollama serve`前运行`set OLLAMA_NUM_PARALLEL=1`（Windows）或`export OLLAMA_NUM_PARALLEL=1`（Linux）来减少内存使用。Ollama默认使用的`OLLAMA_NUM_PARALLEL`为4。

#### 9. 执行 `ollama serve`时报 `cannot open shared object file` 错误
执行 `ollama serve` 或 `ollama run <model_name>` 时，如果你在 Linux 上遇到 `./ollama: error while loading shared libraries: libsvml.so: cannot open shared object file: No such file or directory`，或者在 Windows 上执行 `ollama serve` 和 `ollama run <model_name>` 时没有反应，这很可能是由于缺少 sycl 依赖导致的。请检查：

1. Windows：是否已经安装了 conda 并激活了正确的 conda 环境，环境中是否已经使用 pip 安装了 oneAPI 依赖项
2. Linux：是否已经在运行 `./ollama serve` 和 `./ollama run <model_name>` 命令前都执行了 `source /opt/intel/oneapi/setvars.sh`。执行此 source 命令只在当前会话有效。

#### 10. ollama serve 没有输出或响应
当你启动 `ollama serve` 并运行 `ollama run <model_name>` 时，`ollama serve` 没有响应。这可能是由于你的设备上存在多个 ollama 进程导致的。请按照以下命令操作：

在 Linux 上，你可以运行 `systemctl stop ollama` 来停止所有的 ollama 进程，然后在当前目录重新执行 `ollama serve`。
在 Windows 上，你可以运行 `set OLLAMA_HOST=0.0.0.0` 以确保 ollama 命令通过当前的 `ollama serve` 上运行。
