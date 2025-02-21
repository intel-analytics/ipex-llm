# (Experimental) Example of running GGUF model using llama.cpp C++ API on NPU
In this directory, you will find a simple C++ example on how to run GGUF models on Intel NPUs using `llama.cpp` C++ API. See the table blow for verified models.

## Verified Models

| Model | Model link |
|:--|:--|
| LLaMA 3.2 | [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) |
| DeepSeek-R1 | [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B), [deepseek-ai/DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) |

Please refer to [Quickstart](../../../../../docs/mddocs/Quickstart/npu_quickstart.md#experimental-llamacpp-support) for details about verified platforms.

## 0. Prerequisites
For `ipex-llm` NPU support, please refer to [Quickstart](../../../../../docs/mddocs/Quickstart/npu_quickstart.md#install-prerequisites) for details about the required preparations.

## 1. Install & Runtime Configurations
### 1.1 Installation on Windows
We suggest using conda to manage environment:
```cmd
conda create -n llm python=3.11
conda activate llm

:: for building the example
pip install cmake

:: install ipex-llm with 'npu' option
pip install --pre --upgrade ipex-llm[npu]
```

Please refer to [Quickstart](../../../../../docs/mddocs/Quickstart/npu_quickstart.md#install-prerequisites) for more details about `ipex-llm` installation on Intel NPU.

### 1.2 Runtime Configurations
Please refer to [Quickstart](../../../../../docs/mddocs/Quickstart/npu_quickstart.md#runtime-configurations) for environment variables setting based on your device.

## 2. Build C++ Example `simple`

- You can run below cmake script in cmd to build `simple` by yourself, don't forget to replace below <CONDA_ENV_DIR> with your own path.

```cmd
:: under current directory
:: please replace below conda env dir with your own path
set CONDA_ENV_DIR=C:\Users\arda\miniforge3\envs\llm\Lib\site-packages
mkdir build
cd build
cmake ..
cmake --build . --config Release -j
cd Release
```

- You can also directly use our released `simple.exe` which has the same usage as this example `simple.cpp`

## 3. Run `simple`

With built `simple`, you can run the GGUF model

```cmd
# Run simple text completion
simple.exe -m <gguf_model_path> -n 64 -p "Once upon a time,"
```

> **Note**:
>
> **Warmup on first run**: When running specific GGUF models on NPU for the first time, you might notice delays up to several minutes before the first token is generated. This delay occurs because the blob compilation.
