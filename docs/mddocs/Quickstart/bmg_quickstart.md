# Install and Use IPEX-LLM on Intel Arc B-Series GPU (code-named Battlemage)

This guide demonstrates how to install and use IPEX-LLM on the Intel Arc B-Series GPU (such as **B580**). 

> [!NOTE]  
> Ensure your GPU driver and software environment meet the prerequisites before proceeding.

---

## Table of Contents

1. [Linux](#1-linux)  
   1.1 [Install Prerequisites](#11-install-prerequisites)  
   1.2 [Install IPEX-LLM](#for-pytorch-and-huggingface) (for PyTorch and HuggingFace)  
   1.3 [Install IPEX-LLM](#for-llamacpp-and-ollama) (for llama.cpp and Ollama)  
2. [Windows](#2-windows)   
   2.1 [Install Prerequisites](#21-install-prerequisites)  
   2.2 [Install IPEX-LLM](#for-pytorch-and-huggingface-1) (for PyTorch and HuggingFace)  
   2.3 [Install IPEX-LLM](#for-llamacpp-and-ollama-1) (for llama.cpp and Ollama)  
3. [Use Cases](#3-use-cases)  
   3.1 [PyTorch](#31-pytorch)  
   3.2 [Ollama](#32-ollama)  
   3.3 [llama.cpp](#33-llamacpp)  
   3.4 [vLLM](#34-vllm)
4. [Troubleshooting](#4-troubleshooting)  
   4.1 [RuntimeError: could not create an engine](#41-runtimeerror-could-not-create-an-engine)
---

## 1. Linux

### 1.1 Install Prerequisites

> [!NOTE]
> Ensure that **Resizable BAR** is enabled in your system's BIOS before proceeding. This is essential for optimal GPU performance and to avoid potential issues such as `Bus error (core dumped)`. For detailed steps, please refer to the official guidance [here](https://www.intel.com/content/www/us/en/support/articles/000090831/graphics.html).

We recommend using Ubuntu 24.10 and kernel version 6.11 or above, as support for Battle Mage has been backported from kernel version 6.12 to version 6.11, which is included in Ubuntu 24.10, according to the official documentation [here](https://dgpu-docs.intel.com/driver/client/overview.html#installing-client-gpus-on-ubuntu-desktop-24-10). However, since this version of Ubuntu does not include the latest compute and media-related packages, we offer the intel-graphics Personal Package Archive (PPA). The PPA provides early access to newer packages, along with additional tools and features such as EU debugging.

Use the following commands to install the intel-graphics PPA and the necessary compute and media packages:

```bash
sudo apt-get update
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:kobuk-team/intel-graphics
sudo apt-get install -y libze-intel-gpu1 libze1 intel-ocloc intel-opencl-icd clinfo intel-gsc intel-media-va-driver-non-free libmfx1 libmfx-gen1 libvpl2 libvpl-tools libva-glx2 va-driver-all vainfo
sudo apt-get install -y intel-level-zero-gpu-raytracing  # Optional: Hardware ray tracing support
```

#### Setup Python Environment

Download and install Miniforge:
```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
source ~/.bashrc
```

Create and activate a Python environment:
```bash
conda create -n llm python=3.11
conda activate llm
```

---

### 1.2 Install IPEX-LLM

With the `llm` environment active, install the appropriate `ipex-llm` package based on your use case:

#### For PyTorch and HuggingFace:
Install the `ipex-llm[xpu-arc]` package. Choose either the US or CN website for `extra-index-url`:

- For **US**:
  ```bash
  pip install --pre --upgrade ipex-llm[xpu-arc] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
  ```

- For **CN**:
  ```bash
  pip install --pre --upgrade ipex-llm[xpu-arc] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/cn/
  ```

#### For llama.cpp and Ollama:
Install the `ipex-llm[cpp]` package.

  ```bash
  pip install --pre --upgrade ipex-llm[cpp] 
  ```

> [!NOTE]  
> If you encounter network issues during installation, refer to the [troubleshooting guide](../Overview/install_gpu.md#install-ipex-llm-from-wheel-1) for alternative steps.

---

## 2. Windows

### 2.1 Install Prerequisites

#### Update GPU Driver

If your driver version is lower than `32.0.101.6449/32.0.101.101.6256`, update it from the [Intel download page](https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html). After installation, reboot the system.

---

#### Setup Python Environment

Download and install Miniforge for Windows from the [official page](https://conda-forge.org/download/). After installation, create and activate a Python environment:

```cmd
conda create -n llm python=3.11 libuv
conda activate llm
```

---

### 2.2 Install IPEX-LLM

With the `llm` environment active, install the appropriate `ipex-llm` package based on your use case:

#### For PyTorch and HuggingFace:
Install the `ipex-llm[xpu-arc]` package. Choose either the US or CN website for `extra-index-url`:

- For **US**:
  ```cmd
  pip install --pre --upgrade ipex-llm[xpu-arc] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
  ```

- For **CN**:
  ```cmd
  pip install --pre --upgrade ipex-llm[xpu-arc] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/cn/
  ```

#### For llama.cpp and Ollama:
Install the `ipex-llm[cpp]` package. 

  ```cmd
  pip install --pre --upgrade ipex-llm[cpp] 
  ```

> [!NOTE]  
> If you encounter network issues while installing IPEX, refer to [this guide](../Overview/install_gpu.md#install-ipex-llm-from-wheel) for troubleshooting advice.

---


## 3. Use Cases

### 3.1 PyTorch

Run a Quick PyTorch Example:

1. Activate the environment:  
   ```bash
   conda activate llm  # On Windows, use 'cmd'
   ```
2. Run the code:  
   ```python
   import torch
   from ipex_llm.transformers import AutoModelForCausalLM

   tensor_1 = torch.randn(1, 1, 40, 128).to('xpu')
   tensor_2 = torch.randn(1, 1, 128, 40).to('xpu')
   print(torch.matmul(tensor_1, tensor_2).size())
   ```
3. Expected Output:  
   ```
   torch.Size([1, 1, 40, 40])
   ```

For benchmarks and performance measurement, refer to the [Benchmark Quickstart guide](https://github.com/intel-analytics/ipex-llm/blob/main/docs/mddocs/Quickstart/benchmark_quickstart.md).

---

### 3.2 Ollama

To integrate and run with **Ollama**, follow the [Ollama Quickstart guide](https://github.com/intel-analytics/ipex-llm/blob/main/docs/mddocs/Quickstart/ollama_quickstart.md).

### 3.3 llama.cpp

For instructions on how to run **llama.cpp** with IPEX-LLM, refer to the [llama.cpp Quickstart guide](https://github.com/intel-analytics/ipex-llm/blob/main/docs/mddocs/Quickstart/llama_cpp_quickstart.md).

### 3.4 vLLM

To set up and run **vLLM**, follow the [vLLM Quickstart guide](https://github.com/intel-analytics/ipex-llm/blob/main/docs/mddocs/Quickstart/vLLM_quickstart.md).

## 4. Troubleshooting

### 4.1 RuntimeError: could not create an engine

![image](https://github.com/user-attachments/assets/757f0704-9240-46d0-bceb-661fecc96182)

If you encounter a `RuntimeError` like the output shown above while working on Linux after running `conda deactivate` and then reactivating your environment using `conda activate env`, the issue is likely caused by the `OCL_ICD_VENDORS` environment variable. 

To fix this on Linux, run the following command:

```bash
unset OCL_ICD_VENDORS
```

This will remove the conflicting environment variable and allow your program to function correctly.

**Note:** This issue only occurs on Linux systems. It does not affect Windows environments.

