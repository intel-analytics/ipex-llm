# BigDL-LLM Installation: GPU

## Windows

### Prerequisites

BigDL-LLM on Windows supports Intel iGPU and dGPU.

To apply Intel GPU acceleration, there're several prerequisite steps for tools installation and environment preparation:

* Step 1: Install `Visual Studio 2022 <https://visualstudio.microsoft.com/downloads/>`_ Community Edition and select "Desktop development with C++" workload

* Step 2: Install or update to latest GPU driver

* Step 3: Install `Intel® oneAPI Base Toolkit <https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html>`_ 2024.0

### Install BigDL-LLM From PyPi

We recommend using [Conda](https://docs.conda.io/en/latest/miniconda.html) to create a python 3.9 enviroment:

```eval_rst
.. important::

   ``bigdl-llm`` is tested with Python 3.9, which is recommended for best practices.
```

The folloing ways to install `bigdl-llm` is the following commands:

```
conda create -n llm python=3.9 libuv
conda activate llm

pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
```

### Install BigDL-LLM From Wheel

If you encounter network issues when installing IPEX, you can also install BigDL-LLM dependencies for Intel XPU from source achieves. First you need to download and install torch/torchvision/ipex from wheels listed here before installing `bigdl-llm`.

Download the wheels on Windows system:

```
wget https://intel-extension-for-pytorch.s3.amazonaws.com/ipex_stable/xpu/torch-2.1.0a0%2Bcxx11.abi-cp39-cp39-win_amd64.whl
wget https://intel-extension-for-pytorch.s3.amazonaws.com/ipex_stable/xpu/torchvision-0.16.0a0%2Bcxx11.abi-cp39-cp39-win_amd64.whl
wget https://intel-extension-for-pytorch.s3.amazonaws.com/ipex_stable/xpu/intel_extension_for_pytorch-2.1.10%2Bxpu-cp39-cp39-win_amd64.whl
```

Then you may install directly from the wheel archives then install `bigdl-llm` using following commands:

```
pip install torch-2.1.0a0+cxx11.abi-cp39-cp39-win_amd64.whl
pip install torchvision-0.16.0a0+cxx11.abi-cp39-cp39-win_amd64.whl
pip install intel_extension_for_pytorch-2.1.10+xpu-cp39-cp39-win_amd64.whl

pip install --pre --upgrade bigdl-llm[xpu]
```

### Runtime Configuration

To use GPU acceleration on Windows, several environment variables are required or recommended before running a GPU example.

Make sure you are using CMD as PowerShell is not supported:

```
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"

set SYCL_CACHE_PERSISTENT=1
```

Please also set the following environment variable for iGPU:

```
set BIGDL_LLM_XMX_DISABLED=1
```

```eval_rst
.. note::

   For the first time that **each** model runs on **a new machine**, it may take around several minutes to compile.
```

### Known Issue

todo

## Linux

### Prerequisites

```eval_rst
.. tabs::
   .. tab:: PyTorch 2.1

      BigDL-LLM for GPU supports on Linux with PyTorch 2.1 has been verified on:

      * Intel Arc™ A-Series Graphics
      * Intel Data Center GPU Flex Series
      * Intel Data Center GPU Max Series

      .. note::

         We currently support the Ubuntu 20.04 operating system or later.

      To enable BigDL-LLM for Intel GPUs with PyTorch 2.1, here're several prerequisite steps for tools installation and environment preparation:


      * Step 1: Install Intel GPU Driver version >= stable_775_20_20231219. Highly recommend installing the latest version of intel-i915-dkms using apt.

        .. seealso::

           Please refer to our `driver installation <https://dgpu-docs.intel.com/driver/installation.html>`_ for general purpose GPU capabilities.

           See `release page <https://dgpu-docs.intel.com/releases/index.html>`_ for latest version.

      * Step 2: Download and install `Intel® oneAPI Base Toolkit <https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html>`_ with version 2024.0. Onednn, OneMKL and DPC++ compiler are needed, others are optional.

        .. seealso::

           We recommend you to use `this offline package <https://registrationcenter-download.intel.com/akdlm/IRC_NAS/20f4e6a1-6b0b-4752-b8c1-e5eacba10e01/l_BaseKit_p_2024.0.0.49564_offline.sh>`_ to install oneapi.

   .. tab:: Pytorch 2.0

      BigDL-LLM for GPU supports on Linux with PyTorch 2.0 has been verified on:

      * Intel Arc™ A-Series Graphics
      * Intel Data Center GPU Flex Series
      * Intel Data Center GPU Max Series

      .. note::

         We currently support the Ubuntu 20.04 operating system or later.

      To enable BigDL-LLM for Intel GPUs with PyTorch 2.0, here're several prerequisite steps for tools installation and environment preparation:


      * Step 1: Install Intel GPU Driver version >= stable_775_20_20231219. Highly recommend installing the latest version of intel-i915-dkms using apt.

        .. seealso::

           Please refer to our `driver installation <https://dgpu-docs.intel.com/driver/installation.html>`_ for general purpose GPU capabilities.

           See `release page <https://dgpu-docs.intel.com/releases/index.html>`_ for latest version.

      * Step 2: Download and install `Intel® oneAPI Base Toolkit <https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html>`_ with version 2023.2.0. Onednn, OneMKL and DPC++ compiler are needed, others are optional.

        .. seealso::

           We recommend you to use `this offline package <https://registrationcenter-download.intel. com/akdlm/IRC_NAS/992857b9-624c-45de-9701-f6445d845359/l_BaseKit_p_2023.2.0.49397_offline. sh>`_ to install oneapi.

```

### Install BigDL-LLM From PyPi

We recommend using [Conda](https://docs.conda.io/en/latest/miniconda.html) to create a python 3.9 enviroment:

```eval_rst
.. important::

   ``bigdl-llm`` is tested with Python 3.9, which is recommended for best practices.
```

```eval_rst
.. tabs::
   .. tab:: Pytorch 2.1

      .. code-block:: bash

         conda create -n llm python=3.9
         conda activate llm

         pip install --pre --upgrade bigdl-llm[xpu_2.1] -f https://developer.intel.com/ipex-whl-stable-xpu

   .. tab:: Pytorch 2.0

      .. code-block:: cmd

         conda create -n llm python=3.9
         conda activate llm

         pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu

```

### Install BigDL-LLM From Wheel

If you encounter network issues when installing IPEX, you can also install BigDL-LLM dependencies for Intel XPU from source achieves. First you need to download and install torch/torchvision/ipex from wheels listed here before installing `bigdl-llm`.

```eval_rst
.. tabs::
   .. tab:: PyTorch 2.1

      .. code-block:: bash

         # get the wheels on Linux system for IPEX 2.1.10+xpu
         wget https://intel-extension-for-pytorch.s3.amazonaws.com/ipex_stable/xpu/torch-2.1.0a0%2Bcxx11.abi-cp39-cp39-linux_x86_64.whl
         wget https://intel-extension-for-pytorch.s3.amazonaws.com/ipex_stable/xpu/torchvision-0.16.0a0%2Bcxx11.abi-cp39-cp39-linux_x86_64.whl
         wget https://intel-extension-for-pytorch.s3.amazonaws.com/ipex_stable/xpu/intel_extension_for_pytorch-2.1.10%2Bxpu-cp39-cp39-linux_x86_64.whl

      Then you may install directly from the wheel archives using following commands:

      .. code-block:: bash

         # install the packages from the wheels
         pip install torch-2.1.0a0+cxx11.abi-cp39-cp39-linux_x86_64.whl
         pip install torchvision-0.16.0a0+cxx11.abi-cp39-cp39-linux_x86_64.whl
         pip install intel_extension_for_pytorch-2.1.10+xpu-cp39-cp39-linux_x86_64.whl

         # install bigdl-llm for Intel GPU
         pip install --pre --upgrade bigdl-llm[xpu_2.1]

   .. tab:: PyTorch 2.0

      .. code-block:: bash

         # get the wheels on Linux system for IPEX 2.0.110+xpu
         wget https://intel-extension-for-pytorch.s3.amazonaws.com/ipex_stable/xpu/torch-2.0.1a0%2Bcxx11.abi-cp39-cp39-linux_x86_64.whl
         wget https://intel-extension-for-pytorch.s3.amazonaws.com/ipex_stable/xpu/torchvision-0.15.2a0%2Bcxx11.abi-cp39-cp39-linux_x86_64.whl
         wget https://intel-extension-for-pytorch.s3.amazonaws.com/ipex_stable/xpu/intel_extension_for_pytorch-2.0.110%2Bxpu-cp39-cp39-linux_x86_64.whl

      Then you may install directly from the wheel archives using following commands:

      .. code-block:: bash

         # install the packages from the wheels
         pip install torch-2.0.1a0+cxx11.abi-cp39-cp39-linux_x86_64.whl
         pip install torchvision-0.15.2a0+cxx11.abi-cp39-cp39-linux_x86_64.whl
         pip install intel_extension_for_pytorch-2.0.110+xpu-cp39-cp39-linux_x86_64.whl

         # install bigdl-llm for Intel GPU
         pip install --pre --upgrade bigdl-llm[xpu]

```

### Runtime Configuration

To use GPU acceleration on Linux, several environment variables are required or recommended before running a GPU example.

```eval_rst
.. tabs::
   .. tab:: Intel Arc™ A-Series Graphics and Intel Data Center GPU Flex Series

      For Intel Arc™ A-Series Graphics and Intel Data Center GPU Flex Series, we recommend:

      .. code-block:: bash

         # configures OneAPI environment variables
         source /opt/intel/oneapi/setvars.sh

         export USE_XETLA=OFF
         export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1

   .. tab:: Intel Data Center GPU Max Series

      For Intel Data Center GPU Max Series, we recommend:

      .. code-block:: bash

         # configures OneAPI environment variables
         source /opt/intel/oneapi/setvars.sh

         export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
         export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
         export ENABLE_SDP_FUSION=1

      Please note that ``libtcmalloc.so`` can installed by ``conda install -c conda-forge -y gperftools=2.10``

```

### Known issues

#### 1. Ubuntu 22.04 and Linux kernel 6.2.0 may cause performance bad (driver version < stable_775_20_20231219)

For driver version < stable_775_20_20231219, the performance on Linux kernel 6.2.0 is worse than Linux kernel 5.19.0. You can use `sudo apt update && sudo apt install -y intel-i915-dkms intel-fw-gpu` to install the latest driver to solve this issue (need reboot OS).

Tips: You can use `sudo apt list --installed | grep intel-i915-dkms` to check your intel-i915-dkms's version, the version should be latest and >= `1.23.9.11.231003.15+i19-1`.

#### 2. Driver installation meet unmet dependencies: intel-i915-dkms

The last apt install command of the driver installation may get following error:

```
The following packages have unmet dependencies:
 intel-i915-dkms : Conflicts: intel-platform-cse-dkms
                   Conflicts: intel-platform-vsec-dkms
```

You can use `sudo apt install -y intel-i915-dkms intel-fw-gpu` to instead. As the intel-platform-cse-dkms and intel-platform-vsec-dkms are already provided by intel-i915-dkms.
