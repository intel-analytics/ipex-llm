# BigDL-LLM Installation: GPU


## PyTorch 2.1
### Prerequisite

```eval_rst
.. tabs::
   .. tab:: Linux

      BigDL-LLM for GPU supports on Linux with PyTorch 2.1 has been verified on:

      * Intel Arc™ A-Series Graphics
      * Intel Data Center GPU Flex Series
      * Intel Data Center GPU Max Series

      .. note::

         We currently support the Ubuntu 20.04 operating system or later.

      To enable BigDL-LLM for Intel GPUs with PyTorch 2.1, here're several prerequisite steps for tools installation and environment preparation:


      * Step 1: Install Intel GPU Driver version >= stable_736_25_20231031.

        .. seealso::

           Please refer to our `driver installation <https://dgpu-docs.intel.com/driver/installation.html>`_ for general purpose GPU capabilities.

           See `release page <https://dgpu-docs.intel.com/releases/index.html>`_ for latest version.

      * Step 2: Download and install `Intel® oneAPI Base Toolkit <https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html>`_ with version 2024.0. OneMKL and DPC++ compiler are needed, others are optional.

        .. seealso::

           We recommend you to use `this offline package <https://registrationcenter-download.intel.com/akdlm/IRC_NAS/20f4e6a1-6b0b-4752-b8c1-e5eacba10e01/l_BaseKit_p_2024.0.0.49564_offline.sh>`_ to install oneapi.

   .. tab:: Windows

      BigDL-LLM on Windows supports Intel iGPU and dGPU.

      To apply Intel GPU acceleration, there're several prerequisite steps for tools installation and environment preparation:

      * Step 1: Install `Visual Studio 2022 <https://visualstudio.microsoft.com/downloads/>`_ Community Edition and select "Desktop development with C++" workload

      * Step 2: Install or update to latest GPU driver

      * Step 3: Install `Intel® oneAPI Base Toolkit <https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html>`_ 2024.0
```

### Install BigDL-LLM

We recommend using [Conda](https://docs.conda.io/en/latest/miniconda.html) to create a python 3.9 enviroment:

```eval_rst
.. important::

   ``bigdl-llm`` is tested with Python 3.9, which is recommended for best practices.
```

```eval_rst
.. tabs::
   .. tab:: Linux

      .. code-block:: bash

         conda create -n llm python=3.9
         conda activate llm

         pip install --pre --upgrade bigdl-llm[xpu_2.1] -f https://developer.intel.com/ipex-whl-stable-xpu

   .. tab:: Windows

      .. code-block:: cmd

         conda create -n llm python=3.9 libuv
         conda activate llm

         pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu

```

## PyTorch 2.0
```eval_rst
.. note::

   BigDL-LLM for GPU with PyTorch 2.0 supports Ubuntu 20.04 operating system or later.
```

### Prerequisite

BigDL-LLM for GPU supports on Linux with PyTorch 2.0 has been verified on:

* Intel Arc™ A-Series Graphics
* Intel Data Center GPU Flex Series
* Intel Data Center GPU Max Series

To enable BigDL-LLM for Intel GPUs with PyTorch 2.0, here're several prerequisite steps for tools installation and environment preparation:

- Step 1: Install Intel GPU Driver version >= stable_647_21_20230714.
  ```eval_rst
  .. seealso::

     Please refer to our `driver installation <https://dgpu-docs.intel.com/driver/installation.html>`_ for general purpose GPU capabilities.

     See `release page <https://dgpu-docs.intel.com/releases/index.html>`_ for latest version.
  ```
- Step 2: Download and install [Intel® oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html) with version 2023.2.0. OneMKL and DPC++ compiler are needed, others are optional.
  ```eval_rst
  .. seealso::

     We recommend you to use `this offline package <https://registrationcenter-download.intel.com/akdlm/IRC_NAS/992857b9-624c-45de-9701-f6445d845359/l_BaseKit_p_2023.2.0.49397_offline.sh>`_ to install oneapi.
  ```

### Install BigDL-LLM

We recommend using [Conda](https://docs.conda.io/en/latest/miniconda.html) to create a python 3.9 enviroment:

```eval_rst
.. important::

   ``bigdl-llm`` is tested with Python 3.9, which is recommended for best practices.
```

```bash
conda create -n llm python=3.9
conda activate llm

pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
```

## Known issues
### 1. For Linux users, Ubuntu 22.04 and Linux kernel 5.19.0 is prefered

Ubuntu 22.04 and Linux kernel 5.19.0-41-generic is mostly used in our test environment. But default linux kernel of ubuntu 22.04.3 is 6.2.0-35-generic, so we recommonded you to downgrade kernel to 5.19.0-41-generic to archive the best performance.

### 2. Best known configurations

For running a LLM model with BigDL-LLM optimizations, several environment variables are recommended:

```eval_rst
.. tabs::
   .. tab:: Linux

      For Intel Arc™ A-Series Graphics and Intel Data Center GPU Flex Series, we recommend:

      .. code-block:: bash

         # configures OneAPI environment variables
         source /opt/intel/oneapi/setvars.sh

         export USE_XETLA=OFF
         export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
      
      For Intel Data Center GPU Max Series, we recommend:

      .. code-block:: bash

         # configures OneAPI environment variables
         source /opt/intel/oneapi/setvars.sh

         export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
         export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
         export ENABLE_SDP_FUSION=1

      Please note that ``libtcmalloc.so`` can installed by ``conda install -c conda-forge -y gperftools=2.10``



   .. tab:: Windows

      Make sure you are using CMD as PowerShell is not supported:

      .. code-block:: cmd

         # configures OneAPI environment variables
         call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"

         set SYCL_CACHE_PERSISTENT=1

      Please also set the following environment variable for iGPU:

      .. code-block:: cmd

         set BIGDL_LLM_XMX_DISABLED=1

      .. note::

         For the first time that **each** model runs on **a new machine**, it may take around several minutes to compile.

```

### 3. How to install from wheel

If you encounter network issues when installing IPEX, you can also install BigDL-LLM dependencies for Intel XPU from source achieves. First you need to install the target torch/torchvision/ipex versions from downloaded wheels [here](https://developer.intel.com/ipex-whl-stable-xpu) before installing `bigdl-llm`. 

```eval_rst
.. tabs::
   .. tab:: PyTorch 2.1 Linux

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

   .. tab:: PyTorch 2.1 Windows

      .. code-block:: bash

         # get the wheels on Windows system
         wget https://intel-extension-for-pytorch.s3.amazonaws.com/ipex_stable/xpu/torch-2.1.0a0%2Bcxx11.abi-cp39-cp39-win_amd64.whl
         wget https://intel-extension-for-pytorch.s3.amazonaws.com/ipex_stable/xpu/torchvision-0.16.0a0%2Bcxx11.abi-cp39-cp39-win_amd64.whl
         wget https://intel-extension-for-pytorch.s3.amazonaws.com/ipex_stable/xpu/intel_extension_for_pytorch-2.1.10%2Bxpu-cp39-cp39-win_amd64.whl

      Then you may install directly from the wheel archives using following commands:

      .. code-block:: cmd

         # install the packages from the wheels
         pip install torch-2.1.0a0+cxx11.abi-cp39-cp39-win_amd64.whl
         pip install torchvision-0.16.0a0+cxx11.abi-cp39-cp39-win_amd64.whl
         pip install intel_extension_for_pytorch-2.1.10+xpu-cp39-cp39-win_amd64.whl

         # install bigdl-llm for Intel GPU
         pip install --pre --upgrade bigdl-llm[xpu]

   .. tab:: PyTorch 2.0 Linux

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