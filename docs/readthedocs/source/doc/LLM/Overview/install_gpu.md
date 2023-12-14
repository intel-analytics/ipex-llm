# BigDL-LLM Installation: GPU

## Quick Installation

Install BigDL-LLM for GPU supports using pip through:

```bash
pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
```

Please refer to [Environment Setup](#environment-setup) for more information.

```eval_rst
.. note::

   For Linux users, the above command will install ``intel_extension_for_pytorch==2.0.110+xpu`` as default. You could also use

   .. code-block:: bash

      pip install --pre --upgrade bigdl-llm[xpu_2.1] -f https://developer.intel.com/ipex-whl-stable-xpu

   to install ``bigdl-llm`` with ``intel_extension_for_pytorch==2.1.10+xpu``.

   For Windows users, the above command will install ``intel_extension_for_pytorch==2.1.10+xpu`` as default.

.. important::

   Currently ``https://developer.intel.com/ipex-whl-stable-xpu`` is  the only achievable source for ``-f`` option since IPEX 2.0.110+xpu or IPEX 2.1.10+xpu and corresponding torch versions are not released on pypi.


.. important::

   ``bigdl-llm`` is tested with Python 3.9, which is recommended for best practices.
```

## Recommended Requirements

```eval_rst
.. tabs::
   .. tab:: Linux

      BigDL-LLM for GPU supports on Linux has been verified on:

      * Intel Arc™ A-Series Graphics
      * Intel Data Center GPU Flex Series
      * Intel Data Center GPU Max Series

      .. note::

         We currently supoort the Ubuntu 20.04 operating system or later.

      To apply Intel GPU acceleration, there're several steps for tools installation and environment preparation:

      * Step 1, Ubuntu 22.04 and Linux kernel 5.19.0 is prefered.

        .. note::

           Ubuntu 22.04 and Linux kernel 5.19.0-41-generic is mostly used in our test environment. But default linux kernel of ubuntu 22.04.3 is 6.2.0-35-generic, so we recommonded you to downgrade kernel to 5.19.0-41-generic to archive the best performance.

      * Step 2, please refer to our [driver installation](https://dgpu-docs.intel.com/driver/installation.html) for general purpose GPU capabilities.
        
        .. note::

            * IPEX 2.0.110+xpu requires Intel GPU Driver version >= stable_647_21_20230714
            * IPEX 2.1.10+xpu requires Intel GPU Driver version >= stable_736_25_20231031
            
            See `release page <https://dgpu-docs.intel.com/releases/index.html>`_ for latest version.

      * Step 3, you also need to download and install [Intel® oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html). OneMKL and DPC++ compiler are needed, others are optional.
         
        .. note::

           * IPEX 2.0.110+xpu requires Intel® oneAPI Base Toolkit's version == 2023.2.0. We recommand you to use `this offline package <https://registrationcenter-download.intel.com/akdlm/IRC_NAS/992857b9-624c-45de-9701-f6445d845359/l_BaseKit_p_2023.2.0.49397_offline.sh>`_ to install oneapi.
           * IPEX 2.1.10+xpu requires Intel® oneAPI Base Toolkit's version == 2024.0. We recommand you to use `this offline package <https://registrationcenter-download.intel.com/akdlm/IRC_NAS/20f4e6a1-6b0b-4752-b8c1-e5eacba10e01/l_BaseKit_p_2024.0.0.49564_offline.sh>`_ to install oneapi.

   .. tab:: Windows

      BigDL-LLM on Linux supports Intel iGPU and dGPU.

      To apply Intel GPU acceleration, there're several steps for tools installation and environment preparation:

      * Step 1: Install `Visual Studio 2022 <https://visualstudio.microsoft.com/downloads/>`_ Community Edition and select "Desktop development with C++" workload
      * Step 2: Install or update to latest GPU driver
      * Step 3: Install `Intel® oneAPI Base Toolkit <https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html>`_ 2024.0

```


## Environment Setup

For optimal performance with LLM models using BigDL-LLM optimizations on Intel GPUs, here are some best practices for setting up environment:

First we recommend using [Conda](https://docs.conda.io/en/latest/miniconda.html) to create a python 3.9 enviroment:

```bash
conda create -n llm python=3.9
conda activate llm

pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu # install bigdl-llm for GPU
```

If you encounter network issues when installing ipex, you can refer to  [Installing bigdl-llm[xpu] dependencies from Wheels](#installing-bigdl-llm[xpu]-dependencies-from-wheels) as an alternative method.

Then for running a LLM model with BigDL-LLM optimizations, several environment variables are recommended:

```eval_rst
.. tabs::
   .. tab:: Linux

      .. code-block:: bash

         # configures OneAPI environment variables
         source /opt/intel/oneapi/setvars.sh

         export USE_XETLA=OFF
         export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1

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

         For the first time that **each** model runs on **a new machine**, it may take around 10 min to compile.

```

## Installing bigdl-llm[xpu] dependencies from Wheels

You can also install BigDL-LLM dependencies for Intel XPU from source achieves. First you need to install the target torch/torchvision/ipex versions from downloaded whls [here](https://developer.intel.com/ipex-whl-stable-xpu) before installing bigdl-llm. 


```eval_rst
.. tabs::
   .. tab:: Linux IPEX 2.0

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

   .. tab:: Linux IPEX 2.1

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

   .. tab:: Windows

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

```