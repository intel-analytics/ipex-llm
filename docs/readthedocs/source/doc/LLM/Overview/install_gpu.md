# IPEX-LLM Installation: GPU

## Windows

### Prerequisites

IPEX-LLM on Windows supports Intel iGPU and dGPU.

```eval_rst
.. important::

    IPEX-LLM on Windows only supports PyTorch 2.1.
```

To apply Intel GPU acceleration, there're several prerequisite steps for tools installation and environment preparation:

* Step 1: Install [Visual Studio 2022](https://visualstudio.microsoft.com/downloads/) Community Edition and select "Desktop development with C++" workload, like [this](https://learn.microsoft.com/en-us/cpp/build/vscpp-step-0-installation?view=msvc-170#step-4---choose-workloads)

* Step 2: Install or update to latest [GPU driver](https://www.intel.com/content/www/us/en/download/785597/intel-arc-iris-xe-graphics-windows.html)

* Step 3 (Recommended): Install [Miniconda](https://docs.anaconda.com/free/miniconda/) for Python environment management. Choose Miniconda installer for Windows.

* Step 4: Install Intel® oneAPI Base Toolkit 2024.0:

  First, Create a Python 3.11 enviroment and activate it. In Anaconda Prompt:

  ```cmd
  conda create -n llm python=3.11 libuv

  conda activate llm
  ```

  ```eval_rst
  .. important::

     ``ipex-llm`` is tested with Python 3.9, 3.10 and 3.11. Python 3.11 is recommended for best practices.
  ```

  Then, use `pip` to install the Intel oneAPI Base Toolkit 2024.0:
  ```cmd
  pip install dpcpp-cpp-rt==2024.0.2 mkl-dpcpp==2024.0.0 onednn==2024.0.0
  ```

<!-- Intel® oneAPI Base Toolkit 2024.0 installation methods:

```eval_rst
.. tabs::
   .. tab:: Offline installer
      Download and install `Intel® oneAPI Base Toolkit <https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=window&distributions=offline>`_ version 2024.0 through Offline Installer.
      
      During installation, you could just continue with "Recommended Installation". If you would like to continue with "Custom Installation", please note that oneAPI Deep Neural Network Library, oneAPI Math Kernel Library, and oneAPI DPC++/C++ Compiler are required, the other components are optional.

   .. tab:: PIP installer
      Pip install oneAPI in your working conda environment.

      .. code-block:: bash

         pip install dpcpp-cpp-rt==2024.0.2 mkl-dpcpp==2024.0.0 onednn==2024.0.0

      .. note::
         Activating your working conda environment will automatically configure oneAPI environment variables.
``` -->

### Install IPEX-LLM
#### Install IPEX-LLM From PyPI

The easiest ways to install `ipex-llm` is the following commands,
choosing either US or CN website for `extra-index-url`:

```eval_rst
.. tabs::
   .. tab:: US

      .. code-block:: cmd

         conda activate llm

         pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

   .. tab:: CN

      .. code-block:: cmd

         conda activate llm

         pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/cn/
```

#### Install IPEX-LLM From Wheel

If you encounter network issues when installing IPEX, you can also install IPEX-LLM dependencies for Intel XPU from source archives. First you need to download and install torch/torchvision/ipex from wheels listed below before installing `ipex-llm`.

Download the wheels on Windows system:

```
wget https://intel-extension-for-pytorch.s3.amazonaws.com/ipex_stable/xpu/torch-2.1.0a0%2Bcxx11.abi-cp311-cp311-win_amd64.whl
wget https://intel-extension-for-pytorch.s3.amazonaws.com/ipex_stable/xpu/torchvision-0.16.0a0%2Bcxx11.abi-cp311-cp311-win_amd64.whl
wget https://intel-extension-for-pytorch.s3.amazonaws.com/ipex_stable/xpu/intel_extension_for_pytorch-2.1.10%2Bxpu-cp311-cp311-win_amd64.whl
```

You may install dependencies directly from the wheel archives and then install `ipex-llm` using following commands:

```
pip install torch-2.1.0a0+cxx11.abi-cp311-cp311-win_amd64.whl
pip install torchvision-0.16.0a0+cxx11.abi-cp311-cp311-win_amd64.whl
pip install intel_extension_for_pytorch-2.1.10+xpu-cp311-cp311-win_amd64.whl

pip install --pre --upgrade ipex-llm[xpu]
```

```eval_rst
.. note::

   All the wheel packages mentioned here are for Python 3.11. If you would like to use Python 3.9 or 3.10, you should modify the wheel names for ``torch``, ``torchvision``, and ``intel_extension_for_pytorch`` by replacing ``cp11`` with ``cp39`` or ``cp310``, respectively.
```

### Runtime Configuration

To use GPU acceleration on Windows, several environment variables are required before running a GPU example:

<!-- Make sure you are using CMD (Anaconda Prompt if using conda) as PowerShell is not supported, and configure oneAPI environment variables with: 

```cmd
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
```

Please also set the following environment variable if you would like to run LLMs on: -->

```eval_rst
.. tabs::
   .. tab:: Intel iGPU

      .. code-block:: cmd

         set SYCL_CACHE_PERSISTENT=1
         set BIGDL_LLM_XMX_DISABLED=1

   .. tab:: Intel Arc™ A-Series Graphics

      .. code-block:: cmd

         set SYCL_CACHE_PERSISTENT=1
```

```eval_rst
.. note::

   For **the first time** that **each model** runs on Intel iGPU/Intel Arc™ A300-Series or Pro A60, it may take several minutes to compile.
```

### Troubleshooting

#### 1. Error loading `intel_extension_for_pytorch`

If you met error when importing `intel_extension_for_pytorch`, please ensure that you have completed the following steps:

* Ensure that you have installed Visual Studio with "Desktop development with C++" workload.

* Make sure that the correct version of oneAPI, specifically 2024.0, is installed.

* Ensure that `libuv` is installed in your conda environment. This can be done during the creation of the environment with the command:
  ```cmd
  conda create -n llm python=3.11 libuv
  ```
  If you missed `libuv`, you can add it to your existing environment through
  ```cmd
  conda install libuv
  ```

<!-- * For oneAPI installed using the Offline installer, make sure you have configured oneAPI environment variables in your Anaconda Prompt through
  ```cmd
  call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
  ```
  Please note that you need to set these environment variables again once you have a new Anaconda Prompt window. -->

## Linux

### Prerequisites

IPEX-LLM GPU support on Linux has been verified on:

* Intel Arc™ A-Series Graphics
* Intel Data Center GPU Flex Series
* Intel Data Center GPU Max Series

```eval_rst
.. important::

    IPEX-LLM on Linux supports PyTorch 2.0 and PyTorch 2.1.
```

```eval_rst
.. important::

    We currently support the Ubuntu 20.04 operating system and later.
```

```eval_rst
.. tabs::
   .. tab:: PyTorch 2.1

      To enable IPEX-LLM for Intel GPUs with PyTorch 2.1, here are several prerequisite steps for tools installation and environment preparation:


      * Step 1: Install Intel GPU Driver version >= stable_775_20_20231219. We highly recommend installing the latest version of intel-i915-dkms using apt.

        .. seealso::

           Please refer to our `driver installation <https://dgpu-docs.intel.com/driver/installation.html>`_ for general purpose GPU capabilities.

           See `release page <https://dgpu-docs.intel.com/releases/index.html>`_ for latest version.

      * Step 2: Download and install `Intel® oneAPI Base Toolkit <https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html>`_ with version 2024.0. OneDNN, OneMKL and DPC++ compiler are needed, others are optional.

      Intel® oneAPI Base Toolkit 2024.0 installation methods:

      .. tabs::

         .. tab:: APT installer

            Step 1: Set up repository

            .. code-block:: bash

               wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
               echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
               sudo apt update

            Step 2: Install the package

            .. code-block:: bash

               sudo apt install intel-oneapi-common-vars=2024.0.0-49406 \
                  intel-oneapi-common-oneapi-vars=2024.0.0-49406 \
                  intel-oneapi-diagnostics-utility=2024.0.0-49093 \
                  intel-oneapi-compiler-dpcpp-cpp=2024.0.2-49895 \
                  intel-oneapi-dpcpp-ct=2024.0.0-49381 \
                  intel-oneapi-mkl=2024.0.0-49656 \
                  intel-oneapi-mkl-devel=2024.0.0-49656 \
                  intel-oneapi-mpi=2021.11.0-49493 \
                  intel-oneapi-mpi-devel=2021.11.0-49493 \
                  intel-oneapi-dal=2024.0.1-25 \
                  intel-oneapi-dal-devel=2024.0.1-25 \
                  intel-oneapi-ippcp=2021.9.1-5 \
                  intel-oneapi-ippcp-devel=2021.9.1-5 \
                  intel-oneapi-ipp=2021.10.1-13 \
                  intel-oneapi-ipp-devel=2021.10.1-13 \
                  intel-oneapi-tlt=2024.0.0-352 \
                  intel-oneapi-ccl=2021.11.2-5 \
                  intel-oneapi-ccl-devel=2021.11.2-5 \
                  intel-oneapi-dnnl-devel=2024.0.0-49521 \
                  intel-oneapi-dnnl=2024.0.0-49521 \
                  intel-oneapi-tcm-1.0=1.0.0-435

            .. note::

               You can uninstall the package by running the following command:

               .. code-block:: bash
               
                  sudo apt autoremove intel-oneapi-common-vars

         .. tab:: PIP installer

            Step 1: Install oneAPI in a user-defined folder, e.g., ``~/intel/oneapi``.

            .. code-block:: bash

               export PYTHONUSERBASE=~/intel/oneapi
               pip install dpcpp-cpp-rt==2024.0.2 mkl-dpcpp==2024.0.0 onednn==2024.0.0 --user

            .. note::

               The oneAPI packages are visible in ``pip list`` only if ``PYTHONUSERBASE`` is properly set.

            Step 2: Configure your working conda environment (e.g. with name ``llm``) to append oneAPI path (e.g. ``~/intel/oneapi/lib``) to the environment variable ``LD_LIBRARY_PATH``.

            .. code-block:: bash

               conda env config vars set LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/intel/oneapi/lib -n llm

            .. note::
               You can view the configured environment variables for your environment (e.g. with name ``llm``) by running ``conda env config vars list -n llm``.
               You can continue with your working conda environment and install ``ipex-llm`` as guided in the next section.

            .. note::

               You are recommended not to install other pip packages in the user-defined folder for oneAPI (e.g. ``~/intel/oneapi``).
               You can uninstall the oneAPI package by simply deleting the package folder, and unsetting the configuration of your working conda environment (e.g., with name ``llm``).
               
               .. code-block:: bash
               
                  rm -r ~/intel/oneapi
                  conda env config vars unset LD_LIBRARY_PATH -n llm

         .. tab:: Offline installer
         
            Using the offline installer allows you to customize the installation path.

            .. code-block:: bash
            
               wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/20f4e6a1-6b0b-4752-b8c1-e5eacba10e01/l_BaseKit_p_2024.0.0.49564_offline.sh
               sudo sh ./l_BaseKit_p_2024.0.0.49564_offline.sh

            .. note::

                  You can also modify the installation or uninstall the package by running the following commands:

                  .. code-block:: bash

                     cd /opt/intel/oneapi/installer
                     sudo ./installer

   .. tab:: PyTorch 2.0

      To enable IPEX-LLM for Intel GPUs with PyTorch 2.0, here're several prerequisite steps for tools installation and environment preparation:


      * Step 1: Install Intel GPU Driver version >= stable_775_20_20231219. Highly recommend installing the latest version of intel-i915-dkms using apt.

        .. seealso::

           Please refer to our `driver installation <https://dgpu-docs.intel.com/driver/installation.html>`_ for general purpose GPU capabilities.

           See `release page <https://dgpu-docs.intel.com/releases/index.html>`_ for latest version.

      * Step 2: Download and install `Intel® oneAPI Base Toolkit <https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html>`_ with version 2023.2. OneDNN, OneMKL and DPC++ compiler are needed, others are optional.

      Intel® oneAPI Base Toolkit 2023.2 installation methods:

      .. tabs::
         .. tab:: APT installer

            Step 1: Set up repository

            .. code-block:: bash

               wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
               echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
               sudo apt update

            Step 2: Install the packages

            .. code-block:: bash

               sudo apt install -y intel-oneapi-common-vars=2023.2.0-49462 \
                  intel-oneapi-compiler-cpp-eclipse-cfg=2023.2.0-49495 intel-oneapi-compiler-dpcpp-eclipse-cfg=2023.2.0-49495 \
                  intel-oneapi-diagnostics-utility=2022.4.0-49091 \
                  intel-oneapi-compiler-dpcpp-cpp=2023.2.0-49495 \
                  intel-oneapi-mkl=2023.2.0-49495 intel-oneapi-mkl-devel=2023.2.0-49495 \
                  intel-oneapi-mpi=2021.10.0-49371 intel-oneapi-mpi-devel=2021.10.0-49371 \
                  intel-oneapi-tbb=2021.10.0-49541 intel-oneapi-tbb-devel=2021.10.0-49541\
                  intel-oneapi-ccl=2021.10.0-49084 intel-oneapi-ccl-devel=2021.10.0-49084\
                  intel-oneapi-dnnl-devel=2023.2.0-49516 intel-oneapi-dnnl=2023.2.0-49516

            .. note::

               You can uninstall the package by running the following command:

               .. code-block:: bash
               
                  sudo apt autoremove intel-oneapi-common-vars

         .. tab:: PIP installer

            Step 1: Install oneAPI in a user-defined folder, e.g., ``~/intel/oneapi``

            .. code-block:: bash

               export PYTHONUSERBASE=~/intel/oneapi
               pip install dpcpp-cpp-rt==2023.2.0 mkl-dpcpp==2023.2.0 onednn-cpu-dpcpp-gpu-dpcpp==2023.2.0 --user

            .. note::

               The oneAPI packages are visible in ``pip list`` only if ``PYTHONUSERBASE`` is properly set.

            Step 2: Configure your working conda environment (e.g. with name ``llm``) to append oneAPI path (e.g. ``~/intel/oneapi/lib``) to the environment variable ``LD_LIBRARY_PATH``.

            .. code-block:: bash

               conda env config vars set LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/intel/oneapi/lib -n llm

            .. note::
               You can view the configured environment variables for your environment (e.g. with name ``llm``) by running ``conda env config vars list -n llm``.
               You can continue with your working conda environment and install ``ipex-llm`` as guided in the next section.

            .. note::

               You are recommended not to install other pip packages in the user-defined folder for oneAPI (e.g. ``~/intel/oneapi``).
               You can uninstall the oneAPI package by simply deleting the package folder, and unsetting the configuration of your working conda environment (e.g., with name ``llm``).
               
               .. code-block:: bash
               
                  rm -r ~/intel/oneapi
                  conda env config vars unset LD_LIBRARY_PATH -n llm

         .. tab:: Offline installer
         
            Using the offline installer allows you to customize the installation path.

            .. code-block:: bash
            
               wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/992857b9-624c-45de-9701-f6445d845359/l_BaseKit_p_2023.2.0.49397_offline.sh
               sudo sh ./l_BaseKit_p_2023.2.0.49397_offline.sh

            .. note::

               You can also modify the installation or uninstall the package by running the following commands:

               .. code-block:: bash

                  cd /opt/intel/oneapi/installer
                  sudo ./installer
```

### Install IPEX-LLM
#### Install IPEX-LLM From PyPI

We recommend using [miniconda](https://docs.conda.io/en/latest/miniconda.html) to create a python 3.11 enviroment:

```eval_rst
.. important::

   ``ipex-llm`` is tested with Python 3.9, 3.10 and 3.11. Python 3.11 is recommended for best practices.
```

```eval_rst
.. important::
   Make sure you install matching versions of ipex-llm/pytorch/IPEX and oneAPI Base Toolkit. IPEX-LLM with Pytorch 2.1 should be used with oneAPI Base Toolkit version 2024.0. IPEX-LLM with Pytorch 2.0 should be used with oneAPI Base Toolkit version 2023.2.
```

```eval_rst
.. tabs::
   .. tab:: PyTorch 2.1
      Choose either US or CN website for `extra-index-url`:

      .. tabs::
         .. tab:: US

            .. code-block:: bash

               conda create -n llm python=3.11
               conda activate llm

               pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

            .. note::

               The ``xpu`` option will install IPEX-LLM with PyTorch 2.1 by default, which is equivalent to

               .. code-block:: bash

                  pip install --pre --upgrade ipex-llm[xpu_2.1] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

         .. tab:: CN

            .. code-block:: bash

               conda create -n llm python=3.11
               conda activate llm

               pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/cn/

            .. note::

               The ``xpu`` option will install IPEX-LLM with PyTorch 2.1 by default, which is equivalent to

               .. code-block:: bash

                  pip install --pre --upgrade ipex-llm[xpu_2.1] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/cn/
            

   .. tab:: PyTorch 2.0
      Choose either US or CN website for `extra-index-url`:

      .. tabs::
         .. tab:: US

            .. code-block:: bash

               conda create -n llm python=3.11
               conda activate llm

               pip install --pre --upgrade ipex-llm[xpu_2.0] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

         .. tab:: CN

            .. code-block:: bash

               conda create -n llm python=3.11
               conda activate llm

               pip install --pre --upgrade ipex-llm[xpu_2.0] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/cn/

```

#### Install IPEX-LLM From Wheel

If you encounter network issues when installing IPEX, you can also install IPEX-LLM dependencies for Intel XPU from source archives. First you need to download and install torch/torchvision/ipex from wheels listed below before installing `ipex-llm`.

```eval_rst
.. tabs::
   .. tab:: PyTorch 2.1

      .. code-block:: bash

         # get the wheels on Linux system for IPEX 2.1.10+xpu
         wget https://intel-extension-for-pytorch.s3.amazonaws.com/ipex_stable/xpu/torch-2.1.0a0%2Bcxx11.abi-cp311-cp311-linux_x86_64.whl
         wget https://intel-extension-for-pytorch.s3.amazonaws.com/ipex_stable/xpu/torchvision-0.16.0a0%2Bcxx11.abi-cp311-cp311-linux_x86_64.whl
         wget https://intel-extension-for-pytorch.s3.amazonaws.com/ipex_stable/xpu/intel_extension_for_pytorch-2.1.10%2Bxpu-cp311-cp311-linux_x86_64.whl

      Then you may install directly from the wheel archives using following commands:

      .. code-block:: bash

         # install the packages from the wheels
         pip install torch-2.1.0a0+cxx11.abi-cp311-cp311-linux_x86_64.whl
         pip install torchvision-0.16.0a0+cxx11.abi-cp311-cp311-linux_x86_64.whl
         pip install intel_extension_for_pytorch-2.1.10+xpu-cp311-cp311-linux_x86_64.whl

         # install ipex-llm for Intel GPU
         pip install --pre --upgrade ipex-llm[xpu]

   .. tab:: PyTorch 2.0

      .. code-block:: bash

         # get the wheels on Linux system for IPEX 2.0.110+xpu
         wget https://intel-extension-for-pytorch.s3.amazonaws.com/ipex_stable/xpu/torch-2.0.1a0%2Bcxx11.abi-cp311-cp311-linux_x86_64.whl
         wget https://intel-extension-for-pytorch.s3.amazonaws.com/ipex_stable/xpu/torchvision-0.15.2a0%2Bcxx11.abi-cp311-cp311-linux_x86_64.whl
         wget https://intel-extension-for-pytorch.s3.amazonaws.com/ipex_stable/xpu/intel_extension_for_pytorch-2.0.110%2Bxpu-cp311-cp311-linux_x86_64.whl

      Then you may install directly from the wheel archives using following commands:

      .. code-block:: bash

         # install the packages from the wheels
         pip install torch-2.0.1a0+cxx11.abi-cp311-cp311-linux_x86_64.whl
         pip install torchvision-0.15.2a0+cxx11.abi-cp311-cp311-linux_x86_64.whl
         pip install intel_extension_for_pytorch-2.0.110+xpu-cp311-cp311-linux_x86_64.whl

         # install ipex-llm for Intel GPU
         pip install --pre --upgrade ipex-llm[xpu_2.0]

```

```eval_rst
.. note::

   All the wheel packages mentioned here are for Python 3.11. If you would like to use Python 3.9 or 3.10, you should modify the wheel names for ``torch``, ``torchvision``, and ``intel_extension_for_pytorch`` by replacing ``cp11`` with ``cp39`` or ``cp310``, respectively.
```

### Runtime Configuration

To use GPU acceleration on Linux, several environment variables are required or recommended before running a GPU example.

```eval_rst
.. tabs::
   .. tab:: Intel Arc™ A-Series and Intel Data Center GPU Flex

      For Intel Arc™ A-Series Graphics and Intel Data Center GPU Flex Series, we recommend:

      .. code-block:: bash

         # Configure oneAPI environment variables. Required step for APT or offline installed oneAPI.
         # Skip this step for PIP-installed oneAPI since the environment has already been configured in LD_LIBRARY_PATH.
         source /opt/intel/oneapi/setvars.sh

         # Recommended Environment Variables for optimal performance
         export USE_XETLA=OFF
         export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
         export SYCL_CACHE_PERSISTENT=1

   .. tab:: Intel Data Center GPU Max

      For Intel Data Center GPU Max Series, we recommend:

      .. code-block:: bash

         # Configure oneAPI environment variables. Required step for APT or offline installed oneAPI.
         # Skip this step for PIP-installed oneAPI since the environment has already been configured in LD_LIBRARY_PATH.
         source /opt/intel/oneapi/setvars.sh

         # Recommended Environment Variables for optimal performance
         export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
         export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
         export SYCL_CACHE_PERSISTENT=1
         export ENABLE_SDP_FUSION=1

      Please note that ``libtcmalloc.so`` can be installed by ``conda install -c conda-forge -y gperftools=2.10``

   .. tab:: Intel iGPU

      .. code-block:: bash

         # Configure oneAPI environment variables. Required step for APT or offline installed oneAPI.
         # Skip this step for PIP-installed oneAPI since the environment has already been configured in LD_LIBRARY_PATH.
         source /opt/intel/oneapi/setvars.sh

         export SYCL_CACHE_PERSISTENT=1
         export BIGDL_LLM_XMX_DISABLED=1

```

```eval_rst
.. note::

   For **the first time** that **each model** runs on Intel iGPU/Intel Arc™ A300-Series or Pro A60, it may take several minutes to compile.
```

### Known issues

#### 1. Potential suboptimal performance with Linux kernel 6.2.0

For Ubuntu 22.04 and driver version < stable_775_20_20231219, the performance on Linux kernel 6.2.0 is worse than Linux kernel 5.19.0. You can use `sudo apt update && sudo apt install -y intel-i915-dkms intel-fw-gpu` to install the latest driver to solve this issue (need to reboot OS).

Tips: You can use `sudo apt list --installed | grep intel-i915-dkms` to check your intel-i915-dkms's version, the version should be latest and >= `1.23.9.11.231003.15+i19-1`.

#### 2. Driver installation unmet dependencies error: intel-i915-dkms

The last apt install command of the driver installation may produce the following error:

```
The following packages have unmet dependencies:
 intel-i915-dkms : Conflicts: intel-platform-cse-dkms
                   Conflicts: intel-platform-vsec-dkms
```

You can use `sudo apt install -y intel-i915-dkms intel-fw-gpu` to install instead. As the intel-platform-cse-dkms and intel-platform-vsec-dkms are already provided by intel-i915-dkms.

### Troubleshooting

#### 1. Cannot open shared object file: No such file or directory

Error where libmkl file is not found, for example,

```
OSError: libmkl_intel_lp64.so.2: cannot open shared object file: No such file or directory
```
```
Error: libmkl_sycl_blas.so.4: cannot open shared object file: No such file or directory
```

The reason for such errors is that oneAPI has not been initialized properly before running IPEX-LLM code or before importing IPEX package.

* For oneAPI installed using APT or Offline Installer, make sure you execute `setvars.sh` of oneAPI Base Toolkit before running IPEX-LLM.
* For PIP-installed oneAPI, activate your working environment and run ``echo $LD_LIBRARY_PATH`` to check if the installation path is properly configured for the environment. If the output does not contain oneAPI path (e.g. ``~/intel/oneapi/lib``), check [Prerequisites](#id1) to re-install oneAPI with PIP installer.
* Make sure you install matching versions of ipex-llm/pytorch/IPEX and oneAPI Base Toolkit. IPEX-LLM with PyTorch 2.1 should be used with oneAPI Base Toolkit version 2024.0. IPEX-LLM with PyTorch 2.0 should be used with oneAPI Base Toolkit version 2023.2.
