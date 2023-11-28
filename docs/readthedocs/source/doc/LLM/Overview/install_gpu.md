# BigDL-LLM Installation: GPU

## Quick Installation

Install BigDL-LLM for GPU supports using pip through:

```bash
pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu # install bigdl-llm for GPU
```

Please refer to [Environment Setup](#environment-setup) for more information.

```eval_rst
.. note::

   The above command will install ``intel_extension_for_pytorch==2.0.110+xpu`` as default. You can install specific ``ipex``/``torch`` version for your need.

.. important::

   ``bigdl-llm`` is tested with Python 3.9, which is recommended for best practices.
```

## Recommended Requirements

BigDL-LLM for GPU supports has been verified on:

* Intel Arc™ A-Series Graphics
* Intel Data Center GPU Flex Series
* Intel Data Center GPU Max Series

```eval_rst
.. note::

   We currently supoort the Ubuntu 20.04 operating system or later. Windows supoort is in progress.
```

To apply Intel GPU acceleration, there're several steps for tools installation and environment preparation:

* Step 1, only Linux system is supported now, Ubuntu 22.04 and Linux kernel 5.19.0 is prefered.
  ```eval_rst
  .. note::

     As th
  ```
* Step 2, please refer to our [driver installation](https://dgpu-docs.intel.com/driver/installation.html) for general purpose GPU capabilities.
  ```eval_rst
  .. note::

    IPEX 2.0.110+xpu requires Intel GPU Driver version >= stable_647_21_20230714, see <release page>[https://dgpu-docs.intel.com/releases/index.html] for latest version.
  ```
* Step 3, you also need to download and install [Intel® oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html). OneMKL and DPC++ compiler are needed, others are optional.
  ```eval_rst
  .. note::

    IPEX 2.0.110+xpu requires Intel® oneAPI Base Toolkit's version == 2023.2.0.
  ```

## Environment Setup

For optimal performance with LLM models using BigDL-LLM optimizations on Intel GPUs, here are some best practices for setting up environment:

First we recommend using [Conda](https://docs.conda.io/en/latest/miniconda.html) to create a python 3.9 enviroment:

```bash
conda create -n llm python=3.9
conda activate llm

pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu # install bigdl-llm for GPU
```

Then for running a LLM model with BigDL-LLM optimizations, several environment variables are recommended:

```bash
# configures OneAPI environment variables
source /opt/intel/oneapi/setvars.sh

export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
```
