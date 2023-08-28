# BigDL-LLM Installation

## Quick Installation

Install BigDL-LLM based on the device you choose:

```eval_rst
.. tabs::

   .. tab:: CPU

      .. code-block:: bash

         pip install bigdl-llm[all]

   .. tab:: GPU

      .. code-block:: bash

         pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
```

## System Recommendation
Here's a list of recommended hardware and OS.

### ⚠️Hardware

* PCs equipped with 12th Gen Intel® Core™ processor or higher, and at least 16GB RAM
* Servers equipped with Intel® Xeon® processors, at least 32G RAM.

### ⚠️Operating System

* Ubuntu 20.04 or later
* CentOS 7 or later
* Windows 10/11, with or without WSL

## Environment Management

We recommend using [Conda](https://docs.conda.io/en/latest/miniconda.html) to create a python 3.9 enviroment and install necessary libs.

```bash
conda create -n llm python=3.9
conda activate llm
```