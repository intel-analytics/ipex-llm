# IPEX-LLM Installation: CPU

## Quick Installation

Install IPEX-LLM for CPU supports using pip through:

```bash
pip install --pre --upgrade ipex-llm[all] # install the latest ipex-llm nightly build with 'all' option
```

Please refer to [Environment Setup](#environment-setup) for more information.

```eval_rst
.. note::

   ``all`` option will trigger installation of all the dependencies for common LLM application development.

.. important::

   ``ipex-llm`` is tested with Python 3.9, 3.10 and 3.11; Python 3.11 is recommended for best practices.
```

## Recommended Requirements

Here list the recommended hardware and OS for smooth IPEX-LLM optimization experiences on CPU:

* Hardware

  * PCs equipped with 12th Gen Intel® Core™ processor or higher, and at least 16GB RAM
  * Servers equipped with Intel® Xeon® processors, at least 32G RAM.

* Operating System

  * Ubuntu 20.04 or later
  * CentOS 7 or later
  * Windows 10/11, with or without WSL

## Environment Setup

For optimal performance with LLM models using IPEX-LLM optimizations on Intel CPUs, here are some best practices for setting up environment:

First we recommend using [Conda](https://docs.conda.io/en/latest/miniconda.html) to create a python 3.11 enviroment:

```bash
conda create -n llm python=3.11
conda activate llm

pip install --pre --upgrade ipex-llm[all] # install the latest ipex-llm nightly build with 'all' option
```

Then for running a LLM model with IPEX-LLM optimizations (taking an `example.py` an example):

```eval_rst	
.. tabs::

   .. tab:: Client

      It is recommended to run directly with full utilization of all CPU cores:

      .. code-block:: bash

         python example.py

   .. tab:: Server

      It is recommended to run with all the physical cores of a single socket:

      .. code-block:: bash

         # e.g. for a server with 48 cores per socket
         export OMP_NUM_THREADS=48
         numactl -C 0-47 -m 0 python example.py
```
