# IPEX-LLM Installation: CPU

## Quick Installation

Install IPEX-LLM for CPU supports using pip through:

- For **Linux users**:

  ```bash
  pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu
  ```

- For **Windows users**:

  ```cmd
  pip install --pre --upgrade ipex-llm[all]
  ```

Please refer to [Environment Setup](#environment-setup) for more information.

> [!NOTE]
> `all` option will trigger installation of all the dependencies for common LLM application development.

> [!IMPORTANT]
> `ipex-llm` is tested with Python 3.9, 3.10 and 3.11; Python 3.11 is recommended for best practices.


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

First we recommend using [Conda](https://conda-forge.org/download/) to create a python 3.11 enviroment:

- For **Linux users**:

  ```bash
  conda create -n llm python=3.11
  conda activate llm

  pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu
  ```

- For **Windows users**:

  ```cmd
  conda create -n llm python=3.11
  conda activate llm

  pip install --pre --upgrade ipex-llm[all]
  ```

Then for running a LLM model with IPEX-LLM optimizations (taking an `example.py` an example):

- For **running on Client**:

  It is recommended to run directly with full utilization of all CPU cores:

  ```bash
  python example.py
  ```

- For **running on Server**:

  It is recommended to run with all the physical cores of a single socket:

  ```bash
  # e.g. for a server with 48 cores per socket
  export OMP_NUM_THREADS=48
  numactl -C 0-47 -m 0 python example.py
  ```
