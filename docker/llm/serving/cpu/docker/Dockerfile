# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Stage 1: Build stage to handle file preparation
FROM ubuntu:22.04 as build

# Copy the files to the build image
COPY ./start-notebook.sh               /llm/
COPY ./model_adapter.py.patch          /llm/
COPY ./vllm_offline_inference.py       /llm/
COPY ./payload-1024.lua                /llm/
COPY ./start-vllm-service.sh           /llm/
COPY ./benchmark_vllm_throughput.py    /llm/
COPY ./start-fastchat-service.sh       /llm/

# Stage 2: Final image that only includes necessary runtime artifacts
FROM ubuntu:22.04

# Copy the scripts from the build stage
COPY --from=build /llm /llm/

ARG http_proxy
ARG https_proxy
# Disable pip's cache behavior
ARG PIP_NO_CACHE_DIR=false
ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    # Install basic utilities
    libunwind8-dev vim less \
    # Version control and download tools
    git curl wget \
    # add-apt-repository requires gnupg, gpg-agent, software-properties-common
    gnupg gpg-agent software-properties-common \
    # Install performance testing tool, NUMA (Non-Uniform Memory Access) support, and patch tool
    wrk numactl patch && \
# Install Python 3.11
    # Add Python 3.11 PPA repository
    add-apt-repository ppa:deadsnakes/ppa -y && \
    # Install Python 3.11 and related packages
    apt-get update && apt-get install -y --no-install-recommends python3.11 python3-pip python3.11-dev python3-wheel python3.11-distutils && \
    # Remove the original /usr/bin/python3 symbolic link
    rm /usr/bin/python3 && \
    # Create a symbolic link pointing to Python 3.11 at /usr/bin/python3
    ln -s /usr/bin/python3.11 /usr/bin/python3 && \
    # Create a symbolic link pointing to /usr/bin/python3 at /usr/bin/python
    ln -s /usr/bin/python3 /usr/bin/python && \
# Download and install pip, install FastChat from source requires PEP 660 support
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py && \
# Install Basic Python utilities
    pip install --upgrade requests argparse urllib3 && \
# Download ipex-llm-tutorial
    pip install --upgrade jupyterlab && \
    git clone https://github.com/intel-analytics/ipex-llm-tutorial /llm/ipex-llm-tutorial && \
    chmod +x /llm/start-notebook.sh && \
# Download all-in-one benchmark
    git clone https://github.com/intel-analytics/IPEX-LLM && \
    cp -r ./IPEX-LLM/python/llm/dev/benchmark/ /llm/benchmark && \
# Copy chat.py script
    pip install --upgrade colorama && \
    cp -r ./IPEX-LLM/python/llm/portable-zip/ /llm/portable-zip && \
# Install all-in-one dependencies
    pip install --upgrade omegaconf && \
    pip install --upgrade pandas && \
# Install ipex-llm
    pip uninstall -y setuptools && \
    pip install --pre --upgrade ipex-llm[serving] && \
    # Fix Trivy CVE Issues
    pip install Jinja2==3.1.3 transformers==4.36.2 gradio==4.19.2 cryptography==42.0.4 && \
    # Fix CVE-2024-22195
    pip install Jinja2==3.1.3 && \
    pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install intel-extension-for-pytorch==2.2.0 && \
    pip install oneccl_bind_pt==2.2.0 --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/cn/ && \
    pip install transformers==4.36.2 && \
# Install vllm dependencies
    pip install --upgrade fastapi && \
    pip install --upgrade "uvicorn[standard]" && \
# Add Qwen support
    pip install --upgrade transformers_stream_generator einops && \
# Fix Qwen model adapter in fastchat
    patch /usr/local/lib/python3.11/dist-packages/fastchat/model/model_adapter.py < /llm/model_adapter.py.patch && \
# Copy vLLM-Serving
    cp -r ./IPEX-LLM/python/llm/example/CPU/vLLM-Serving/ /llm/vLLM-Serving && \
    rm -rf ./IPEX-LLM && \
# Fix vllm service 
    pip install pydantic==1.10.11 && \
# Install vllm
    apt-get install -y g++ gcc-12 g++-12 libnuma-dev && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12 && \
    git clone https://github.com/vllm-project/vllm.git && \
    cd ./vllm && \
    git checkout v0.6.6.post1 && \
    pip install cmake>=3.26 wheel packaging ninja "setuptools-scm>=8" numpy && \
    pip uninstall -y intel-extension-for-pytorch && \
    pip install -v -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu && \
    VLLM_TARGET_DEVICE=cpu python3 setup.py install && \
    pip install ray triton==3.1.0 && \
# Clean up unnecessary files to reduce image size
    rm -rf /var/lib/apt/lists/*

WORKDIR /llm/
