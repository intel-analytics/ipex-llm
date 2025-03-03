# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

FROM intel/oneapi-basekit:2024.0.1-devel-ubuntu22.04

ARG http_proxy
ARG https_proxy
ARG PIP_NO_CACHE_DIR=false

ENV TZ=Asia/Shanghai SYCL_CACHE_PERSISTENT=1

RUN set -eux && \
    # Set timezone
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    #
    # Retrieve Intel OneAPI and GPU repository keys
    wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/intel-oneapi-archive-keyring.gpg > /dev/null && \	
    echo "deb [signed-by=/usr/share/keyrings/intel-oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main " | tee /etc/apt/sources.list.d/oneAPI.list && \	
    chmod 644 /usr/share/keyrings/intel-oneapi-archive-keyring.gpg && \	
    rm /etc/apt/sources.list.d/intel-graphics.list && \	
    wget -O- https://repositories.intel.com/graphics/intel-graphics.key | gpg --dearmor | tee /usr/share/keyrings/intel-graphics.gpg > /dev/null && \	
    echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/graphics/ubuntu jammy arc" | tee /etc/apt/sources.list.d/intel.gpu.jammy.list && \	
    chmod 644 /usr/share/keyrings/intel-graphics.gpg && \
    #
    # Update package lists and install dependencies
    apt-get update && \
    apt-get install -y --no-install-recommends \
        curl wget git vim less libunwind8-dev \
        intel-opencl-icd intel-level-zero-gpu level-zero level-zero-dev \
        gnupg gpg-agent software-properties-common && \
    #
    # Add Python 3.11 PPA and install Python 3.11
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install -y --no-install-recommends \
        python3.11 python3-pip python3.11-dev python3-wheel python3.11-distutils && \
    #
    # Remove unnecessary packages and clean up
    apt-get remove -y python3-blinker && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    #
    # Set Python 3.11 as default
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    #
    # Upgrade pip
    wget -qO /tmp/get-pip.py https://bootstrap.pypa.io/get-pip.py && \
    python3 /tmp/get-pip.py && \
    rm /tmp/get-pip.py && \
    #
    # Install Intel XPU ipex-llm and dependencies
    pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/ && \
    pip install transformers==4.36.0 peft==0.10.0 datasets bitsandbytes scipy fire && \
    #
    # Clone finetuning scripts and setup configuration
    git clone https://github.com/intel-analytics/IPEX-LLM.git && \
    mv IPEX-LLM/python/llm/example/GPU/LLM-Finetuning /LLM-Finetuning && \
    rm -rf IPEX-LLM && \
    mkdir -p /root/.cache/huggingface/accelerate && \
    mv /LLM-Finetuning/axolotl/default_config.yaml /root/.cache/huggingface/accelerate/

# Copy startup script
COPY ./start-qlora-finetuning-on-xpu.sh /start-qlora-finetuning-on-xpu.sh
