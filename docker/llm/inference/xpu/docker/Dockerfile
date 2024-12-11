FROM intel/oneapi:2024.2.1-0-devel-ubuntu22.04

ARG http_proxy
ARG https_proxy

ENV TZ=Asia/Shanghai
ENV PYTHONUNBUFFERED=1

# When cache is enabled SYCL runtime will try to cache and reuse JIT-compiled binaries.
ENV SYCL_CACHE_PERSISTENT=1

COPY chat.py /llm/chat.py
COPY benchmark.sh /llm/benchmark.sh

# Disable pip's cache behavior
ARG PIP_NO_CACHE_DIR=false

RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/intel-oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/intel-oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main " | tee /etc/apt/sources.list.d/oneAPI.list && \
    chmod 644 /usr/share/keyrings/intel-oneapi-archive-keyring.gpg && \
    rm /etc/apt/sources.list.d/intel-graphics.list && \
    wget -O- https://repositories.intel.com/graphics/intel-graphics.key | gpg --dearmor | tee /usr/share/keyrings/intel-graphics.gpg > /dev/null && \
    echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/graphics/ubuntu jammy arc" | tee /etc/apt/sources.list.d/intel.gpu.jammy.list && \
    chmod 644 /usr/share/keyrings/intel-graphics.gpg && \
    apt-get update && \
    apt-get install -y --no-install-recommends curl wget git libunwind8-dev vim less && \
    # Install PYTHON 3.11 and IPEX-LLM[xpu]
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    env DEBIAN_FRONTEND=noninteractive apt-get update && \
    # add-apt-repository requires gnupg, gpg-agent, software-properties-common
    apt-get install -y --no-install-recommends gnupg gpg-agent software-properties-common && \
    export PRE_DIR=$(pwd) && \
    # Install Compute Runtime
    mkdir -p /tmp/neo && \
    cd /tmp/neo && \
    wget https://github.com/oneapi-src/level-zero/releases/download/v1.18.5/level-zero_1.18.5+u22.04_amd64.deb && \
    wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.17791.9/intel-igc-core_1.0.17791.9_amd64.deb && \
    wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.17791.9/intel-igc-opencl_1.0.17791.9_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/24.39.31294.12/intel-level-zero-gpu_1.6.31294.12_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/24.39.31294.12/intel-opencl-icd_24.39.31294.12_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/24.39.31294.12/libigdgmm12_22.5.2_amd64.deb && \
    dpkg -i *.deb && \
    rm -rf /tmp/neo && \
    cd $PRE_DIR && \
    # Add Python 3.11 PPA repository
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install -y --no-install-recommends python3.11 git curl wget && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.11 /usr/bin/python3 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get install -y --no-install-recommends python3-pip python3.11-dev python3-wheel python3.11-distutils && \
    wget https://bootstrap.pypa.io/get-pip.py -O get-pip.py && \
    # Install FastChat from source requires PEP 660 support
    python3 get-pip.py && \
    rm get-pip.py && \
    pip install --upgrade requests argparse urllib3 && \
    pip install --pre --upgrade ipex-llm[xpu_arc] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/ && \
    pip install --pre pytorch-triton-xpu==3.0.0+1b2f15840e --index-url https://download.pytorch.org/whl/nightly/xpu && \
    # Fix Trivy CVE Issues
    pip install transformers_stream_generator einops tiktoken && \
    # Install opencl-related repos
    apt-get update && \
    # Install related libary of chat.py
    pip install --upgrade colorama && \
    # Download all-in-one benchmark and examples
    git clone https://github.com/intel-analytics/ipex-llm && \
    cp -r ./ipex-llm/python/llm/dev/benchmark/ ./benchmark && \
    cp -r ./ipex-llm/python/llm/example/GPU/HuggingFace/LLM ./examples && \
    # Install vllm dependencies
    pip install --upgrade fastapi && \
    pip install --upgrade "uvicorn[standard]" && \
    # Download vLLM-Serving
    cp -r ./ipex-llm/python/llm/example/GPU/vLLM-Serving/ ./vLLM-Serving && \
    # Download pp_serving
    mkdir -p /llm/pp_serving && \
    cp ./ipex-llm/python/llm/example/GPU/Pipeline-Parallel-Serving/*.py /llm/pp_serving/ && \
    # Download lightweight_serving
    mkdir -p /llm/lightweight_serving && \
    cp ./ipex-llm/python/llm/example/GPU/Lightweight-Serving/*.py /llm/lightweight_serving/ && \
    # Install related library of benchmarking
    pip install pandas omegaconf && \
    chmod +x /llm/benchmark.sh && \
    # Download Deepspeed-AutoTP
    cp -r ./ipex-llm/python/llm/example/GPU/Deepspeed-AutoTP/ ./Deepspeed-AutoTP && \
    # Install related library of Deepspeed-AutoTP
    pip install oneccl_bind_pt==2.3.100 --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/ && \
    pip install git+https://github.com/microsoft/DeepSpeed.git@ed8aed5 && \
    pip install git+https://github.com/intel/intel-extension-for-deepspeed.git@0eb734b && \
    pip install mpi4py && \
    apt-get update && \
    apt-get install -y --no-install-recommends google-perftools && \
    ln -s /usr/local/lib/python3.11/dist-packages/ipex_llm/libs/libtcmalloc.so /lib/libtcmalloc.so && \
    rm -rf ./ipex-llm


WORKDIR /llm/
ENV BIGDL_CHECK_DUPLICATE_IMPORT=0
