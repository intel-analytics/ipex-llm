# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# First stage: build oneccl
FROM intel/oneapi-basekit:2025.0.1-0-devel-ubuntu22.04 AS build

ARG http_proxy
ARG https_proxy
ARG PIP_NO_CACHE_DIR=false

# Set environment variables
ENV TZ=Asia/Shanghai PYTHONUNBUFFERED=1

# Copy patch file and benchmark scripts
COPY ./ccl_torch.patch /tmp/
COPY ./vllm_online_benchmark.py ./vllm_offline_inference.py ./vllm_offline_inference_vision_language.py \
     ./payload-1024.lua ./start-vllm-service.sh ./benchmark_vllm_throughput.py ./benchmark_vllm_latency.py \
     ./start-pp_serving-service.sh /llm/
     
RUN set -eux && \
    #
    # Update and install basic dependencies
    apt-get update && \
    apt-get install -y --no-install-recommends \
      curl wget git libunwind8-dev vim less gnupg gpg-agent software-properties-common \
      libfabric-dev wrk libaio-dev numactl && \
    # 
    # Set timezone
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone && \
    # 
    # Install Python 3.11
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install -y --no-install-recommends python3.11 python3-pip python3.11-dev python3.11-distutils python3-wheel && \
    rm /usr/bin/python3 && ln -s /usr/bin/python3.11 /usr/bin/python3 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    # 
    # Install pip and essential Python packages
    wget https://bootstrap.pypa.io/get-pip.py -O get-pip.py && \
    python3 get-pip.py && rm get-pip.py && \
    #
    # Install Intel GPU OpenCL Driver and Compute Runtime
    mkdir -p /tmp/neo && \
    cd /tmp/neo && \
    wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.5.6/intel-igc-core-2_2.5.6+18417_amd64.deb && \
    wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.5.6/intel-igc-opencl-2_2.5.6+18417_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-level-zero-gpu-dbgsym_1.6.32224.5_amd64.ddeb && \
    wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-level-zero-gpu_1.6.32224.5_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-opencl-icd-dbgsym_24.52.32224.5_amd64.ddeb && \
    wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-opencl-icd_24.52.32224.5_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/libigdgmm12_22.5.5_amd64.deb && \
    dpkg -i *.deb && \
    #
    # Install Intel PyTorch extension for LLM inference
    pip install --pre --upgrade ipex-llm[xpu_2.6] --extra-index-url https://download.pytorch.org/whl/test/xpu && \
    #
    # Build torch-ccl
    mkdir /build && \
    cd /build && \
    git clone https://github.com/intel/torch-ccl.git && \
    cd torch-ccl && \
    git checkout ccl_torch2.5.0+xpu && \
    git submodule sync && \
    git submodule update --init --recursive && \
    # This patch will enable build torch-ccl with pytorch 2.6 environment
    git apply /tmp/ccl_torch.patch && \
    USE_SYSTEM_ONECCL=ON COMPUTE_BACKEND=dpcpp python setup.py bdist_wheel
    # File path: /build/torch-ccl/dist/oneccl_bind_pt-2.5.0+xpu-cp311-cp311-linux_x86_64.whl

# Second stage: Final runtime image
FROM intel/oneapi-basekit:2025.0.1-0-devel-ubuntu22.04

# Copy the built torch-ccl package from the build stage
COPY --from=build /build/torch-ccl/dist/oneccl_bind_pt-2.5.0+xpu-cp311-cp311-linux_x86_64.whl /opt/
COPY --from=build /llm/ /llm/

ARG http_proxy
ARG https_proxy
ARG PIP_NO_CACHE_DIR=false

# Set environment variables
ENV TZ=Asia/Shanghai PYTHONUNBUFFERED=1 VLLM_RPC_TIMEOUT=100000

RUN set -eux && \
    #
    # Update and install basic dependencies
    apt-get update && \
    apt-get install -y --no-install-recommends \
      curl wget git libunwind8-dev vim less gnupg gpg-agent software-properties-common \
      libfabric-dev wrk libaio-dev numactl && \
    # 
    # Set timezone
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone && \
    # 
    # Install Python 3.11
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install -y --no-install-recommends python3.11 python3-pip python3.11-dev python3.11-distutils python3-wheel && \
    rm /usr/bin/python3 && ln -s /usr/bin/python3.11 /usr/bin/python3 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    # 
    # Install pip and essential Python packages
    wget https://bootstrap.pypa.io/get-pip.py -O get-pip.py && \
    python3 get-pip.py && rm get-pip.py && \
    pip install --upgrade requests argparse urllib3 && \
    pip install --pre --upgrade ipex-llm[xpu_2.6] --extra-index-url https://download.pytorch.org/whl/test/xpu && \
    pip install transformers_stream_generator einops tiktoken && \
    pip install --upgrade colorama && \
    # 
    git clone https://github.com/intel/ipex-llm.git && \
    cp -r ./ipex-llm/python/llm/dev/benchmark/ ./benchmark && \
    cp -r ./ipex-llm/python/llm/example/GPU/HuggingFace/LLM ./examples && \
    cp -r ./ipex-llm/python/llm/example/GPU/vLLM-Serving/ ./vLLM-Serving && \
    #
    # Download pp_serving
    mkdir -p /llm/pp_serving && \
    cp ./ipex-llm/python/llm/example/GPU/Pipeline-Parallel-Serving/*.py /llm/pp_serving/ && \
    #
    # Download lightweight_serving
    mkdir -p /llm/lightweight_serving && \
    cp ./ipex-llm/python/llm/example/GPU/Lightweight-Serving/*.py /llm/lightweight_serving/ && \
    rm -rf ./ipex-llm && \
    #
    # Install vllm dependencies
    pip install --upgrade fastapi && \
    pip install --upgrade "uvicorn[standard]" && \
    #
    # Install torch-ccl
    pip install /opt/oneccl_bind_pt-2.5.0+xpu-cp311-cp311-linux_x86_64.whl && \
    #
    # Install Internal oneccl
    cd /opt && \
    wget https://sourceforge.net/projects/oneccl-wks/files/2025.0.0.6.6-release/oneccl_wks_installer_2025.0.0.6.6.sh && \
    bash oneccl_wks_installer_2025.0.0.6.6.sh && \
    apt-get update && \
    apt-get install -y --no-install-recommends libfabric-dev wrk libaio-dev numactl && \
    # 
    # Remove breaks install packages
    apt-get remove -y libze-dev libze-intel-gpu1 && \
    #
    # Install compute runtime
    mkdir -p /tmp/neo && \
    cd /tmp/neo && \
    wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.5.6/intel-igc-core-2_2.5.6+18417_amd64.deb && \
    wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.5.6/intel-igc-opencl-2_2.5.6+18417_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-level-zero-gpu-dbgsym_1.6.32224.5_amd64.ddeb && \
    wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-level-zero-gpu_1.6.32224.5_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-opencl-icd-dbgsym_24.52.32224.5_amd64.ddeb && \
    wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-opencl-icd_24.52.32224.5_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/libigdgmm12_22.5.5_amd64.deb && \
    dpkg -i *.deb && rm -rf /tmp/neo && \
    mkdir -p /llm && \
    cd /llm && \
    rm -rf /tmp/neo && \
    #
    # Install vllm
    git clone -b 0.6.6 https://github.com/analytics-zoo/vllm.git /llm/vllm && \
    cd /llm/vllm && \
    pip install setuptools-scm && \
    pip install --upgrade cmake && \
    VLLM_TARGET_DEVICE=xpu pip install --no-build-isolation -v /llm/vllm && \
    pip install mpi4py fastapi uvicorn openai && \
    pip install gradio==4.43.0 && \
    pip install ray

WORKDIR /llm/
ENTRYPOINT ["bash", "/llm/start-vllm-service.sh"]
