# First stage: Build torch-ccl
FROM intel/oneapi-basekit:2025.0.1-0-devel-ubuntu22.04 AS build

ARG http_proxy
ARG https_proxy
ARG PIP_NO_CACHE_DIR=false

# Set environment variables
ENV TZ=Asia/Shanghai PYTHONUNBUFFERED=1

# Copy patch file and benchmark scripts
ADD ./ccl_torch.patch /tmp/
COPY ./vllm_online_benchmark.py ./vllm_offline_inference.py ./vllm_offline_inference_vision_language.py \
     ./payload-1024.lua ./start-vllm-service.sh ./benchmark_vllm_throughput.py ./benchmark_vllm_latency.py \
     ./start-pp_serving-service.sh /llm/

# Install necessary packages and dependencies, configure Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget git libunwind8-dev vim less gnupg gpg-agent software-properties-common \
    python3.11 python3-pip python3.11-dev python3-wheel python3.11-distutils \
    libfabric-dev wrk libaio-dev numactl && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    ln -s /usr/bin/python3.11 /usr/bin/python3 && ln -s /usr/bin/python3 /usr/bin/python && \
    wget https://bootstrap.pypa.io/get-pip.py -O get-pip.py && python3 get-pip.py && rm get-pip.py && \
    pip install --upgrade requests argparse urllib3 && \
    #
    # Install Intel compute runtime
    mkdir -p /tmp/neo && cd /tmp/neo && \
    wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.5.6/intel-igc-core-2_2.5.6+18417_amd64.deb && \
    wget https://github.com/intel/intel-graphics-compiler/releases/download/v2.5.6/intel-igc-opencl-2_2.5.6+18417_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-level-zero-gpu_1.6.32224.5_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/intel-opencl-icd_24.52.32224.5_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/24.52.32224.5/libigdgmm12_22.5.5_amd64.deb && \
    dpkg -i *.deb && rm -rf /tmp/neo && \
    #
    # Install Intel PyTorch extension for LLM inference
    pip install --pre --upgrade ipex-llm[xpu_2.6] --extra-index-url https://download.pytorch.org/whl/test/xpu && \
    #
    # Build torch-ccl
    mkdir /build && cd /build && \
    git clone https://github.com/intel/torch-ccl.git && cd torch-ccl && \
    git checkout ccl_torch2.5.0+xpu && git submodule update --init --recursive && \
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

# Install necessary system dependencies, configure Python 3.11, and install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget git libunwind8-dev vim less gnupg gpg-agent software-properties-common kmod \
    python3.11 python3-pip python3.11-dev python3-wheel python3.11-distutils \
    libfabric-dev wrk libaio-dev numactl && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    ln -s /usr/bin/python3.11 /usr/bin/python3 && ln -s /usr/bin/python3 /usr/bin/python && \
    wget https://bootstrap.pypa.io/get-pip.py -O get-pip.py && python3 get-pip.py && rm get-pip.py && \
    pip install --upgrade requests argparse urllib3 && \
    pip install --pre --upgrade ipex-llm[xpu_2.6] --extra-index-url https://download.pytorch.org/whl/test/xpu && \
    pip install transformers_stream_generator einops tiktoken colorama fastapi "uvicorn[standard]" && \
    #
    # Install torch-ccl
    pip install /opt/oneccl_bind_pt-2.5.0+xpu-cp311-cp311-linux_x86_64.whl && \
    #
    # Install Intel oneCCL (internal version)
    cd /opt && wget https://sourceforge.net/projects/oneccl-wks/files/2025.0.0.6.6-release/oneccl_wks_installer_2025.0.0.6.6.sh && \
    bash oneccl_wks_installer_2025.0.0.6.6.sh && \
    #
    # Install vLLM dependencies and repository
    git clone -b 0.6.6 https://github.com/analytics-zoo/vllm.git /llm/vllm && cd /llm/vllm && \
    pip install setuptools-scm cmake mpi4py openai gradio==4.43.0 ray && \
    VLLM_TARGET_DEVICE=xpu pip install --no-build-isolation -v . && \
    #
    # Cleanup
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /llm/
