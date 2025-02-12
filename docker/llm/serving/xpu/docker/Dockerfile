# We will need to build the oneccl from ONEAPI 2025 environment...
FROM intel/oneapi-basekit:2025.0.1-0-devel-ubuntu22.04 AS build

ARG http_proxy
ARG https_proxy

ENV TZ=Asia/Shanghai
ENV PYTHONUNBUFFERED=1

ARG PIP_NO_CACHE_DIR=false

ADD ./ccl_torch.patch /tmp/

# TODO: decide if we want to patch torch-ccl

# Install python
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl wget git libunwind8-dev vim less && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    env DEBIAN_FRONTEND=noninteractive apt-get update && \
    # add-apt-repository requires gnupg, gpg-agent, software-properties-common
    apt-get install -y --no-install-recommends gnupg gpg-agent software-properties-common && \
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
    apt-get install -y --no-install-recommends libfabric-dev wrk libaio-dev numactl && \
    # If we do not install this compute-runtime, we will fail the build later
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
    pip install --pre --upgrade ipex-llm[xpu_2.6] --extra-index-url https://download.pytorch.org/whl/test/xpu && \
    mkdir /build && \
    cd /build && \
    git clone https://github.com/intel/torch-ccl.git && \
    cd torch-ccl && \
    git checkout ccl_torch2.5.0+xpu && \
    git submodule sync && \
    git submodule update --init --recursive && \
    git apply /tmp/ccl_torch.patch && \
    USE_SYSTEM_ONECCL=ON COMPUTE_BACKEND=dpcpp python setup.py bdist_wheel
    # File path: /build/torch-ccl/dist/oneccl_bind_pt-2.5.0+xpu-cp311-cp311-linux_x86_64.whl

FROM intel/oneapi-basekit:2025.0.1-0-devel-ubuntu22.04

COPY --from=build /build/torch-ccl/dist/oneccl_bind_pt-2.5.0+xpu-cp311-cp311-linux_x86_64.whl /opt/oneccl_bind_pt-2.5.0+xpu-cp311-cp311-linux_x86_64.whl

ARG http_proxy
ARG https_proxy

ENV TZ=Asia/Shanghai
ENV PYTHONUNBUFFERED=1
# To prevent RPC_TIMEOUT ERROR for the first request
ENV VLLM_RPC_TIMEOUT=100000


# Disable pip's cache behavior
ARG PIP_NO_CACHE_DIR=false

ADD ./gradio_web_server.patch  /tmp/gradio_web_server.patch

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl wget git libunwind8-dev vim less && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    env DEBIAN_FRONTEND=noninteractive apt-get update && \
    # add-apt-repository requires gnupg, gpg-agent, software-properties-common
    apt-get install -y --no-install-recommends gnupg gpg-agent software-properties-common kmod && \
    # Add Python 3.11 PPA repository
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install -y --no-install-recommends python3.11 git curl wget && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.11 /usr/bin/python3 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get install -y --no-install-recommends python3-pip python3.11-dev python3-wheel python3.11-distutils && \
    wget https://bootstrap.pypa.io/get-pip.py -O get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py && \
    pip install --upgrade requests argparse urllib3 && \
    pip install --pre --upgrade ipex-llm[xpu_2.6] --extra-index-url https://download.pytorch.org/whl/test/xpu && \
    pip install transformers_stream_generator einops tiktoken && \
    pip install --upgrade colorama && \
    # TODO: remove this code segment...
    git clone -b upgrade-vllm-0.6.6 https://github.com/gc-fu/BigDL.git ./ipex-llm && \
    # The following comment segment is used when building from source...
    cd ipex-llm && \
    # git fetch origin pull/12338/head:local_pr && \
    # git checkout local_pr && \
    pip uninstall -y ipex-llm && \
    cd python/llm && \
    python setup.py install && \
    cd ../../../ && \
    cp -r ./ipex-llm/python/llm/dev/benchmark/ ./benchmark && \
    cp -r ./ipex-llm/python/llm/example/GPU/HuggingFace/LLM ./examples && \
    cp -r ./ipex-llm/python/llm/example/GPU/vLLM-Serving/ ./vLLM-Serving && \
    # Install vllm dependencies
    pip install --upgrade fastapi && \
    pip install --upgrade "uvicorn[standard]" && \
    # TODO: uncomment this...
    # rm -rf ./ipex-llm && \
    # Install torch-ccl
    pip install /opt/oneccl_bind_pt-2.5.0+xpu-cp311-cp311-linux_x86_64.whl && \
    # install Internal oneccl
    cd /opt && \
    wget https://sourceforge.net/projects/oneccl-wks/files/2025.0.0.6.6-release/oneccl_wks_installer_2025.0.0.6.6.sh && \
    bash oneccl_wks_installer_2025.0.0.6.6.sh && \
    apt-get update && \
    apt-get install -y --no-install-recommends libfabric-dev wrk libaio-dev numactl && \
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
    dpkg -i *.deb && \
    mkdir -p /llm && \
    cd /llm && \
    rm -rf /tmp/neo && \
    # Install vllm
    git clone -b 0.6.6-pre https://github.com/analytics-zoo/vllm.git /llm/vllm && \
    cd /llm/vllm && \
    pip install setuptools-scm && \
    pip install --upgrade cmake && \
    VLLM_TARGET_DEVICE=xpu pip install --no-build-isolation -v /llm/vllm && \
    pip install mpi4py fastapi uvicorn openai && \
    pip install gradio==4.43.0 && \
    patch /usr/local/lib/python3.11/dist-packages/fastchat/serve/gradio_web_server.py < /tmp/gradio_web_server.patch && \
    pip install ray

# TODO: check whether these files works or not...
COPY ./vllm_online_benchmark.py                   /llm/
COPY ./vllm_offline_inference.py                  /llm/
COPY ./vllm_offline_inference_vision_language.py  /llm/
COPY ./payload-1024.lua                           /llm/
COPY ./start-vllm-service.sh                      /llm/
COPY ./benchmark_vllm_throughput.py               /llm/
COPY ./benchmark_vllm_latency.py                  /llm/
COPY ./start-pp_serving-service.sh                /llm/

WORKDIR /llm/
