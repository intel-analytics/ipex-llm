FROM intel/oneapi-basekit:2024.1.1-devel-ubuntu22.04

ARG http_proxy
ARG https_proxy

ENV TZ=Asia/Shanghai
ENV PYTHONUNBUFFERED=1


# Disable pip's cache behavior
ARG PIP_NO_CACHE_DIR=false
ADD ./gradio_web_server.patch  /tmp/gradio_web_server.patch

RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/intel-oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/intel-oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main " | tee /etc/apt/sources.list.d/oneAPI.list && \
    chmod 644 /usr/share/keyrings/intel-oneapi-archive-keyring.gpg && \
    rm /etc/apt/sources.list.d/intel-graphics.list && \
    wget -O- https://repositories.intel.com/graphics/intel-graphics.key | gpg --dearmor | tee /usr/share/keyrings/intel-graphics.gpg > /dev/null && \
    echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/graphics/ubuntu jammy arc" | tee /etc/apt/sources.list.d/intel.gpu.jammy.list && \
    chmod 644 /usr/share/keyrings/intel-graphics.gpg && \
    apt-get update && \
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
    pip install --pre --upgrade ipex-llm[xpu,serving] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/ && \
    pip install transformers_stream_generator einops tiktoken && \
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
    rm -rf ./ipex-llm && \
    # Install torch-ccl
    cd /tmp/ && \
    pip install torch==2.1.0.post2 torchvision==0.16.0.post2 torchaudio==2.1.0.post2 intel-extension-for-pytorch==2.1.30.post0 --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/ && \
    # Internal oneccl
    wget https://sourceforge.net/projects/oneccl-wks/files/2024.0.0.5.1-release/oneccl_wks_installer_2024.0.0.5.1.sh && \
    bash oneccl_wks_installer_2024.0.0.5.1.sh && \
    git clone https://github.com/intel/torch-ccl -b v2.1.300+xpu && \
    cd torch-ccl && \
    USE_SYSTEM_ONECCL=ON COMPUTE_BACKEND=dpcpp python setup.py install && \
    apt-get update && \
    apt-get install -y --no-install-recommends libfabric-dev wrk libaio-dev numactl && \
    # apt-get install -y intel-opencl-icd intel-level-zero-gpu=1.3.26241.33-647~22.04 level-zero level-zero-dev --allow-downgrades && \
    mkdir -p /tmp/neo && \
    cd /tmp/neo && \
    wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.15136.4/intel-igc-core_1.0.15136.4_amd64.deb && \
    wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.15136.4/intel-igc-opencl_1.0.15136.4_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/23.35.27191.9/intel-level-zero-gpu-dbgsym_1.3.27191.9_amd64.ddeb && \
    wget https://github.com/intel/compute-runtime/releases/download/23.35.27191.9/intel-level-zero-gpu_1.3.27191.9_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/23.35.27191.9/intel-opencl-icd-dbgsym_23.35.27191.9_amd64.ddeb && \
    wget https://github.com/intel/compute-runtime/releases/download/23.35.27191.9/intel-opencl-icd_23.35.27191.9_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/23.35.27191.9/libigdgmm12_22.3.11.ci17747749_amd64.deb && \
    dpkg -i *.deb && \
    rm -rf /tmp/neo && \
    mkdir -p /llm && \
    cd /llm && \
    git clone -b 0.5.4 https://github.com/analytics-zoo/vllm.git /llm/vllm && \
    cd /llm/vllm && \
    pip install -r /llm/vllm/requirements-xpu.txt && \
    VLLM_TARGET_DEVICE=xpu python setup.py install && \
    pip install mpi4py fastapi uvicorn openai && \
    pip install gradio==4.43.0 && \
    pip install transformers==4.44.2 && \
    # patch /usr/local/lib/python3.11/dist-packages/fastchat/serve/gradio_web_server.py < /tmp/gradio_web_server.patch && \
    pip install ray && \
    patch /usr/local/lib/python3.11/dist-packages/fastchat/serve/gradio_web_server.py < /tmp/gradio_web_server.patch

COPY ./vllm_online_benchmark.py        /llm/
COPY ./vllm_offline_inference.py       /llm/
COPY ./payload-1024.lua                /llm/
COPY ./start-vllm-service.sh           /llm/
COPY ./benchmark_vllm_throughput.py   /llm/
COPY ./start-fastchat-service.sh       /llm/
COPY ./start-pp_serving-service.sh       /llm/
COPY ./start-lightweight_serving-service.sh       /llm/

ENV LD_LIBRARY_PATH /usr/local/lib/python3.11/dist-packages/intel_extension_for_pytorch/lib/:/opt/intel/oneapi/tbb/2021.12/env/../lib/intel64/gcc4.8:/opt/intel/oneapi/mpi/2021.12/opt/mpi/libfabric/lib:/opt/intel/oneapi/mpi/2021.12/lib:/opt/intel/oneapi/mkl/2024.1/lib:/opt/intel/oneapi/ippcp/2021.11/lib/:/opt/intel/oneapi/ipp/2021.11/lib:/opt/intel/oneapi/dpl/2022.5/lib:/opt/intel/oneapi/dnnl/2024.1/lib:/opt/intel/oneapi/debugger/2024.1/opt/debugger/lib:/opt/intel/oneapi/dal/2024.2/lib:/opt/intel/oneapi/compiler/2024.1/opt/oclfpga/host/linux64/lib:/opt/intel/oneapi/compiler/2024.1/opt/compiler/lib:/opt/intel/oneapi/compiler/2024.1/lib:/opt/intel/oneapi/ccl/2021.12/lib/

WORKDIR /llm/
