FROM intelanalytics/ipex-llm-serving-xpu:latest as build

ARG http_proxy
ARG https_proxy

ADD ./oneccl-binding.patch  /tmp/oneccl-binding.patch

RUN cd /tmp/ && \
    pip install --upgrade setuptools wheel twine && \
    pip install "setuptools<70.0.0" && \
    git clone https://github.com/intel/torch-ccl -b v2.1.300+xpu && \
    cd torch-ccl && \
    patch -p1 < /tmp/oneccl-binding.patch && \
    git submodule sync && \
    git submodule update --init --recursive && \
    USE_SYSTEM_ONECCL=ON COMPUTE_BACKEND=dpcpp python setup.py install sdist bdist_wheel && \
    mv /tmp/torch-ccl/dist/oneccl_bind_pt-2.1.300+xpu-cp311-cp311-linux_x86_64.whl /tmp/


FROM intel/oneapi-basekit:2024.1.1-devel-ubuntu22.04

ARG http_proxy
ARG https_proxy

# Disable pip's cache behavior
ARG PIP_NO_CACHE_DIR=false
COPY --from=build /tmp/oneccl_bind_pt-2.1.100+xpu-cp311-cp311-linux_x86_64.whl /tmp/
ADD ./gradio_web_server.patch /tmp/gradio_web_server.patch

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
    pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/ && \
    # Fix Trivy CVE Issues
    pip install transformers==4.36.2 && \
    pip install transformers_stream_generator einops tiktoken && \
    # # Install opencl-related repos
    # apt-get update && \
    # apt-get install -y --no-install-recommends intel-opencl-icd=23.35.27191.42-775~22.04 intel-level-zero-gpu=1.3.27191.42-775~22.04 level-zero=1.14.0-744~22.04 && \
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
    cp -r ./ipex-llm/python/llm/example/GPU/vLLM-Serving/ ./vLLM-Serving 

# Install Serving Dependencies
# Install ipex-llm[serving] only will update ipex_llm source code without updating
# bigdl-core-xe, which will lead to problems
RUN apt-get update && \
    apt-get install -y --no-install-recommends libfabric-dev wrk libaio-dev && \
    mkdir -p /llm/neo && \
    cd /llm/neo && \
    wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.15136.4/intel-igc-core_1.0.15136.4_amd64.deb && \
    wget https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.15136.4/intel-igc-opencl_1.0.15136.4_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/23.35.27191.9/intel-level-zero-gpu-dbgsym_1.3.27191.9_amd64.ddeb && \
    wget https://github.com/intel/compute-runtime/releases/download/23.35.27191.9/intel-level-zero-gpu_1.3.27191.9_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/23.35.27191.9/intel-opencl-icd-dbgsym_23.35.27191.9_amd64.ddeb && \
    wget https://github.com/intel/compute-runtime/releases/download/23.35.27191.9/intel-opencl-icd_23.35.27191.9_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/23.35.27191.9/libigdgmm12_22.3.11.ci17747749_amd64.deb && \
    dpkg -i *.deb && \
    pip install --pre --upgrade ipex-llm[xpu,serving] && \
    pip install transformers==4.37.0 gradio==4.19.2 && \
    # Use ipex-vllm-mainline
    git clone -b vllm_202411_0807 https://github.com/xiangyuT/ipex-llm.git /llm/ipex-llm && \
    cp /llm/ipex-llm/python/llm/src/ipex_llm/transformers/convert.py /usr/local/lib/python3.11/dist-packages/ipex_llm/transformers/convert.py && \
    cp /llm/ipex-llm/python/llm/src/ipex_llm/transformers/low_bit_linear.py /usr/local/lib/python3.11/dist-packages/ipex_llm/transformers/low_bit_linear.py && \
    rm -rf /usr/local/lib/python3.11/dist-packages/ipex_llm/vllm && \
    cp -r /llm/ipex-llm/python/llm/src/ipex_llm/vllm /usr/local/lib/python3.11/dist-packages/ipex_llm/ && \
    # install ipex 2.1.30
    python -m pip install torch==2.1.0.post2 torchvision==0.16.0.post2 torchaudio==2.1.0.post2 intel-extension-for-pytorch==2.1.30.post0 oneccl_bind_pt==2.1.300+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/ && \
    python -m pip install setuptools==69.5.1 numpy==1.26.4 && \
    # Install vLLM-v2 dependencies 
    git clone -b xiangyu_test_202411_0806 https://github.com/analytics-zoo/vllm.git /llm/vllm && \
    pip install -r /llm/vllm/requirements-xpu.txt && \
    pip install --no-deps xformers && \
    cd /llm/vllm && \
    VLLM_TARGET_DEVICE=xpu python setup.py install && \
    pip install outlines==0.0.34 --no-deps && \
    pip install interegular cloudpickle diskcache joblib lark nest-asyncio numba scipy && \
    # For Qwen series models support
    pip install transformers_stream_generator einops tiktoken && \
    # For pipeline serving support
    pip install mpi4py fastapi uvicorn openai && \
    # for gradio web UI
    pip install gradio && \
    # Install internal oneccl && \
    cd /tmp/ && \
    wget https://sourceforge.net/projects/oneccl-wks/files/oneccl_wks_installer_2024.0.0.2.sh && \
    bash oneccl_wks_installer_2024.0.0.2.sh && \
    pip uninstall -y oneccl_bind_pt && \
    pip install /tmp/oneccl_bind_pt-2.1.300+xpu-cp311-cp311-linux_x86_64.whl && \ 
    rm /tmp/oneccl_bind_pt-2.1.300+xpu-cp311-cp311-linux_x86_64.whl && \
    patch /usr/local/lib/python3.11/dist-packages/fastchat/serve/gradio_web_server.py < /tmp/gradio_web_server.patch

COPY ./vllm_online_benchmark.py        /llm/
COPY ./vllm_offline_inference.py       /llm/
COPY ./payload-1024.lua                /llm/
COPY ./start-vllm-service.sh           /llm/
COPY ./benchmark_vllm_throughput.py   /llm/
COPY ./start-fastchat-service.sh       /llm/
COPY ./start-pp_serving-service.sh       /llm/
COPY ./start-lightweight_serving-service.sh       /llm/


WORKDIR /llm/
