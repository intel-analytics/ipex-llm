FROM ghcr.io/huggingface/text-generation-inference:latest-intel as tgi
FROM intel/oneapi-basekit:2024.0.1-devel-ubuntu22.04

ARG http_proxy
ARG https_proxy

ENV TZ=Asia/Shanghai
ENV PYTHONUNBUFFERED=1
ENV USE_XETLA=OFF
ENV SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1

COPY  --from=tgi /usr/src/proto /usr/src/proto
COPY  --from=tgi /usr/src/server /usr/src/server
COPY  --from=tgi /usr/local/bin/text-generation-benchmark /usr/local/bin/text-generation-benchmark
COPY  --from=tgi /usr/local/bin/text-generation-router /usr/local/bin/text-generation-router
COPY  --from=tgi /usr/local/bin/text-generation-launcher /usr/local/bin/text-generation-launcher

COPY chat.py /llm/chat.py
COPY benchmark.sh /llm/benchmark.sh

# Text Generation Inference base env
ENV HUGGINGFACE_HUB_CACHE=/data \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PORT=80

# avoid ipex auto import error
ENV BIGDL_IMPORT_IPEX=false
ENV BIGDL_QUANTIZE_KV_CACHE=0

# Disable pip's cache behavior
ARG PIP_NO_CACHE_DIR=false

# patch files
COPY ./causal_lm.patch        /llm/
COPY ./server.patch           /llm/
RUN patch -R /usr/src/server/text_generation_server/server.py /llm/server.patch && \
    patch -R /usr/src/server/text_generation_server/models/causal_lm.py /llm/causal_lm.patch


RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/intel-oneapi-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/intel-oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main " | tee /etc/apt/sources.list.d/oneAPI.list && \
    chmod 644 /usr/share/keyrings/intel-oneapi-archive-keyring.gpg && \
    rm /etc/apt/sources.list.d/intel-graphics.list && \
    wget -O- https://repositories.intel.com/graphics/intel-graphics.key | gpg --dearmor | tee /usr/share/keyrings/intel-graphics.gpg > /dev/null && \
    echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/graphics/ubuntu jammy arc" | tee /etc/apt/sources.list.d/intel.gpu.jammy.list && \
    chmod 644 /usr/share/keyrings/intel-graphics.gpg && \
    apt-get update && \
    apt-get install -y --no-install-recommends curl wget git libunwind8-dev vim less ninja-build pciutils && \
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
    # install server dependencies
    cd /usr/src/server && \
    make gen-server && \
    pip install -r requirements_intel.txt && \
    pip install ".[accelerate, peft, outlines]" --no-cache-dir && \
    pip install --upgrade requests argparse urllib3 && \
    # install ipex-llm libs
    pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/ && \
    # Fix Trivy CVE Issues
    pip install transformers==4.37.0 && \
    pip install transformers_stream_generator einops tiktoken && \
    # Install opencl-related repos
    apt-get update && \
    apt-get install -y --no-install-recommends intel-opencl-icd=23.35.27191.42-775~22.04 intel-level-zero-gpu=1.3.27191.42-775~22.04 level-zero=1.14.0-744~22.04 && \
    # Install related libary of chat.py
    pip install --upgrade colorama && \
    # Download all-in-one benchmark and examples
    git clone https://github.com/intel-analytics/ipex-llm && \
    cp -r ./ipex-llm/python/llm/dev/benchmark/ ./benchmark && \
    cp -r ./ipex-llm/python/llm/example/GPU/HF-Transformers-AutoModels/Model ./examples && \
    # Install related library of benchmarking
    pip install pandas omegaconf && \
    chmod +x /llm/benchmark.sh && \
    ln -s /usr/local/lib/python3.11/dist-packages/ipex_llm/libs/libtcmalloc.so /lib/libtcmalloc.so && \
    rm -rf ./ipex-llm


WORKDIR /llm/
