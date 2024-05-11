FROM intel/oneapi-basekit:2024.0.1-devel-ubuntu22.04

ARG http_proxy
ARG https_proxy

ENV TZ=Asia/Shanghai
ENV PYTHONUNBUFFERED=1

# Disable pip's cache behavior
ARG PIP_NO_CACHE_DIR=false

RUN curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB | gpg --dearmor | tee /usr/share/keyrings/intel-oneapi-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/intel-oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main " > /etc/apt/sources.list.d/oneAPI.list && \
    apt-get update && \
    apt-get install -y curl wget git gnupg gpg-agent sudo && \
    wget -qO - https://repositories.intel.com/graphics/intel-graphics.key | gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg && \
    echo 'deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/graphics/ubuntu jammy arc' | tee /etc/apt/sources.list.d/intel.gpu.jammy.list && \
    rm /etc/apt/sources.list.d/intel-graphics.list && \
    # Install PYTHON 3.11 and IPEX-LLM[xpu]
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    env DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt install software-properties-common libunwind8-dev vim less -y && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install -y python3.11 git curl wget && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.11 /usr/bin/python3 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get install -y python3-pip python3.11-dev python3-wheel python3.11-distutils && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py && \
    pip install --upgrade requests argparse urllib3 && \
    pip install --pre --upgrade ipex-llm[cpp] && \
    # Fix Trivy CVE Issues
    pip install transformers==4.36.2 && \
    pip install transformers_stream_generator einops tiktoken && \
    # Install opencl-related repos
    apt-get update && \
    apt-get install -y intel-opencl-icd intel-level-zero-gpu=1.3.26241.33-647~22.04 level-zero level-zero-dev --allow-downgrades && \
    # install nodejs and npm and get webui
    apt purge nodejs -y && \
    apt purge libnode-dev -y && \
    apt autoremove -y && \
    apt clean -y && \
    curl -sL https://deb.nodesource.com/setup_18.x | sudo -E bash - && \
    apt install -y nodejs && \
    mkdir -p /llm/scripts && cd /llm && \
    git clone https://github.com/open-webui/open-webui.git && \
    cd /llm/open-webui/ && \
    cp -RPp .env.example .env && \
    # Build frontend
    npm i && \
    npm run build && \
    # Install Dependencies
    cd ./backend && \
    # remove blinker to avoid error
    find /usr/lib/python3/dist-packages/ -name 'blinker*' -exec rm -rf {} + && \
    pip install -r requirements.txt -U

COPY ./start-llama-cpp.sh /llm/scripts/start-llama-cpp.sh
COPY ./start-ollama.sh /llm/scripts/start-ollama.sh
COPY ./start-open-webui.sh /llm/scripts/start-open-webui.sh

WORKDIR /llm/
