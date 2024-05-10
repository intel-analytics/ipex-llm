FROM intel/oneapi-basekit:2024.0.1-devel-ubuntu22.04
ARG http_proxy
ARG https_proxy
ENV TZ=Asia/Shanghai
ARG PIP_NO_CACHE_DIR=false

# retrive oneapi repo public key
RUN curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB | gpg --dearmor | tee /usr/share/keyrings/intel-oneapi-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/intel-oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main " > /etc/apt/sources.list.d/oneAPI.list && \
    # retrive intel gpu driver repo public key 
    wget -qO - https://repositories.intel.com/graphics/intel-graphics.key | gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg && \
    echo 'deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/graphics/ubuntu jammy arc' | tee /etc/apt/sources.list.d/intel.gpu.jammy.list && \
    rm /etc/apt/sources.list.d/intel-graphics.list && \
    # update dependencies
    apt-get update && \
    # install basic dependencies
    apt-get install -y curl wget git gnupg gpg-agent software-properties-common libunwind8-dev vim less && \
    # install Intel GPU driver
    apt-get install -y intel-opencl-icd intel-level-zero-gpu level-zero level-zero-dev --allow-downgrades && \
    # install python 3.11
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    env DEBIAN_FRONTEND=noninteractive apt-get update && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install -y python3.11 && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.11 /usr/bin/python3 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get install -y python3-pip python3.11-dev python3-wheel python3.11-distutils && \
    # remove apt cache
    rm -rf /var/lib/apt/lists/* && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    # install XPU ipex-llm
    pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/ && \
    # apt-get remove -y python3-blinker && \
    # prepare finetune code and scripts
    git clone https://github.com/intel-analytics/IPEX-LLM.git && \
    mv IPEX-LLM/python/llm/example/GPU/LLM-Finetuning /LLM-Finetuning && \
    rm -rf IPEX-LLM && \
    # install axolotl
    git clone https://github.com/OpenAccess-AI-Collective/axolotl && \
    cd axolotl && git checkout v0.4.0 && \
    mv /LLM-Finetuning/axolotl/requirements-xpu.txt requirements.txt && \
    pip install -e . --ignore-installed blinker && \
    # install transformers & peft dependencies
    pip install transformers==4.36.0 && \
    pip install peft==0.10.0 datasets accelerate==0.23.0 && \
    pip install bitsandbytes scipy fire && \
    # Prepare accelerate config
    mkdir -p /root/.cache/huggingface/accelerate && \
    mv /LLM-Finetuning/axolotl/default_config.yaml /root/.cache/huggingface/accelerate/

COPY ./start-qlora-finetuning-on-xpu.sh /start-qlora-finetuning-on-xpu.sh
