FROM intel/oneapi-basekit:2023.2.1-devel-ubuntu22.04
ARG http_proxy
ARG https_proxy
ENV TZ=Asia/Shanghai
ENV PIP_NO_CACHE_DIR=false

RUN apt-get update && \
    # install Intel GPU driver
    apt-get install -y curl wget git gnupg gpg-agent && \
    wget -qO - https://repositories.intel.com/graphics/intel-graphics.key | gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg && \
    echo 'deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/graphics/ubuntu jammy arc' | tee /etc/apt/sources.list.d/intel.gpu.jammy.list && \
    rm /etc/apt/sources.list.d/intel-graphics.list && \
    # install python 3.9
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    env DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt install software-properties-common libunwind8-dev vim less -y && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install -y python3.9 git curl wget && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.9 /usr/bin/python3 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get install -y python3-pip python3.9-dev python3-wheel python3.9-distutils && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    # install XPU bigdl-llm
    pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu && \
    # install huggingface dependencies
    pip install git+https://github.com/huggingface/transformers.git@95fe0f5 && \
    pip install peft==0.5.0 datasets

ADD ./qlora_finetune_xpu.py /qlora_finetune_xpu.py
ADD ./start-qlora-finetuning-on-xpu.sh /start-qlora-finetuning-on-xpu.sh
