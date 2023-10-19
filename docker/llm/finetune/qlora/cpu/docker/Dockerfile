FROM intel/oneapi-basekit:2023.2.1-devel-ubuntu22.04
ARG http_proxy
ARG https_proxy
ENV TZ=Asia/Shanghai
ARG PIP_NO_CACHE_DIR=false
ENV TRANSFORMERS_COMMIT_ID=95fe0f5

# retrive oneapi repo public key
RUN curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB | gpg --dearmor | tee /usr/share/keyrings/intel-oneapi-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/intel-oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main " > /etc/apt/sources.list.d/oneAPI.list && \
    # update dependencies
    apt-get update && \
    # install basic dependencies
    apt-get install -y curl wget git gnupg gpg-agent software-properties-common libunwind8-dev vim less && \
    # install python 3.9
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    env DEBIAN_FRONTEND=noninteractive apt-get update && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install -y python3.9 && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.9 /usr/bin/python3 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get install -y python3-pip python3.9-dev python3-wheel python3.9-distutils && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    # install torch and ipex and oneccl to reduce bigdl-llm size
    pip install --upgrade torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install intel_extension_for_pytorch==2.0.100 && \
    pip install oneccl_bind_pt -f https://developer.intel.com/ipex-whl-stable && \
    # install CPU bigdl-llm
    pip install --pre --upgrade bigdl-llm[cpu] && \
    # install huggingface dependencies
    pip install git+https://github.com/huggingface/transformers.git@${TRANSFORMERS_COMMIT_ID} && \
    pip install peft==0.5.0 datasets && \
    wget https://raw.githubusercontent.com/intel-analytics/BigDL/main/python/llm/example/GPU/QLoRA-FineTuning/qlora_finetuning.py

ADD ./qlora_finetuning_cpu.py /qlora_finetuning_cpu.py
ADD ./start-qlora-finetuning-on-cpu.sh /start-qlora-finetuning-on-cpu.sh
