FROM ubuntu:20.04 as key-getter
ARG http_proxy
ARG https_proxy

RUN apt-get update && \
    apt-get install -y curl gpg && \
    curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB | gpg --dearmor | tee /root/intel-oneapi-archive-keyring.gpg

FROM mpioperator/intel as builder

ARG http_proxy
ARG https_proxy
ENV TZ=Asia/Shanghai
ARG PIP_NO_CACHE_DIR=false
ENV TRANSFORMERS_COMMIT_ID=95fe0f5

# add public key
COPY --from=key-getter /root/intel-oneapi-archive-keyring.gpg /usr/share/keyrings/intel-oneapi-archive-keyring.gpg
RUN echo "deb [signed-by=/usr/share/keyrings/intel-oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main " > /etc/apt/sources.list.d/oneAPI.list

RUN mkdir -p /bigdl/data && mkdir -p /bigdl/model && \
    # install pytorch 2.1.0
    apt-get update && \
    apt-get install -y --no-install-recommends python3-pip python3.9-dev python3-wheel python3.9-distutils git software-properties-common && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install --upgrade pip && \
    export PIP_DEFAULT_TIMEOUT=100 && \
    pip install --upgrade torch==2.1.0 && \
    # install CPU bigdl-llm
    pip3 install --pre --upgrade bigdl-llm[all] && \
    # install ipex and oneccl
    pip install https://intel-extension-for-pytorch.s3.amazonaws.com/ipex_stable/cpu/intel_extension_for_pytorch-2.1.0%2Bcpu-cp39-cp39-linux_x86_64.whl && \
    pip install oneccl_bind_pt -f https://developer.intel.com/ipex-whl-stable && \
    # install huggingface dependencies
    pip install datasets transformers==4.35.0 && \
    pip install fire peft==0.5.0 && \
    pip install accelerate==0.23.0 && \
    pip install bitsandbytes && \
    # install basic dependencies
    apt-get update && apt-get install -y curl wget gpg gpg-agent software-properties-common libunwind8-dev && \
    # get qlora example code
    ln -s /usr/bin/python3 /usr/bin/python && \
    cd /bigdl && \
    git clone https://github.com/intel-analytics/BigDL.git && \
    mv BigDL/python/llm/example/CPU/QLoRA-FineTuning/* . && \
    mkdir -p /GPU/LLM-Finetuning && \
    mv BigDL/python/llm/example/GPU/LLM-Finetuning/common /GPU/LLM-Finetuning/common && \
    rm -r BigDL && \
    chown -R mpiuser /bigdl

# for standalone
COPY ./start-qlora-finetuning-on-cpu.sh /bigdl/start-qlora-finetuning-on-cpu.sh

USER mpiuser

ENTRYPOINT ["/bin/bash"]
