FROM ubuntu:20.04 as key-getter
ARG http_proxy
ARG https_proxy

RUN env DEBIAN_FRONTEND=noninteractive apt-get update && \
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
    env DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y --no-install-recommends python3-pip python3.9-dev python3-wheel python3.9-distutils git software-properties-common && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install --upgrade pip && \
    export PIP_DEFAULT_TIMEOUT=100 && \
    pip install --upgrade torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu && \
    # install CPU bigdl-llm
    pip3 install --pre --upgrade bigdl-llm[all] && \
    # install ipex and oneccl
    pip install intel_extension_for_pytorch==2.0.100 && \
    pip install oneccl_bind_pt -f https://developer.intel.com/ipex-whl-stable && \
    # install huggingface dependencies
    pip install datasets && \
    # Fix Trivy CVE Issues
    pip install transformers==4.36.2 Jinja2==3.1.3 && \
    pip install fire peft==0.5.0 && \
    pip install accelerate==0.23.0 && \
    # install basic dependencies
    env DEBIAN_FRONTEND=noninteractive apt-get install -y curl wget gpg gpg-agent software-properties-common libunwind8-dev && \
    # get qlora example code
    ln -s /usr/bin/python3 /usr/bin/python && \
    cd /bigdl && \
    git clone https://github.com/intel-analytics/BigDL.git && \
    mv BigDL/python/llm/example/CPU/QLoRA-FineTuning/* . && \
    rm -r BigDL && \
    chown -R mpiuser /bigdl

# for docker directly run example
COPY ./start-qlora-finetuning-on-cpu.sh /bigdl/start-qlora-finetuning-on-cpu.sh
# for k8s
COPY ./bigdl-qlora-finetuing-entrypoint.sh /bigdl/bigdl-qlora-finetuing-entrypoint.sh

USER mpiuser

ENTRYPOINT ["/bin/bash"]
