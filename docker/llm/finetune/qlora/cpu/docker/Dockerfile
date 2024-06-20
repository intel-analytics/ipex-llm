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

RUN mkdir -p /ipex_llm/data && mkdir -p /ipex_llm/model && \
    # Install python 3.11.1
    apt-get update && \
    apt-get install -y curl wget gpg gpg-agent git gcc g++ make libunwind8-dev libbz2-dev zlib1g-dev libssl-dev libffi-dev xz-utils && \
    mkdir -p /opt/python && \
    cd /opt/python && \
    wget https://www.python.org/ftp/python/3.11.1/Python-3.11.1.tar.xz && \
    tar -xf Python-3.11.1.tar.xz && \
    cd Python-3.11.1 && \
    ./configure --enable-optimizations --with-zlib && \
    make altinstall && \
    # Create a symbolic link pointing to Python 3.11 at /usr/bin/python3
    ln -s /opt/python/Python-3.11.1/python /usr/bin/python3 && \
    # Create a symbolic link pointing to /usr/bin/python3 at /usr/bin/python
    ln -s /usr/bin/python3 /usr/bin/python && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --upgrade pip && \
    export PIP_DEFAULT_TIMEOUT=100 && \
    # install torch CPU version
    pip install --upgrade torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu && \
    # install CPU ipex-llm
    pip install --pre --upgrade ipex-llm[all] && \
    # install ipex and oneccl
    pip install https://intel-extension-for-pytorch.s3.amazonaws.com/ipex_stable/cpu/intel_extension_for_pytorch-2.1.0%2Bcpu-cp311-cp311-linux_x86_64.whl && \
    pip install oneccl_bind_pt -f https://developer.intel.com/ipex-whl-stable && \
    # install huggingface dependencies
    pip install datasets transformers==4.36.0 && \
    pip install fire peft==0.10.0 && \
    pip install bitsandbytes && \
    # get qlora example code
    cd /ipex_llm && \
    git clone https://github.com/intel-analytics/IPEX-LLM.git && \
    mv IPEX-LLM/python/llm/example/CPU/QLoRA-FineTuning/* . && \
    mkdir -p /GPU/LLM-Finetuning && \
    mv IPEX-LLM/python/llm/example/GPU/LLM-Finetuning/common /GPU/LLM-Finetuning/common && \
    rm -r IPEX-LLM

# for standalone
COPY ./start-qlora-finetuning-on-cpu.sh /ipex_llm/start-qlora-finetuning-on-cpu.sh
RUN chown -R mpiuser /ipex_llm

USER mpiuser

ENTRYPOINT ["/bin/bash"]
