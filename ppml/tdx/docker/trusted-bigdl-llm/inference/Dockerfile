FROM ubuntu:20.04

ARG http_proxy
ARG https_proxy
ARG TINI_VERSION=v0.18.0

RUN mkdir -p /ppml/


COPY ./fschat.diff /opt/fschat.diff
COPY ./requirements.txt /opt/requirements.txt
COPY ./entrypoint.sh /opt/entrypoint.sh
COPY ./attestation_cli.py /opt/attestation_cli.py
COPY ./attestation.sh /opt/attestation.sh
COPY ./llama_cpp.patch /opt/llama_cpp.patch
COPY ./llama.patch /opt/llama.patch
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /sbin/tini


# Install PYTHON 3.9
RUN env DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt install software-properties-common libunwind8-dev vim less -y && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install -y python3.9 git curl wget libcrypto++6 libcrypto++-dev libcrypto++-utils && \
    apt-get install -y git-lfs && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.9 /usr/bin/python3 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get install -y python3-pip python3.9-dev python3-wheel && \
    pip3 install --no-cache requests argparse cryptography==3.3.2 urllib3 && \
    # Install bigdl-llm and pin its version as we have made certain changes to llm.cpp
    pip3 install --pre --upgrade bigdl-llm[all]==2.4.0b20230710 && \
    patch /usr/local/lib/python3.9/dist-packages/bigdl/llm/ggml/model/llama/llama.py /opt/llama.patch && \
    patch /usr/local/lib/python3.9/dist-packages/bigdl/llm/ggml/model/llama/llama_cpp.py /opt/llama_cpp.patch && \
    cd /usr/local/lib/python3.9/dist-packages/bigdl/llm/libs && \
    rm *avx512* && \
    rm quantize* && \
    curl "https://sourceforge.net/projects/analytics-zoo/rss?path=/ppml-llm-encrypted" | grep "<link>.*</link>" | sed 's|<link>||;s|</link>||' | while read url; do url=`echo $url | sed 's|/download$||'`; wget $url ; done && \
    rm index.html && \
    chmod +x * && \
    # FastChat related dependencies
    pip3 install fschat==0.2.18 && \
    # We would also patch fastchat
    patch /usr/local/lib/python3.9/dist-packages/fastchat/model/model_adapter.py /opt/fschat.diff && \
    pip3 install --pre --upgrade gradio==3.36.1 && \
    pip3 install --pre --upgrade bigdl-nano && \
# Install dependencies for bigdl-llm-demo
    pip3 install -r /opt/requirements.txt && \
    pip3 install typing-inspect==0.8.0 typing_extensions==4.5.0 && \
    pip3 install ray==2.3.1 && \
    pip3 install transformers==4.36.0 && \
    pip3 install sentence-transformers --no-deps && \
    pip3 install cmake ninja cpuid && \
    pip3 install scikit-build && \
    pip3 install cchardet && \
    pip3 install cryptography && \
    pip3 install pytorch_lightning==1.6.4 && \
    pip3 uninstall -y torchvision && \
    pip3 install https://download.pytorch.org/whl/nightly/cpu-cxx11-abi/torch-2.0.0.dev20230228%2Bcpu.cxx11.abi-cp39-cp39-linux_x86_64.whl && \
# Install bigdl-ppml
    cd /root && \
    git clone https://github.com/intel-analytics/BigDL.git && \
    mv /root/BigDL/python/ppml/src/bigdl/ppml/ /usr/local/lib/python3.9/dist-packages/bigdl/ && \
    rm -rf /root/BigDL && \
# Remote attestation dependencies
    mkdir -p /opt/intel/ && \
    cd /opt/intel && \
    wget https://download.01.org/intel-sgx/sgx-dcap/1.16/linux/distro/ubuntu20.04-server/sgx_linux_x64_sdk_2.19.100.3.bin && \
    chmod a+x ./sgx_linux_x64_sdk_2.19.100.3.bin && \
    printf "no\n/opt/intel\n"|./sgx_linux_x64_sdk_2.19.100.3.bin && \
    . /opt/intel/sgxsdk/environment && \
    cd /opt/intel && \
    wget https://download.01.org/intel-sgx/sgx-dcap/1.16/linux/distro/ubuntu20.04-server/sgx_debian_local_repo.tgz && \
    tar xzf sgx_debian_local_repo.tgz && \
    echo 'deb [trusted=yes arch=amd64] file:///opt/intel/sgx_debian_local_repo focal main' | tee /etc/apt/sources.list.d/intel-sgx.list && \
    wget -qO - https://download.01.org/intel-sgx/sgx_repo/ubuntu/intel-sgx-deb.key | apt-key add - && \
    apt-get update && \
    apt-get install -y libsgx-enclave-common-dev  libsgx-ae-qe3 libsgx-ae-qve libsgx-urts libsgx-dcap-ql libsgx-dcap-default-qpl libsgx-dcap-quote-verify-dev libsgx-dcap-ql-dev libsgx-dcap-default-qpl-dev libsgx-quote-ex-dev libsgx-uae-service libsgx-ra-network libsgx-ra-uefi libtdx-attest libtdx-attest-dev && \
    chmod +x /opt/entrypoint.sh && \
    chmod +x /opt/attestation.sh && \
    chmod +x /sbin/tini && \
    cp /sbin/tini /usr/bin/tini


# To boost fschat performance, do the following:
# source bigdl-nano-init -t
# export KMP_AFFINITY=granularity=fine,compact
# export OMP_NUM_THREADS=#phy cores


ENV PYTHONPATH /usr/lib/python3.9:/usr/lib/python3.9/lib-dynload:/usr/local/lib/python3.9/dist-packages:/usr/lib/python3/dist-packages

WORKDIR /ppml
ENTRYPOINT [ "/opt/entrypoint.sh" ]
