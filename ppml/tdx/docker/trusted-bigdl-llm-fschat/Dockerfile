FROM ubuntu:20.04

ARG http_proxy
ARG https_proxy
ARG TINI_VERSION=v0.18.0

RUN mkdir -p /ppml/


ADD ./fschat.diff /opt/fschat.diff
ADD ./entrypoint.sh /opt/entrypoint.sh
ADD ./attestation_cli.py /opt/attestation_cli.py
ADD ./attestation.sh /opt/attestation.sh
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /sbin/tini


# Install PYTHON 3.9
RUN env DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt install software-properties-common libunwind8-dev vim less -y && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install -y python3.9 git curl wget && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.9 /usr/bin/python3 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get install -y python3-pip python3.9-dev python3-wheel && \
    pip3 install --no-cache --upgrade requests argparse cryptography==3.3.2 urllib3 && \
    pip3 install --pre --upgrade bigdl-llm[all] && \
    cd /ppml/ && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    # Install FastChat from source requires PEP 660 support
    python3 get-pip.py && \
    rm get-pip.py && \
    # Install FastChat
    git clone https://github.com/analytics-zoo/FastChat.git && \
    cd /ppml/FastChat && \
    pip install -e . && \
    # Pin gradio version because this error:https://github.com/lm-sys/FastChat/issues/1925
    pip install --pre --upgrade gradio==3.36.1 && \
    pip install --pre --upgrade bigdl-nano && \
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


ENV PYTHONPATH /usr/lib/python3.9:/usr/lib/python3.9/lib-dynload:/usr/local/lib/python3.9/dist-packages:/usr/lib/python3/dist-packages

WORKDIR /ppml
ENTRYPOINT [ "/opt/entrypoint.sh" ]
