ARG BASE_IMAGE_NAME=intelanalytics/bigdl-llm-serving-cpu
ARG BASE_IMAGE_TAG=2.5.0-SNAPSHOT

FROM $BASE_IMAGE_NAME:$BASE_IMAGE_TAG

ARG http_proxy
ARG https_proxy

RUN mkdir -p /ppml/


COPY ./entrypoint.sh /opt/entrypoint.sh
COPY ./attestation.sh /opt/attestation.sh


# Install bigdl-ppml
RUN cd /root && \
    git clone https://github.com/intel-analytics/BigDL.git && \
    mv /root/BigDL/python/ppml/src/bigdl/ppml/ /usr/local/lib/python3.9/dist-packages/bigdl/ && \
    cp /usr/local/lib/python3.9/dist-packages/bigdl/ppml/attestation/attestation_cli.py /opt/ && \
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
    echo 'deb [signed-by=/etc/apt/keyrings/intel-sgx-keyring.asc arch=amd64] https://download.01.org/intel-sgx/sgx_repo/ubuntu jammy main' | tee /etc/apt/sources.list.d/intel-sgx.list && \
    rm -f /etc/apt/keyrings/intel-sgx-keyring.asc && \
    wget https://download.01.org/intel-sgx/sgx_repo/ubuntu/intel-sgx-deb.key && \
    cat intel-sgx-deb.key | tee /etc/apt/keyrings/intel-sgx-keyring.asc > /dev/null && \
    apt-get update && \
    apt-get install -y libsgx-enclave-common-dev  libsgx-ae-qe3 libsgx-ae-qve libsgx-urts libsgx-dcap-ql libsgx-dcap-default-qpl libsgx-dcap-quote-verify-dev libsgx-dcap-ql-dev libsgx-dcap-default-qpl-dev libsgx-quote-ex-dev libsgx-uae-service libsgx-ra-network libsgx-ra-uefi libtdx-attest libtdx-attest-dev && \
    cd /ppml && \
    chmod +x /opt/entrypoint.sh && \
    chmod +x /opt/attestation.sh 


ENV PYTHONPATH /usr/lib/python3.9:/usr/lib/python3.9/lib-dynload:/usr/local/lib/python3.9/dist-packages:/usr/lib/python3/dist-packages

WORKDIR /ppml
ENTRYPOINT [ "/opt/entrypoint.sh" ]
