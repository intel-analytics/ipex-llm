FROM intelanalytics/bigdl-lora-finetuning:2.4.0-SNAPSHOT

# install DCAP
RUN apt-get update && \
    apt install -y build-essential apt-utils wget git sudo vim && \
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
    env DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt install -y libsgx-enclave-common-dev libsgx-qe3-logic libsgx-pce-logic libsgx-ae-qe3 libsgx-ae-qve libsgx-urts libsgx-dcap-ql libsgx-dcap-default-qpl libsgx-dcap-quote-verify-dev libsgx-dcap-ql-dev libsgx-dcap-default-qpl-dev libsgx-ra-network libsgx-ra-uefi libtdx-attest libtdx-attest-dev && \
# set ssh configs for TDX
    echo 'port=4050' | tee /etc/tdx-attest.conf && \
    pip install flask && \
    echo "mpiuser ALL = NOPASSWD:SETENV: /opt/intel/oneapi/mpi/2021.9.0/bin/mpirun\nmpiuser ALL = NOPASSWD:SETENV: /usr/bin/python" > /etc/sudoers.d/mpivisudo && \
    chmod 440 /etc/sudoers.d/mpivisudo

ADD ./bigdl_aa.py /ppml/bigdl_aa.py
ADD ./quote_generator.py /ppml/quote_generator.py
ADD ./worker_quote_generate.py /ppml/worker_quote_generate.py
ADD ./get_worker_quote.sh /ppml/get_worker_quote.sh
