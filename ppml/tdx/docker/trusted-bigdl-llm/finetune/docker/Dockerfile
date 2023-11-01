ARG BIGDL_LLM_IMAGE_NAME
ARG BIGDL_LLM_IMAGE_TAG
ARG http_proxy
ARG https_proxy

FROM $BIGDL_LLM_IMAGE_NAME:$BIGDL_LLM_IMAGE_TAG as native

FROM mpioperator/intel as builder

ARG http_proxy
ARG https_proxy
ENV PIP_NO_CACHE_DIR=false
COPY --from=native /ppml/requirements.txt /ppml/requirements.txt

RUN mkdir /ppml/data && mkdir /ppml/model && \
# Basic dependencies and DCAP
    apt-get update && \
    apt install -y build-essential apt-utils wget git sudo vim git software-properties-common python3.9 python3-pip python3.9-dev python3-wheel && \
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
    env DEBIAN_FRONTEND=noninteractive apt-get update && apt install -y libsgx-enclave-common-dev libsgx-qe3-logic libsgx-pce-logic libsgx-ae-qe3 libsgx-ae-qve libsgx-urts libsgx-dcap-ql libsgx-dcap-default-qpl libsgx-dcap-quote-verify-dev libsgx-dcap-ql-dev libsgx-dcap-default-qpl-dev libsgx-ra-network libsgx-ra-uefi libtdx-attest libtdx-attest-dev && \
    apt-get install -y python3-pip python3.9-dev python3-wheel && \
    pip3 install --upgrade pip && \
    pip install torch==2.0.1 && \
# install ipex and oneccl
    pip install intel_extension_for_pytorch==2.0.100 && \
    pip install oneccl_bind_pt -f https://developer.intel.com/ipex-whl-stable && \
# install transformers etc.
    cd /ppml && \
    git clone https://github.com/huggingface/transformers.git && \
    cd transformers && \
    git reset --hard 057e1d74733f52817dc05b673a340b4e3ebea08c && \
    pip install . && \
    pip install -r /ppml/requirements.txt && \
# install python
    add-apt-repository ppa:deadsnakes/ppa -y && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.9 /usr/bin/python3 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip install --upgrade pip && \
    pip install --no-cache requests argparse cryptography==3.3.2 urllib3 && \
    pip install --upgrade requests && \
    pip install setuptools==58.4.0 && \
# Install OpenSSH for MPI to communicate between containers
    apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd && \
# Allow OpenSSH to talk to containers without asking for confirmation
# by disabling StrictHostKeyChecking.
# mpi-operator mounts the .ssh folder from a Secret. For that to work, we need
# to disable UserKnownHostsFile to avoid write permissions.
# Disabling StrictModes avoids directory and files read permission checks.
    sed -i 's/[ #]\(.*StrictHostKeyChecking \).*/ \1no/g' /etc/ssh/ssh_config && \
    echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config && \
    sed -i 's/#\(StrictModes \).*/\1no/g' /etc/ssh/sshd_config && \
    echo 'port=4050' | tee /etc/tdx-attest.conf && \
    pip install flask && \
    echo "mpiuser ALL = NOPASSWD:SETENV: /opt/intel/oneapi/mpi/2021.9.0/bin/mpirun\nmpiuser ALL = NOPASSWD:SETENV: /usr/bin/python" > /etc/sudoers.d/mpivisudo && \
    chmod 440 /etc/sudoers.d/mpivisudo

COPY ./bigdl_aa.py /ppml/bigdl_aa.py
COPY ./quote_generator.py /ppml/quote_generator.py
COPY ./worker_quote_generate.py /ppml/worker_quote_generate.py
COPY ./get_worker_quote.sh /ppml/get_worker_quote.sh
COPY --from=native /ppml/bigdl-lora-finetuing-entrypoint.sh /ppml/bigdl-lora-finetuing-entrypoint.sh
COPY --from=native /ppml/lora_finetune.py /ppml/lora_finetune.py

RUN chown -R mpiuser /ppml
USER mpiuser
