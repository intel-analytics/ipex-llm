FROM ubuntu:20.04 as key-getter
ARG http_proxy
ARG https_proxy

RUN apt-get update && \
    apt-get install -y curl gpg && \
    curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB | gpg --dearmor | tee /root/intel-oneapi-archive-keyring.gpg


FROM mpioperator/intel as builder

ARG http_proxy
ARG https_proxy
ENV PIP_NO_CACHE_DIR=false
COPY ./requirements.txt /ipex_llm/requirements.txt

# add public key
COPY --from=key-getter /root/intel-oneapi-archive-keyring.gpg /usr/share/keyrings/intel-oneapi-archive-keyring.gpg
RUN echo "deb [signed-by=/usr/share/keyrings/intel-oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main " > /etc/apt/sources.list.d/oneAPI.list

RUN mkdir /ipex_llm/data && mkdir /ipex_llm/model && \
# install pytorch 2.0.1
    apt-get update && \
    apt-get install -y python3-pip python3.11-dev python3-wheel git software-properties-common && \
    pip3 install --upgrade pip && \
    export PIP_DEFAULT_TIMEOUT=100 && \
    pip install --upgrade torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu && \
# install ipex and oneccl
    pip install intel_extension_for_pytorch==2.0.100 && \
    pip install oneccl_bind_pt -f https://developer.intel.com/ipex-whl-stable && \
# install transformers etc.
    cd /ipex_llm && \
    git clone https://github.com/huggingface/transformers.git && \
    cd transformers && \
    git reset --hard 057e1d74733f52817dc05b673a340b4e3ebea08c && \
    pip install . && \
    pip install -r /ipex_llm/requirements.txt && \
# install python
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install -y python3.11 && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.11 /usr/bin/python3 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
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
    sed -i 's/#\(StrictModes \).*/\1no/g' /etc/ssh/sshd_config

COPY ./ipex-llm-lora-finetuing-entrypoint.sh /ipex_llm/ipex-llm-lora-finetuing-entrypoint.sh
COPY ./lora_finetune.py /ipex_llm/lora_finetune.py

RUN chown -R mpiuser /ipex_llm
USER mpiuser
ENTRYPOINT ["/bin/bash"]
