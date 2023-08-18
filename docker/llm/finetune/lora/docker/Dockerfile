FROM mpioperator/intel as builder

RUN mkdir /ppml && mkdir /ppml/data && mkdir /ppml/model
ADD ./requirements.txt /ppml/requirements.txt

# install pytorch 2.0.1
RUN export http_proxy=http://child-prc.intel.com:913 && \
    export https_proxy=http://child-prc.intel.com:913 && \
    apt-get update && \
    apt-get install -y python3-pip python3.9-dev python3-wheel && \
    pip3 install --upgrade pip && \
    pip install torch==2.0.1

# install ipex and oneccl
RUN export http_proxy=http://child-prc.intel.com:913 && \
    export https_proxy=http://child-prc.intel.com:913 && \
    pip install intel_extension_for_pytorch==2.0.100 && \
    pip install oneccl_bind_pt -f https://developer.intel.com/ipex-whl-stable

# install transformers etc.
RUN export http_proxy=http://child-prc.intel.com:913 && \
    export https_proxy=http://child-prc.intel.com:913 && \
    cd /ppml && \
    apt-get update && \
    apt-get install -y git && \
    git clone https://github.com/huggingface/transformers.git && \
    cd transformers && \
    git reset --hard 057e1d74733f52817dc05b673a340b4e3ebea08c && \
    pip install . && \
    pip install -r /ppml/requirements.txt

# install python
RUN export http_proxy=http://child-prc.intel.com:913 && \
    export https_proxy=http://child-prc.intel.com:913 && \
    env DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install -y python3.9 && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.9 /usr/bin/python3 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get install -y python3-pip python3.9-dev python3-wheel && \
    pip3 install --upgrade pip && \
    pip3 install --no-cache requests argparse cryptography==3.3.2 urllib3 && \
    pip3 install --upgrade requests && \
    pip3 install setuptools==58.4.0

# Install OpenSSH for MPI to communicate between containers
RUN export http_proxy=http://child-prc.intel.com:913 && \
    export https_proxy=http://child-prc.intel.com:913 && \
    apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd

# Allow OpenSSH to talk to containers without asking for confirmation
# by disabling StrictHostKeyChecking.
# mpi-operator mounts the .ssh folder from a Secret. For that to work, we need
# to disable UserKnownHostsFile to avoid write permissions.
# Disabling StrictModes avoids directory and files read permission checks.
RUN sed -i 's/[ #]\(.*StrictHostKeyChecking \).*/ \1no/g' /etc/ssh/ssh_config && \
    echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config && \
    sed -i 's/#\(StrictModes \).*/\1no/g' /etc/ssh/sshd_config

ADD ./bigdl-ppml-finetuing-entrypoint.sh /ppml/bigdl-ppml-finetuing-entrypoint.sh
ADD ./lora_finetune.py /ppml/lora_finetune.py
RUN chmod -R a+x /ppml
