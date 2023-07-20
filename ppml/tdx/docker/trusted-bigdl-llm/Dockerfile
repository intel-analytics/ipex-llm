FROM ubuntu:20.04

ARG http_proxy
ARG https_proxy

RUN mkdir -p /ppml/


ADD ./fschat.diff /opt/fschat.diff
ADD ./requirements.txt /opt/requirements.txt


# Install PYTHON 3.9
RUN env DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt install software-properties-common libunwind8-dev vim less -y && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install -y python3.9 git curl wget && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.9 /usr/bin/python3 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get install -y python3-pip python3.9-dev python3-wheel && \
    pip3 install --no-cache requests argparse cryptography==3.3.2 urllib3 && \
    # Install bigdl-llm
    pip3 install --pre --upgrade bigdl-llm[all] && \
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
    rm -rf /root/BigDL


# To boost fschat performance, do the following:
# source bigdl-nano-init -t
# export KMP_AFFINITY=granularity=fine,compact
# export OMP_NUM_THREADS=#phy cores


ENV PYTHONPATH /usr/lib/python3.9:/usr/lib/python3.9/lib-dynload:/usr/local/lib/python3.9/dist-packages:/usr/lib/python3/dist-packages

WORKDIR /ppml
ENTRYPOINT [ "/bin/bash" ]
