FROM ubuntu:20.04

ARG http_proxy
ARG https_proxy

RUN mkdir -p /ppml/


ADD ./fschat.diff /opt/fschat.diff


# Install PYTHON 3.9
RUN env DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt install software-properties-common libunwind8-dev vim -y && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install -y python3.9 && \
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
    pip3 install --pre --upgrade bigdl-nano

ENV PYTHONPATH /usr/lib/python3.9:/usr/lib/python3.9/lib-dynload:/usr/local/lib/python3.9/dist-packages:/usr/lib/python3/dist-packages

WORKDIR /ppml
ENTRYPOINT [ "/bin/bash" ]
