FROM ubuntu:20.04

ARG http_proxy
ARG https_proxy
ARG PIP_NO_CACHE_DIR=false

# Install PYTHON 3.9
RUN env DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt install software-properties-common libunwind8-dev vim less -y && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install -y python3.9 git curl wget && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.9 /usr/bin/python3 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    apt-get install -y python3-pip python3.9-dev python3-wheel python3.9-distutils && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    cd / && \
    git clone https://github.com/analytics-zoo/GPTCache.git && \
    cd GPTCache && \
    pip install -r requirements.txt && \
    python setup.py install

ENTRYPOINT ["python /GPTCache/openai_api_server.py"]