FROM intelanalytics/ipex-llm-cpu:2.1.0-SNAPSHOT

ARG http_proxy
ARG https_proxy
ARG TINI_VERSION=v0.18.0

# Disable pip's cache behavior
ARG PIP_NO_CACHE_DIR=false

COPY ./model_adapter.py.patch /llm/model_adapter.py.patch
ADD  https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /sbin/tini
# Install Serving Dependencies
RUN cd /llm && \
    apt-get update && \
    apt-get install -y wrk && \
    pip install --pre --upgrade ipex-llm[serving] && \
    # Fix Trivy CVE Issues
    pip install Jinja2==3.1.3 transformers==4.36.2 gradio==4.19.2 cryptography==42.0.4 && \
    # Fix Qwen model adpater in fastchat
    patch /usr/local/lib/python3.11/dist-packages/fastchat/model/model_adapter.py < /llm/model_adapter.py.patch && \
    chmod +x /sbin/tini && \
    cp /sbin/tini /usr/bin/tini && \
    # Install vllm
    git clone https://github.com/vllm-project/vllm.git && \
    cd ./vllm && \
    git checkout v0.4.2 && \
    pip install wheel packaging ninja setuptools>=49.4.0 numpy && \
    pip install -v -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu && \
    VLLM_TARGET_DEVICE=cpu python3 setup.py install

ADD ./vllm_offline_inference.py       /llm/
ADD ./payload-1024.lua                /llm/
ADD ./start-vllm-service.sh           /llm/
ADD ./benchmark_vllm_throughput.py    /llm/
ADD ./start-fastchat-service.sh       /llm/

WORKDIR /llm/
