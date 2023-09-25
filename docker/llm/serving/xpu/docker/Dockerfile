FROM intelanalytics/bigdl-llm-xpu:2.4.0-SNAPSHOT

ARG http_proxy
ARG https_proxy

# Disable pip's cache behavior
ARG PIP_NO_CACHE_DIR=false

# Install Serving Dependencies
RUN mkdir /llm && \
    cd /llm && \
    # Install FastChat will require to update pip
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    # Install FastChat from source requires PEP 660 support
    python3 get-pip.py && \
    rm get-pip.py && \
    git clone https://github.com/analytics-zoo/FastChat.git && \
    cd FastChat && \
    git checkout dev-2023-09-22 && \
    pip3 install -e ".[model_worker,webui]" && \
    cd /llm


WORKDIR /llm/
