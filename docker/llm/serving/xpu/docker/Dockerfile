FROM intelanalytics/bigdl-llm-xpu:2.5.0-SNAPSHOT

ARG http_proxy
ARG https_proxy

# Disable pip's cache behavior
ARG PIP_NO_CACHE_DIR=false

COPY ./entrypoint.sh /opt/entrypoint.sh

# Install Serving Dependencies
RUN cd /llm && \
    git clone https://github.com/analytics-zoo/FastChat.git && \
    cd FastChat && \
    git checkout dev-2023-09-22 && \
    pip3 install -e ".[model_worker,webui]" && \
    cd /llm && \
    chmod +x /opt/entrypoint.sh


WORKDIR /llm/
ENTRYPOINT [ "/opt/entrypoint.sh" ]
