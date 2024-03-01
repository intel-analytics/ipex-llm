FROM intelanalytics/bigdl-llm-xpu:2.5.0-SNAPSHOT

ARG http_proxy
ARG https_proxy

# Disable pip's cache behavior
ARG PIP_NO_CACHE_DIR=false

COPY ./entrypoint.sh /opt/entrypoint.sh

# Install Serving Dependencies
RUN cd /llm && \
    pip install --pre --upgrade bigdl-llm[serving] && \
    pip install transformers==4.36.2 gradio==4.19.2 && \
    chmod +x /opt/entrypoint.sh


WORKDIR /llm/
ENTRYPOINT [ "/opt/entrypoint.sh" ]
