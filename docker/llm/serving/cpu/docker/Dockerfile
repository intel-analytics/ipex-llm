FROM intelanalytics/bigdl-llm-cpu:2.4.0-SNAPSHOT

ARG http_proxy
ARG https_proxy
ARG TINI_VERSION=v0.18.0

# Disable pip's cache behavior
ARG PIP_NO_CACHE_DIR=false

COPY ./entrypoint.sh /opt/entrypoint.sh
ADD  https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /sbin/tini
# Install Serving Dependencies
RUN cd /llm && \
    git clone https://github.com/analytics-zoo/FastChat.git && \
    cd FastChat && \
    git checkout dev-2023-09-22 && \
    pip3 install -e ".[model_worker,webui]" && \
    cd /llm && \
    chmod +x /opt/entrypoint.sh && \
    chmod +x /sbin/tini && \
    cp /sbin/tini /usr/bin/tini


WORKDIR /llm/
ENTRYPOINT [ "/opt/entrypoint.sh" ]