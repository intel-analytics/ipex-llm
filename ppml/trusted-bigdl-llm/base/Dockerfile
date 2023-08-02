ARG BIGDL_VERSION=2.4.0-SNAPSHOT
ARG BASE_IMAGE_NAME
ARG BASE_IMAGE_TAG

FROM $BASE_IMAGE_NAME:$BASE_IMAGE_TAG 
ARG http_proxy
ARG https_proxy
ARG no_proxy

ARG http_proxy
ARG https_proxy

ADD utils.diff /opt/utils.diff
ADD llm_cli.diff /opt/llm_cli.diff

RUN pip3 install --pre --upgrade bigdl-llm[all]==2.4.0b20230710 && \
    # Remove all dependencies from nvidia as we are supposed to run in SGX
    pip3 list | grep nvidia | awk '{print $1}' | xargs pip3 uninstall -y && \
    # Replace bigdl-llm repo's with our our dependencies
    cd /usr/local/lib/python3.9/dist-packages/bigdl/llm/libs && \
    rm *avx512* && \
    rm quantize* && \
    curl "https://sourceforge.net/projects/analytics-zoo/rss?path=/ppml-llm" | grep "<link>.*</link>" | sed 's|<link>||;s|</link>||' | while read url; do url=`echo $url | sed 's|/download$||'`; wget $url ; done && \
    rm index.html && \
    chmod +x * && \
    cd /ppml && \
    # Patch subprocess call in bigdl-llm
    patch -R /usr/local/lib/python3.9/dist-packages/bigdl/llm/utils/utils.py /opt/utils.diff && \
    patch -R /usr/local/bin/llm-cli /opt/llm_cli.diff && \
    # Gramine commands
    gramine-argv-serializer bash -c 'export TF_MKL_ALLOC_MAX_BYTES=10737418240 && $sgx_command' > /ppml/secured_argvs

WORKDIR /ppml
