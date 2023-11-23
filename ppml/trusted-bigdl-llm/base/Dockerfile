ARG BIGDL_VERSION=2.5.0-SNAPSHOT
ARG BASE_IMAGE_NAME
ARG BASE_IMAGE_TAG

FROM $BASE_IMAGE_NAME:$BASE_IMAGE_TAG 
ARG http_proxy
ARG https_proxy
ARG no_proxy
ARG TINI_VERSION=v0.18.0

ARG http_proxy
ARG https_proxy

COPY utils.diff /opt/utils.diff
COPY llm_cli.diff /opt/llm_cli.diff
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /sbin/tini


RUN pip3 install --pre --upgrade bigdl-llm[all]==2.4.0b20230830 && \
    # Remove all dependencies from nvidia as we are supposed to run in SGX
    pip3 list | grep nvidia | awk '{print $1}' | xargs pip3 uninstall -y && \
#    # Replace bigdl-llm repo's with our our dependencies
#    cd /usr/local/lib/python3.9/dist-packages/bigdl/llm/libs && \
#    rm *avx512* && \
#    rm quantize* && \
#    curl "https://sourceforge.net/projects/analytics-zoo/rss?path=/ppml-llm" | grep "<link>.*</link>" | sed 's|<link>||;s|</link>||' | while read url; do url=`echo $url | sed 's|/download$||'`; wget $url ; done && \
#    rm index.html && \
#    chmod +x * && \
    cd /ppml && \
    # Patch subprocess call in bigdl-llm
    patch /usr/local/lib/python3.9/dist-packages/bigdl/llm/utils/utils.py /opt/utils.diff && \
    patch -R /usr/local/bin/llm-cli /opt/llm_cli.diff && \
    # Gramine commands
    gramine-argv-serializer bash -c 'export TF_MKL_ALLOC_MAX_BYTES=10737418240 && $sgx_command' > /ppml/secured_argvs && \
    cd /ppml/ && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    # Install FastChat from source requires PEP 660 support
    python3 get-pip.py && \
    rm get-pip.py && \
    # Install FastChat
    git clone https://github.com/analytics-zoo/FastChat.git && \
    cd /ppml/FastChat && \
    git checkout dev-2023-08-01 && \
    pip install -e . && \
    # Pin gradio version because this error:https://github.com/lm-sys/FastChat/issues/1925
    pip install --pre --upgrade gradio==3.36.1 && \
    pip install --pre --upgrade bigdl-nano && \
    apt-get install -y libunwind8-dev && \
    mkdir /ppml/data

COPY ./entrypoint.sh /opt/entrypoint.sh
RUN  chmod +x /opt/entrypoint.sh && \
     chmod +x /sbin/tini
WORKDIR /ppml
ENTRYPOINT [ "/opt/entrypoint.sh" ]
