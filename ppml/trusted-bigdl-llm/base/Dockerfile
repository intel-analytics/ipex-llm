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

RUN pip3 install --pre --upgrade bigdl-llm[all] && \
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


ADD ./requirements.txt /opt/requirements.txt

# Add bigdl-llm-demo dependencies
RUN pip3 install -r /opt/requirements.txt && \
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
    sed -i '9i\sys.insecure__allow_eventfd = true' /ppml/bash.manifest.template && \
    pip3 install --no-cache markupsafe==2.0.1

WORKDIR /ppml
ENTRYPOINT [ "/bin/bash" ]
