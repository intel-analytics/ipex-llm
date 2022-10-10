ARG BIGDL_VERSION=2.2.0-SNAPSHOT
ARG TINI_VERSION=v0.18.0
ARG BASE_IMAGE_NAME
ARG BASE_IMAGE_TAG

FROM $BASE_IMAGE_NAME:$BASE_IMAGE_TAG
ARG http_proxy
ARG https_proxy
ARG no_proxy
ARG TINI_VERSION
ENV TINI_VERSION                        $TINI_VERSION
# Environment used for build pytorch
ENV USE_CUDA=0 USE_CUDNN=0 USE_MKLDNN=1
ENV CMAKE_PREFIX_PATH="/usr/local/lib/python3.7/dist-packages/"

ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /sbin/tini
ADD ./entrypoint.sh /opt/entrypoint.sh

# Folders required by bash.menifest.template
RUN mkdir -p /root/.keras/datasets && \
    mkdir -p /root/.zinc && \
    mkdir -p /root/.m2 && \
    mkdir -p /root/.kube/ && \
    # PyTorch dependencies
    pip install --no-cache-dir astunparse numpy ninja pyyaml setuptools cmake cffi typing_extensions future six requests dataclasses mkl mkl-include && \
    pip install --no-cache-dir torchvision==0.13.1 && \
    # PyTorch
    rm -rf ${CMAKE_PREFIX_PATH}/torch && \
    git clone https://github.com/analytics-zoo/pytorch /pytorch && \
    cd /pytorch && git checkout devel-v1.12.1-2022-10-08 && \
    git submodule sync && git submodule update --init --recursive --jobs 0 && \
    rm -rf ./third_party/gloo && \
    cd third_party && git clone https://github.com/analytics-zoo/gloo.git && \
    cd gloo && git checkout devel-pt-v1.12.1-2022-10-08 && \
    cd /pytorch && \
    python3 setup.py install && \
    cd /ppml/ && \
    rm -rf /pytorch && \
    # generate secured_argvs
    gramine-argv-serializer bash -c 'export TF_MKL_ALLOC_MAX_BYTES=10737418240 && $sgx_command' > /ppml/secured_argvs && \
    chmod +x /sbin/tini && \
    chmod +x /opt/entrypoint.sh && \
    cp /sbin/tini /usr/bin/tini

ENTRYPOINT [ "/opt/entrypoint.sh" ]
