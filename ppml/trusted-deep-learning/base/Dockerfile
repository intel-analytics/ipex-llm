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
ARG USE_CUDA=0 USE_CUDNN=0 USE_MKLDNN=1 USE_DISTRIBUTED=1 USE_GLOO=1 USE_GLOO_WITH_OPENSSL=1 USE_MKL=1 BUILD_TEST=0 BLAS=MKL
ARG CMAKE_PREFIX_PATH="/usr/local/lib/python3.7/dist-packages/:/usr/local/lib/"

RUN mkdir /ppml/examples

ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /sbin/tini
ADD ./entrypoint.sh /opt/entrypoint.sh

# Small examples for PyTorch
ADD ./mnist.py       /ppml/examples/mnist.py
ADD ./pert.py        /ppml/examples/pert.py
ADD ./pert_ipex.py   /ppml/examples/pert_ipex.py
# Patch for datasets
ADD ./filelock.patch /filelock.patch
ADD ./bash.manifest.template.patch /bash.manifest.template.patch


# PyTorch Dependencies
RUN env DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y libssl-dev && \
    pip install --no-cache-dir astunparse numpy ninja pyyaml setuptools cmake cffi typing_extensions future six requests dataclasses mkl mkl-include intel-openmp && \
    pip install --no-cache-dir torchvision==0.13.1 && \
    cd /usr/local/lib && \
    ln -s libmkl_gnu_thread.so.2 libmkl_gnu_thread.so && \
    ln -s libmkl_intel_lp64.so.2 libmkl_intel_lp64.so && \
    ln -s libmkl_core.so.2 libmkl_core.so && \
# huggingface related
    pip3 install --no-cache datasets==2.6.1 transformers intel_extension_for_pytorch && \
# Optimization related
    pip3 install --pre --no-cache --upgrade bigdl-nano[pytorch] && \
# PyTorch
    rm -rf /usr/local/lib/python3.7/dist-packages/torch && \
    git clone https://github.com/analytics-zoo/pytorch /pytorch && \
    cd /pytorch && git checkout devel-v1.13.0-2022-11-16 && \
    git submodule sync && git submodule update --init --recursive --jobs 0 && \
    rm -rf ./third_party/gloo && \
    cd third_party && git clone https://github.com/analytics-zoo/gloo.git && \
    cd gloo && git checkout  devel-pt-v1.13.0-2022-11-16 && \
    cd /pytorch && \
    python3 setup.py install && \
    cd /ppml/ && \
    rm -rf /pytorch && \
# generate secured_argvs
    gramine-argv-serializer bash -c 'export TF_MKL_ALLOC_MAX_BYTES=10737418240 && $sgx_command' > /ppml/secured_argvs && \
    chmod +x /sbin/tini && \
    chmod +x /opt/entrypoint.sh && \
    cp /sbin/tini /usr/bin/tini && \
# We need to downgrade markupsafe, the markupsafe required by bigdl-nano removed `soft_unicode`
# which is then required by our third-layer gramine make command
    patch /usr/local/lib/python3.7/dist-packages/datasets/utils/filelock.py /filelock.patch && \
    pip3 install --no-cache markupsafe==2.0.1 pyarrow==6.0.1 && \
    patch /ppml/bash.manifest.template /bash.manifest.template.patch


ENTRYPOINT [ "/opt/entrypoint.sh" ]
