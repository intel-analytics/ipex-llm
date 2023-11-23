ARG BIGDL_VERSION=2.5.0-SNAPSHOT
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
ARG USE_CUDA=0 USE_CUDNN=0 USE_OPENMP=1 USE_MKLDNN=1 USE_DISTRIBUTED=1 USE_GLOO=1 USE_NCCL=0 USE_MPI=0 BUILD_CAFFE2=0 USE_ROCM=0 USE_GLOO_WITH_OPENSSL=1 USE_MKL=1 BUILD_TEST=0 BLAS=MKL
ARG CMAKE_PREFIX_PATH="/usr/local/lib/python3.9/dist-packages/:/usr/local/lib/"

RUN mkdir /ppml/examples

ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /sbin/tini
COPY ./entrypoint.sh /opt/entrypoint.sh

# Small examples for PyTorch
COPY ./mnist.py                   /ppml/examples/mnist.py
COPY ./pert.py                    /ppml/examples/pert.py
COPY ./pert_ipex.py               /ppml/examples/pert_ipex.py
COPY ./pert_nano.py               /ppml/examples/pert_nano.py
COPY ./load_save_encryption_ex.py /ppml/examples/load_save_encryption_ex.py
# Patch for datasets
COPY ./filelock.patch             /filelock.patch


# PyTorch Dependencies
RUN env DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y libssl-dev libpthread-stubs0-dev && \
    pip install --no-cache-dir astunparse numpy ninja pyyaml setuptools cmake cffi typing_extensions future six requests dataclasses mkl mkl-include intel-openmp && \
    cd /usr/local/lib && \
    ln -s libmkl_gnu_thread.so.2 libmkl_gnu_thread.so && \
    ln -s libmkl_intel_lp64.so.2 libmkl_intel_lp64.so && \
    ln -s libmkl_core.so.2 libmkl_core.so && \
    env DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y google-perftools=2.7-1ubuntu2 && \
    # apply tcmalloc in bash.manifest.template in LD_PRELOAD
    if [[ $(awk '/^loader\.env\.LD_PRELOAD/ {print $3}' /ppml/bash.manifest.template) != '""' ]]; then \
        sed -i 's/^\(loader\.env\.LD_PRELOAD.*\)"/\1:"/' /ppml/bash.manifest.template; \
    fi && \
    sed -i 's|^\(loader\.env\.LD_PRELOAD.*\)"|\1/ppml/lib/libgomp.so.1:/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"|' /ppml/bash.manifest.template && \
    sed -i '/^sgx\.trusted_files/ a "file:/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4",' /ppml/bash.manifest.template && \
    sed -i '/^sgx\.trusted_files/ a "file:/usr/lib/x86_64-linux-gnu/libunwind.so.8",' /ppml/bash.manifest.template && \
    sed -i '/^sgx\.trusted_files/ a "file:/ppml/lib/libgomp.so.1",' /ppml/bash.manifest.template && \
    sed -i '/sys\.enable_extra_runtime_domain_names_conf/ a sys.brk.max_size = "4G"' /ppml/bash.manifest.template && \
# kubernetes related
    pip3 install --no-cache kubernetes==23.3.0 && \
# PyTorch
    git clone https://github.com/pytorch/pytorch.git /pytorch && \
    cd /pytorch && git checkout v1.13.1 && \
    git submodule sync && git submodule update --init --recursive --jobs 0 && \
    rm -rf ./third_party/gloo && \
    cd third_party && git clone https://github.com/analytics-zoo/gloo.git && \
    cd gloo && git checkout  devel-pt-v1.13.0-2022-11-16 && \
    cd /pytorch && \
    sed -i 's/^version = get_torch_version()/version = "1.13.1"/' setup.py && \
    python3 setup.py install && \
    cd /ppml/ && \
    rm -rf /pytorch && \
# Install torch related packages
    # huggingface related
    pip3 install --no-cache datasets==2.6.1 transformers && \
    # We need to ensure that the required PyTorch version is 1.13.1
    pip3 install --pre --no-cache --upgrade bigdl-nano[pytorch]==2.3.0b20230331 && \
# generate secured_argvs
    gramine-argv-serializer bash -c 'export TF_MKL_ALLOC_MAX_BYTES=10737418240 && $sgx_command' > /ppml/secured_argvs && \
    chmod +x /sbin/tini && \
    chmod +x /opt/entrypoint.sh && \
    cp /sbin/tini /usr/bin/tini && \
# We need to downgrade markupsafe, the markupsafe required by bigdl-nano removed `soft_unicode`
# which is then required by our third-layer gramine make command
    patch /usr/local/lib/python3.9/dist-packages/datasets/utils/filelock.py /filelock.patch && \
    pip3 install --no-cache markupsafe==2.0.1 pyarrow==6.0.1

ENTRYPOINT [ "/opt/entrypoint.sh" ]
