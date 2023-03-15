ARG BIGDL_VERSION=2.3.0-SNAPSHOT
ARG TINI_VERSION=v0.18.0
ARG BASE_IMAGE_NAME
ARG BASE_IMAGE_TAG
ARG JDK_VERSION=11

#Stage.1 Torchserve Frontend
FROM $BASE_IMAGE_NAME:$BASE_IMAGE_TAG as temp
ARG http_proxy
ARG https_proxy
ARG JDK_VERSION
ENV JDK_HOME                /opt/jdk${JDK_VERSION}
ENV JAVA_HOME                           /opt/jdk${JDK_VERSION}

RUN env DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y openjdk-${JDK_VERSION}-jdk && \
    mkdir -p ${JAVA_HOME} && \
    cp -r /usr/lib/jvm/java-${JDK_VERSION}-openjdk-amd64/* ${JAVA_HOME} && \
    git clone https://github.com/analytics-zoo/pytorch-serve.git && \
    cd pytorch-serve/frontend && \
    ./gradlew clean assemble && \
    mkdir -p /ppml/torchserve && \
    mv server/build/libs/server-1.0.jar /ppml/torchserve/frontend.jar

FROM $BASE_IMAGE_NAME:$BASE_IMAGE_TAG
ARG http_proxy
ARG https_proxy
ARG no_proxy
ARG TINI_VERSION
ENV TINI_VERSION                        $TINI_VERSION
ARG JDK_VERSION
ENV JDK_HOME                            /opt/jdk${JDK_VERSION}
ENV JAVA_HOME                           /opt/jdk${JDK_VERSION}
# Environment used for build pytorch
ARG USE_CUDA=0 USE_CUDNN=0 USE_MKLDNN=1 USE_DISTRIBUTED=1 USE_GLOO=1 USE_GLOO_WITH_OPENSSL=1 USE_MKL=1 BUILD_TEST=0 BLAS=MKL
ARG CMAKE_PREFIX_PATH="/usr/local/lib/python3.8/dist-packages/:/usr/local/lib/"

RUN mkdir /ppml/examples && \
    mkdir /ppml/torchserve

ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /sbin/tini
ADD ./entrypoint.sh /opt/entrypoint.sh

# Small examples for PyTorch
ADD ./mnist.py                   /ppml/examples/mnist.py
ADD ./pert.py                    /ppml/examples/pert.py
ADD ./pert_ipex.py               /ppml/examples/pert_ipex.py
ADD ./pert_nano.py               /ppml/examples/pert_nano.py
ADD ./load_save_encryption_ex.py /ppml/examples/load_save_encryption_ex.py
# Patch for datasets
ADD ./filelock.patch /filelock.patch

# COPY frontend.jar
COPY --from=temp /ppml/torchserve/frontend.jar /ppml/torchserve/frontend.jar
# Start Script for Torchserve
ADD ./start-backend-sgx.sh      /ppml/torchserve/start-backend-sgx.sh
ADD ./start-frontend-sgx.sh     /ppml/torchserve/start-frontend-sgx.sh
ADD ./start-torchserve-sgx.sh   /ppml/torchserve/start-torchserve-sgx.sh


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
# kubernetes related
    pip3 install --no-cache kubernetes==23.3.0 && \
# Torchserve
    pip install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir cython pillow==9.0.1 captum packaging nvgpu && \
    pip install --no-cache-dir torchserve==0.6.1 torch-model-archiver==0.6.1 torch-workflow-archiver==0.2.5 && \
    apt-get install -y openjdk-${JDK_VERSION}-jdk && \
    mkdir -p ${JAVA_HOME} && \
    cp -r /usr/lib/jvm/java-${JDK_VERSION}-openjdk-amd64/* ${JAVA_HOME} && \
    sed -i '/MAX_FAILURE_THRESHOLD = 5/ios.environ\[\"MPLCONFIGDIR\"\]=\"\/tmp\/matplotlib\"' /usr/local/lib/python3.8/dist-packages/ts/model_service_worker.py && \
    sed -i '/import abc/iimport sys' /usr/local/lib/python3.8/dist-packages/ts/torch_handler/base_handler.py && \
    sed -i '/module = importlib.import_module/i\ \ \ \ \ \ \ \ sys.path.append(model_dir)' /usr/local/lib/python3.8/dist-packages/ts/torch_handler/base_handler.py && \
    sed -i 's/SOCKET_ACCEPT_TIMEOUT = 30.0/SOCKET_ACCEPT_TIMEOUT = 3000.0/' /usr/local/lib/python3.8/dist-packages/ts/model_service_worker.py && \
    sed -i '/os.path.join/i\ \ \ \ \ \ \ \ sys.path.append(model_dir)' /usr/local/lib/python3.8/dist-packages/ts/model_loader.py && \
    sed -i '/import json/iimport sys' /usr/local/lib/python3.8/dist-packages/ts/model_loader.py && \
    cp /usr/local/lib/python3.8/dist-packages/ts/configs/metrics.yaml /ppml && \
    chmod +x /ppml/torchserve/start-backend-sgx.sh && \
    chmod +x /ppml/torchserve/start-frontend-sgx.sh && \
    chmod +x /ppml/torchserve/start-torchserve-sgx.sh && \
# PyTorch
    rm -rf /usr/local/lib/python3.8/dist-packages/torch && \
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
    patch /usr/local/lib/python3.8/dist-packages/datasets/utils/filelock.py /filelock.patch && \
    pip3 install --no-cache markupsafe==2.0.1 pyarrow==6.0.1

ENTRYPOINT [ "/opt/entrypoint.sh" ]
