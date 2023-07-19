ARG BIGDL_VERSION=2.3.0-SNAPSHOT
ARG TINI_VERSION=v0.18.0
ARG BASE_IMAGE_NAME
ARG BASE_IMAGE_TAG
ARG JDK_VERSION=11

# Stage.1 Torchserve Frontend
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
    cd pytorch-serve && \
    python setup.py install && \
    mkdir -p /ppml/torchserve && \
    mv ts/frontend/model-server.jar /ppml/torchserve/frontend.jar && \
    cd model-archiver && \
    pip install .
    

# Stage.2 Tritonserver
From ubuntu:20.04 as tritonserver
ARG http_proxy
ARG https_proxy
ARG TRITON_VERSION=2.31.0dev
ARG TRITON_CONTAINER_VERSION=23.02dev
ENV DEBIAN_FRONTEND=noninteractive
ENV TRITON_SERVER_VERSION ${TRITON_VERSION}
ENV NVIDIA_TRITON_SERVER_VERSION ${TRITON_CONTAINER_VERSION}

RUN cd / &&\
    apt-get update &&\
    apt-get install -y --no-install-recommends ca-certificates autoconf automake build-essential docker.io git gperf libre2-dev libssl-dev libtool libboost-dev libcurl4-openssl-dev libb64-dev libgoogle-perftools-dev patchelf python3-dev python3-pip python3-setuptools rapidjson-dev scons software-properties-common unzip wget zlib1g-dev libarchive-dev pkg-config uuid-dev libnuma-dev &&\
    rm -rf /var/lib/apt/lists/* &&\
    pip3 install --upgrade pip &&\
    pip3 install --upgrade wheel setuptools docker &&\
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null &&\
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main' &&\
    apt-get update &&\
    apt-get install -y --no-install-recommends cmake-data=3.21.1-0kitware1ubuntu20.04.1 cmake=3.21.1-0kitware1ubuntu20.04.1 &&\
    wget https://boostorg.jfrog.io/artifactory/main/release/1.78.0/source/boost_1_78_0.tar.gz &&\
    tar -zxvf boost_1_78_0.tar.gz &&\
    rm -rf /usr/include/boost &&\
    cp -r boost_1_78_0/boost /usr/include &&\
    git clone --branch v2.31.0 https://github.com/triton-inference-server/server.git &&\
    cd server &&\
    ./build.py -v --no-container-build --build-dir=`pwd`/build --backend=python --repoagent=checksum --filesystem=gcs --filesystem=s3 --filesystem=azure_storage --endpoint=http --endpoint=grpc --endpoint=sagemaker --endpoint=vertex-ai --enable-logging --enable-stats --enable-tracing && \
    mkdir -p /dl-serving/opt && \
    cd /dl-serving/opt && \
    cp -r /server/build/opt/tritonserver ./tritonserver && \
    cp -r /server/docker/entrypoint.d ./entrypoint.d && \
    cp -r /server/docker/cpu_only/ ./nvidia

FROM nvcr.io/nvidia/tritonserver:23.01-py3 AS min_container

RUN mkdir -p /dl-serving/usr/local/cuda/lib64/stubs && \
    cd /dl-serving/usr/local/cuda/lib64/stubs && \
    cp /usr/local/cuda/lib64/stubs/libcusparse.so ./libcusparse.so.12 && \
    cp /usr/local/cuda/lib64/stubs/libcusolver.so ./libcusolver.so.11 && \
    cp /usr/local/cuda/lib64/stubs/libcurand.so ./libcurand.so.10 && \
    cp /usr/local/cuda/lib64/stubs/libcufft.so ./libcufft.so.11 && \
    cp /usr/local/cuda/lib64/stubs/libcublas.so ./libcublas.so.12 && \
    cp /usr/local/cuda/lib64/stubs/libcublasLt.so ./libcublasLt.so.12 && \
    cp /usr/local/cuda/lib64/stubs/libcublasLt.so ./libcublasLt.so.11 && \
    mkdir -p /dl-serving/usr/local/cuda/targets/x86_64-linux/lib && \
    cd /dl-serving/usr/local/cuda/targets/x86_64-linux/lib && \
    cp /usr/local/cuda-12.0/targets/x86_64-linux/lib/libcudart.so.12 . && \
    cp /usr/local/cuda-12.0/targets/x86_64-linux/lib/libcupti.so.12 . && \
    cp /usr/local/cuda-12.0/targets/x86_64-linux/lib/libnvToolsExt.so.1 . && \
    cp /usr/local/cuda-12.0/targets/x86_64-linux/lib/libnvJitLink.so.12 . && \
    mkdir -p /dl-serving/usr/lib/x86_64-linux-gnu && \
    cd /dl-serving/usr/lib/x86_64-linux-gnu && \
    cp /usr/lib/x86_64-linux-gnu/libcudnn.so.8 ./libcudnn.so.8 && \
    cp /usr/lib/x86_64-linux-gnu/libnccl.so.2 ./libnccl.so.2 && \
    mkdir -p /dl-serving/opt/tritonserver/backends && \
    cd /dl-serving/opt/tritonserver/backends && \
    mkdir pytorch && \
    mkdir tensorflow2 && \
    cp -r /opt/tritonserver/backends/pytorch . && \
    cp -r /opt/tritonserver/backends/tensorflow2 .


# Stage.3 TF-Serving
FROM tensorflow/serving:latest-devel as tf-serving

# DL-Serving
FROM $BASE_IMAGE_NAME:$BASE_IMAGE_TAG
ARG http_proxy
ARG https_proxy
ARG no_proxy
ARG JDK_VERSION
ARG TINI_VERSION
ARG TRITON_VERSION=2.31.0dev
ARG TRITON_CONTAINER_VERSION=23.02dev

ENV TINI_VERSION                        $TINI_VERSION
ENV JDK_HOME                            /opt/jdk${JDK_VERSION}
ENV JAVA_HOME                           /opt/jdk${JDK_VERSION}
ENV TRITON_SERVER_VERSION ${TRITON_VERSION}
ENV NVIDIA_TRITON_SERVER_VERSION ${TRITON_CONTAINER_VERSION}

ENV PATH /opt/tritonserver/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}

ENV TF_ADJUST_HUE_FUSED         1
ENV TF_ADJUST_SATURATION_FUSED  1
ENV TF_ENABLE_WINOGRAD_NONFUSED 1
ENV TF_AUTOTUNE_THRESHOLD       2
ENV TRITON_SERVER_GPU_ENABLED    0

ENV TCMALLOC_RELEASE_RATE 200

ENV TS_MODEL_SERVER_HOME /usr/local/lib/python3.9/dist-packages/

RUN mkdir /ppml/examples && \
    mkdir /ppml/torchserve && \
    mkdir /ppml/tritonserver && \
    mkdir /ppml/tf-serving && \
    mkdir -p /usr/local/cuda/lib64/stubs && \
    mkdir -p /usr/local/cuda/targets/x86_64-linux/lib && \
    mkdir /opt/tritonserver && \
    mkdir /opt/nvidia

ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /sbin/tini

# Torchserve
COPY --from=temp /ppml/torchserve/frontend.jar /ppml/torchserve/frontend.jar
COPY --from=temp /usr/local/lib/python3.9/dist-packages/torchserve*/ts /usr/local/lib/python3.9/dist-packages/ts
COPY --from=temp /usr/loacl/lib/python3.9/dist-packages/model-archiver /usr/local/lib/python3.9/dist-packages/model-archiver
COPY --from=temp /usr/local/bin/torch-model-archiver /usr/local/bin/torch-model-archiver

# Start Script for Torchserve, Tritonserver and TF-Serving
ADD ./ppml /ppml

#Tritonserver
COPY --from=tritonserver /dl-serving/opt /opt
COPY --from=min_container /dl-serving /

#TF-Serving
COPY --from=tf-serving /usr/local/bin/tensorflow_model_server /usr/bin/tensorflow_model_server

# Dependencies
RUN env DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common libb64-0d             libcurl4-openssl-dev libre2-5 git gperf dirmngr libgoogle-perftools-dev libnuma-dev curl libgomp1 openmpi-bin patchelf python3 libarchive-dev libpython3-dev && \
    apt-get install -y libssl-dev && \
# Optimization related
    pip3 install --pre --no-cache --upgrade bigdl-nano[pytorch_112,inference] vit_pytorch && \
# Torchserve
    pip3 install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip3 install --no-cache-dir cython pillow==9.0.1 captum packaging nvgpu && \
    pip3 install --no-cache-dir torch-model-archiver==0.6.1 torch-workflow-archiver==0.2.5 && \
    apt-get install -y openjdk-${JDK_VERSION}-jdk && \
    mkdir -p ${JAVA_HOME} && \
    cp -r /usr/lib/jvm/java-${JDK_VERSION}-openjdk-amd64/* ${JAVA_HOME} && \
    cp /usr/local/lib/python3.9/dist-packages/ts/configs/metrics.yaml /ppml && \
# generate secured_argvs
    gramine-argv-serializer bash -c 'export TF_MKL_ALLOC_MAX_BYTES=10737418240 && $sgx_command' > /ppml/secured_argvs && \
    chmod +x /sbin/tini && \
    chmod +x /ppml/torchserve/frontend-entrypoint.sh && \
    chmod +x /ppml/torchserve/backend-entrypoint.sh && \
    cp /sbin/tini /usr/bin/tini && \
# We need to downgrade markupsafe, the markupsafe required by bigdl-nano removed `soft_unicode`
# which is then required by our third-layer gramine make command
    pip3 install --no-cache markupsafe==2.0.1 pyarrow==6.0.1 transformers cryptography && \
    ls /opt/tritonserver/backends/pytorch && \
    patchelf --add-needed /usr/local/cuda/lib64/stubs/libcublasLt.so.12 /opt/tritonserver/backends/pytorch/libtorch_cuda.so

ENTRYPOINT [ "/ppml/torchserve/frontend-entrypoint.sh" ]
