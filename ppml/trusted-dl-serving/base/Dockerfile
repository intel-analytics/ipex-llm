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
    cd pytorch-serve/frontend && \
    ./gradlew clean assemble && \
    mkdir -p /ppml/torchserve && \
    mv server/build/libs/server-1.0.jar /ppml/torchserve/frontend.jar

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
    ./build.py -v --no-container-build --build-dir=`pwd`/build --backend=python --repoagent=checksum --filesystem=gcs --filesystem=s3 --filesystem=azure_storage --endpoint=http --endpoint=grpc --endpoint=sagemaker --endpoint=vertex-ai --enable-logging --enable-stats --enable-tracing

FROM nvcr.io/nvidia/tritonserver:23.01-py3-min AS min_container

# Stage.3 TF-Serving
FROM tensorflow/serving:latest-devel as tf-serving
ARG http_proxy
ARG https_proxy
ARG no_proxy
RUN bazel build -c opt tensorflow_serving/... || true

# DL-Serving
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
ARG TRITON_VERSION=2.31.0dev
ARG TRITON_CONTAINER_VERSION=23.02dev
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

RUN mkdir /ppml/examples && \
    mkdir /ppml/torchserve && \
    mkdir /ppml/tritonserver && \
    mkdir /ppml/tf-serving && \
    mkdir -p /usr/local/cuda/lib64/stubs && \
    mkdir -p /usr/local/cuda/targets/x86_64-linux/lib && \
    mkdir /opt/tritonserver && \
    mkdir /opt/nvidia

ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /sbin/tini
ADD ./entrypoint.sh /opt/entrypoint.sh

# Small examples for PyTorch
ADD ./mnist.py                   /ppml/examples/mnist.py
ADD ./pert.py                    /ppml/examples/pert.py
ADD ./pert_ipex.py               /ppml/examples/pert_ipex.py
ADD ./load_save_encryption_ex.py /ppml/examples/load_save_encryption_ex.py
# Patch for datasets
ADD ./filelock.patch /filelock.patch

# COPY frontend.jar
COPY --from=temp /ppml/torchserve/frontend.jar /ppml/torchserve/frontend.jar
# Start Script for Torchserve, Tritonserver and TF-Serving
ADD ./start-backend-sgx.sh      /ppml/torchserve/start-backend-sgx.sh
ADD ./start-frontend-sgx.sh     /ppml/torchserve/start-frontend-sgx.sh
ADD ./start-torchserve-sgx.sh   /ppml/torchserve/start-torchserve-sgx.sh
ADD ./start-tritonserver-sgx.sh /ppml/tritonserver/start-tritonserver-sgx.sh
ADD ./start-tf-serving-sgx.sh   /ppml/tf-serving/start-tf-serving-sgx.sh


#For Tritonserver python backend
COPY --from=min_container /usr/local/cuda/lib64/stubs/libcusparse.so /usr/local/cuda/lib64/stubs/libcusparse.so.12
COPY --from=min_container /usr/local/cuda/lib64/stubs/libcusolver.so /usr/local/cuda/lib64/stubs/libcusolver.so.11
COPY --from=min_container /usr/local/cuda/lib64/stubs/libcurand.so /usr/local/cuda/lib64/stubs/libcurand.so.10
COPY --from=min_container /usr/local/cuda/lib64/stubs/libcufft.so /usr/local/cuda/lib64/stubs/libcufft.so.11
COPY --from=min_container /usr/local/cuda/lib64/stubs/libcublas.so /usr/local/cuda/lib64/stubs/libcublas.so.12
COPY --from=min_container /usr/local/cuda/lib64/stubs/libcublasLt.so /usr/local/cuda/lib64/stubs/libcublasLt.so.12
COPY --from=min_container /usr/local/cuda/lib64/stubs/libcublasLt.so /usr/local/cuda/lib64/stubs/libcublasLt.so.11
COPY --from=min_container /usr/local/cuda-12.0/targets/x86_64-linux/lib/libcudart.so.12 /usr/local/cuda/targets/x86_64-linux/lib/.
COPY --from=min_container /usr/local/cuda-12.0/targets/x86_64-linux/lib/libcupti.so.12 /usr/local/cuda/targets/x86_64-linux/lib/.
COPY --from=min_container /usr/local/cuda-12.0/targets/x86_64-linux/lib/libnvToolsExt.so.1 /usr/local/cuda/targets/x86_64-linux/lib/.
COPY --from=min_container /usr/local/cuda-12.0/targets/x86_64-linux/lib/libnvJitLink.so.12 /usr/local/cuda/targets/x86_64-linux/lib/.
COPY --from=min_container /usr/lib/x86_64-linux-gnu/libcudnn.so.8 /usr/lib/x86_64-linux-gnu/libcudnn.so.8
COPY --from=min_container /usr/lib/x86_64-linux-gnu/libnccl.so.2 /usr/lib/x86_64-linux-gnu/libnccl.so.2

#Tritonserver
COPY --from=tritonserver /server/build/opt/tritonserver /opt/tritonserver
COPY --from=tritonserver /server/docker/entrypoint.d/ /opt/nvidia/entrypoint.d/
COPY --from=tritonserver /server/docker/cpu_only/ /opt/nvidia/
COPY --from=min_container /opt/tritonserver/backends/pytorch /opt/tritonserver/backends
COPY --from=min_container /opt/tritonserver/backends/tensorflow2 /opt/tritonserver/backends

#TF-Serving
COPY --from=tf-serving /tensorflow-serving/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server /usr/bin/tensorflow_model_server

# PyTorch Dependencies
RUN env DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common libb64-0d             libcurl4-openssl-dev libre2-5 git gperf dirmngr libgoogle-perftools-dev libnuma-dev curl libgomp1 openmpi-bin patchelf python3 libarchive-dev python3-pip libpython3-dev && \
    rm -rf /var/lib/apt/lists/* &&\
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
    pip3 install --no-cache markupsafe==2.0.1 pyarrow==6.0.1 && \
    patchelf --add-needed /usr/local/cuda/lib64/stubs/libcublasLt.so.12 backends/pytorch/libtorch_cuda.so

ENTRYPOINT [ "/opt/entrypoint.sh" ]
