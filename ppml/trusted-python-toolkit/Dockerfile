ARG http_proxy
ARG https_proxy
ARG BASE_IMAGE_NAME
ARG BASE_IMAGE_TAG
ARG BIGDL_VERSION=2.2.0-SNAPSHOT
ARG TINI_VERSION=v0.18.0
ARG JDK_VERSION=11
ARG TRITON_VERSION=2.27.0
ARG TRITON_CONTAINER_VERSION=22.10

#Stage.1 Torchserve Frontend
FROM $BASE_IMAGE_NAME:$BASE_IMAGE_TAG as torchserve
ARG http_proxy
ARG https_proxy
ARG JDK_VERSION
ENV JDK_HOME				/opt/jdk${JDK_VERSION}
ENV JAVA_HOME                           /opt/jdk${JDK_VERSION}

#build frontend.jar
RUN apt-get install -y openjdk-${JDK_VERSION}-jdk && \
    mkdir -p ${JAVA_HOME} && \
    cp -r /usr/lib/jvm/java-${JDK_VERSION}-openjdk-amd64/* ${JAVA_HOME} && \
    git clone https://github.com/analytics-zoo/pytorch-serve.git && \
    cd pytorch-serve/frontend && \
    ./gradlew clean assemble && \
    mkdir -p /ppml/torchserve && \
    mv server/build/libs/server-1.0.jar /ppml/torchserve/frontend.jar

#Stage.2 Triton Server
FROM nvcr.io/nvidia/tritonserver:22.10-py3-min as min_container


From ubuntu:20.04 as tritonserver
ARG http_proxy
ARG https_proxy
ARG TRITON_VERSION
ARG TRITON_CONTAINER_VERSION
ENV PATH /opt/conda/bin:${PATH}

ENV TRITON_SERVER_VERSION ${TRITON_VERSION}
ENV NVIDIA_TRITON_SERVER_VERSION ${TRITON_CONTAINER_VERSION}

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates autoconf automake build-essential docker.io git libre2-dev libssl-dev libtool libboost-dev libcurl4-openssl-dev libb64-dev patchelf python3-dev python3-pip python3-setuptools rapidjson-dev scons software-properties-common unzip wget zlib1g-dev libarchive-dev pkg-config uuid-dev libnuma-dev && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install --upgrade pip && \
    pip3 install --upgrade wheel setuptools docker && \

# Server build requires recent version of CMake (FetchContent required)
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main' && \
    apt-get update && \
    apt-get install -y --no-install-recommends cmake-data=3.21.1-0kitware1ubuntu20.04.1 cmake=3.21.1-0kitware1ubuntu20.04.1 && \

    mkdir -p /workspace/ && \
    wget "https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh" -O miniconda.sh -q && \
    echo "3190da6626f86eee8abf1b2fd7a5af492994eb2667357ee4243975cdbb175d7a" "miniconda.sh" > shasum && \
    sha256sum -c ./shasum && \
    sh miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh shasum && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy && \
    git clone https://github.com/triton-inference-server/server.git && \
    cd server && \
    cp -r * /workspace && \
    ./build.py -v --backend=pytorch --repoagent=checksum --filesystem=gcs --filesystem=s3 --filesystem=azure_storage --endpoint=http --endpoint=grpc --endpoint=sagemaker --endpoint=vertex-ai --enable-logging --enable-stats --enable-metrics --enable-cpu-metrics --enable-tracing --dryrun && \
    cd build && \
    ./cmake_build

#Stage.3 Flask & Numpy & Pandas & Torchserve & Triton Server
FROM $BASE_IMAGE_NAME:$BASE_IMAGE_TAG
ARG http_proxy
ARG https_proxy
ARG JDK_VERSION
ARG TINI_VERSION
ENV JDK_HOME				/opt/jdk${JDK_VERSION}
ENV JAVA_HOME				/opt/jdk${JDK_VERSION}
ENV TINI_VERSION                        $TINI_VERSION
ARG TRITON_VERSION
ARG TRITON_CONTAINER_VERSION
ENV TRITON_SERVER_VERSION ${TRITON_VERSION}
ENV NVIDIA_TRITON_SERVER_VERSION ${TRITON_CONTAINER_VERSION}
ENV PATH /opt/tritonserver/bin:${PATH}
ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH /usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}
ENV PYTHONPATH /usr/lib/python3.8:/usr/lib/python3.8/lib-dynload:/usr/local/lib/python3.8/dist-packages:/usr/lib/python3/dist-packages

ENV TF_ADJUST_HUE_FUSED         1
ENV TF_ADJUST_SATURATION_FUSED  1
ENV TF_ENABLE_WINOGRAD_NONFUSED 1
ENV TF_AUTOTUNE_THRESHOLD       2
ENV TRITON_SERVER_GPU_ENABLED    0
ENV TRITON_SERVER_CPU_ONLY      1



ADD ./examples/ /ppml/examples
ADD ./entrypoint.sh /opt/entrypoint.sh
ADD ./start-scripts/ /ppml/work/start-scripts
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /sbin/tini
COPY --from=torchserve /ppml/torchserve/frontend.jar /ppml/torchserve/frontend.jar
COPY --from=min_container /usr/local/cuda/lib64/stubs/libcusparse.so /usr/local/cuda/lib64/stubs/libcusparse.so.11
COPY --from=min_container /usr/local/cuda/lib64/stubs/libcusolver.so /usr/local/cuda/lib64/stubs/libcusolver.so.11
COPY --from=min_container /usr/local/cuda/lib64/stubs/libcurand.so /usr/local/cuda/lib64/stubs/libcurand.so.10
COPY --from=min_container /usr/local/cuda/lib64/stubs/libcufft.so /usr/local/cuda/lib64/stubs/libcufft.so.10
COPY --from=min_container /usr/local/cuda/lib64/stubs/libcublas.so /usr/local/cuda/lib64/stubs/libcublas.so.11
COPY --from=min_container /usr/local/cuda/lib64/stubs/libcublasLt.so /usr/local/cuda/lib64/stubs/libcublasLt.so.11
COPY --from=min_container /usr/local/cuda-11.8/targets/x86_64-linux/lib/libcudart.so.11.0 /usr/local/cuda/targets/x86_64-linux/lib/.
COPY --from=min_container /usr/local/cuda-11.8/targets/x86_64-linux/lib/libcupti.so.11.8 /usr/local/cuda/targets/x86_64-linux/lib/.
COPY --from=min_container /usr/local/cuda-11.8/targets/x86_64-linux/lib/libnvToolsExt.so.1 /usr/local/cuda/targets/x86_64-linux/lib/.
COPY --from=min_container /usr/lib/x86_64-linux-gnu/libcudnn.so.8 /usr/lib/x86_64-linux-gnu/libcudnn.so.8
COPY --from=min_container /usr/lib/x86_64-linux-gnu/libnccl.so.2 /usr/lib/x86_64-linux-gnu/libnccl.so.2
COPY --from=tritonserver /workspace/server/docker/entrypoint.d/ /opt/nvidia/entrypoint.d/
COPY --from=tritonserver /workspace/server/docker/cpu_only/ /opt/nvidia/
COPY --from=tritonserver /tmp/tritonbuild/install /opt/tritonserver
COPY --from=tritonserver /workspace/server/NVIDIA_Deep_Learning_Container_License.pdf /opt/tritonserver
COPY --from=tritonserver /workspace/server/docker/sagemaker/serve /usr/bin/.

# Ensure apt-get won't prompt for selecting options

# Common dependencies. 
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common libb64-0d libcurl4-openssl-dev libre2-5 git dirmngr libnuma-dev curl libgomp1 && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /usr/local/cuda/lib64/stubs && \
    mkdir -p /usr/local/cuda/targets/x86_64-linux/lib && \

# patchelf is needed to add deps of libcublasLt.so.11 to libtorch_cuda.so
    apt-get update && \
    apt-get install -y --no-install-recommends openmpi-bin patchelf && \
    patchelf --add-needed /usr/local/cuda/lib64/stubs/libcublasLt.so.11 backends/pytorch/libtorch_cuda.so && \

    pip3 install --upgrade pip && \
    pip install --no-cache-dir flask && \
    pip install --no-cache-dir numpy && \
    pip install --no-cache-dir pandas && \
    pip install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir future cython requests pillow==9.0.1 captum packaging numpy nvgpu pyyaml && \
    pip install --no-cache-dir torchserve==0.6.1 torch-model-archiver==0.6.1 torch-workflow-archiver==0.2.5 && \
    apt-get install -y openjdk-${JDK_VERSION}-jdk && \
    cp -r /usr/local/lib/python3.7/dist-packages/* /usr/local/lib/python3.8/dist-packages/ && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.8 /usr/bin/python3 && \
    mkdir -p ${JAVA_HOME} && \
    cp -r /usr/lib/jvm/java-${JDK_VERSION}-openjdk-amd64/* ${JAVA_HOME} && \
    sed -i '/MAX_FAILURE_THRESHOLD = 5/ios.environ\[\"MPLCONFIGDIR\"\]=\"\/tmp\/matplotlib\"' /usr/local/lib/python3.7/dist-packages/ts/model_service_worker.py && \
    sed -i '/import abc/iimport sys' /usr/local/lib/python3.7/dist-packages/ts/torch_handler/base_handler.py && \
    sed -i '/module = importlib.import_module/i\ \ \ \ \ \ \ \ sys.path.append(model_dir)' /usr/local/lib/python3.7/dist-packages/ts/torch_handler/base_handler.py && \
    sed -i 's/SOCKET_ACCEPT_TIMEOUT = 30.0/SOCKET_ACCEPT_TIMEOUT = 3000.0/' /usr/local/lib/python3.7/dist-packages/ts/model_service_worker.py && \
    mkdir -p /ppml/tests/numpy && \
    mkdir -p /ppml/tests/pandas && \
    cp /usr/local/lib/python3.7/dist-packages/ts/configs/metrics.yaml /ppml && \
    chmod +x /opt/entrypoint.sh && \
    chmod +x /sbin/tini && \
    chmod +x /ppml/work/start-scripts/start-python-flask-sgx.sh && \
    chmod +x /ppml/work/start-scripts/start-python-numpy-example-sgx.sh && \
    chmod +x /ppml/work/start-scripts/start-python-numpy-performance-sgx.sh && \
    chmod +x /ppml/work/start-scripts/start-python-pandas-example-sgx.sh && \
    chmod +x /ppml/work/start-scripts/start-python-pandas-performance-sgx.sh && \
    chmod +x /ppml/work/start-scripts/start-torchserve-sgx.sh && \
    chmod +x /ppml/work/start-scripts/start-frontend-sgx.sh && \
    chmod +x /ppml/work/start-scripts/start-backend-sgx.sh && \
    cp /sbin/tini /usr/bin/tini && \
    gramine-argv-serializer bash -c 'export TF_MKL_ALLOC_MAX_BYTES=10737418240 && export _SPARK_AUTH_SECRET=$_SPARK_AUTH_SECRET && $sgx_command' > /ppml/secured_argvs


WORKDIR /ppml

ENTRYPOINT [ "/opt/entrypoint.sh" ]


