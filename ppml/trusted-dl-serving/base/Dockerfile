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
ENV JAVA_HOME               /opt/jdk${JDK_VERSION}

RUN apt-get install -y openjdk-${JDK_VERSION}-jdk && \
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
RUN mkdir /ppml/examples && \
    mkdir /ppml/torchserve

ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /sbin/tini
ADD ./entrypoint.sh /opt/entrypoint.sh

# COPY frontend.jar
COPY --from=temp /ppml/torchserve/frontend.jar /ppml/torchserve/frontend.jar
# Start Script for Torchserve
ADD ./start-torchserve-backend.sh      /ppml/torchserve/start-torchserve-backend.sh
ADD ./start-torchserve-frontend.sh     /ppml/torchserve/start-torchserve-frontend.sh
ADD ./start-torchserve.sh              /ppml/torchserve/start-torchserve.sh


# Dependencies
RUN env DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y libssl-dev && \
# Optimization related
    pip3 install --pre --no-cache --upgrade bigdl-nano[pytorch,inference]==2.2.0b20221226 && \
# Torchserve
    pip3 install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip3 install --no-cache-dir cython pillow==9.0.1 captum packaging nvgpu && \
    pip3 install --no-cache-dir torchserve==0.6.1 torch-model-archiver==0.6.1 torch-workflow-archiver==0.2.5 && \
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
    chmod +x /ppml/torchserve/start-torchserve-backend.sh && \
    chmod +x /ppml/torchserve/start-torchserve-frontend.sh && \
    chmod +x /ppml/torchserve/start-torchserve.sh && \
#generate secured_argvs
    gramine-argv-serializer bash -c 'export TF_MKL_ALLOC_MAX_BYTES=10737418240 && $sgx_command' > /ppml/secured_argvs && \
    chmod +x /sbin/tini && \
    chmod +x /opt/entrypoint.sh && \
    cp /sbin/tini /usr/bin/tini && \
# We need to downgrade markupsafe, the markupsafe required by bigdl-nano removed `soft_unicode`
# which is then required by our third-layer gramine make command
    pip3 install --no-cache markupsafe==2.0.1 pyarrow==6.0.1

ENTRYPOINT [ "/opt/entrypoint.sh" ]
