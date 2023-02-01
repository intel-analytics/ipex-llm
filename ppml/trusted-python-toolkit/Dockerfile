ARG http_proxy
ARG https_proxy
ARG BASE_IMAGE_NAME
ARG BASE_IMAGE_TAG
ARG BIGDL_VERSION=2.3.0-SNAPSHOT
ARG TINI_VERSION=v0.18.0
ARG JDK_VERSION=11

#Stage.1 Torchserve Frontend
FROM $BASE_IMAGE_NAME:$BASE_IMAGE_TAG as temp
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

#Stage.2 Flask & Numpy & Pandas
FROM $BASE_IMAGE_NAME:$BASE_IMAGE_TAG
ARG http_proxy
ARG https_proxy
ARG JDK_VERSION
ARG TINI_VERSION
ENV JDK_HOME				/opt/jdk${JDK_VERSION}
ENV JAVA_HOME				/opt/jdk${JDK_VERSION}
ENV TINI_VERSION                        $TINI_VERSION

ADD ./examples/ /ppml/examples
ADD ./entrypoint.sh /opt/entrypoint.sh
ADD ./start-scripts/ /ppml/work/start-scripts
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /sbin/tini
COPY --from=temp /ppml/torchserve/frontend.jar /ppml/torchserve/frontend.jar
RUN pip3 install --upgrade pip && \
    pip install --no-cache-dir flask && \
    pip install --no-cache-dir numpy && \
    pip install --no-cache-dir pandas && \
    pip install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir future cython requests pillow==9.0.1 captum packaging numpy nvgpu pyyaml && \
    pip install --no-cache-dir torchserve==0.6.1 torch-model-archiver==0.6.1 torch-workflow-archiver==0.2.5 && \
    apt-get install -y openjdk-${JDK_VERSION}-jdk && \
    mkdir -p ${JAVA_HOME} && \
    cp -r /usr/lib/jvm/java-${JDK_VERSION}-openjdk-amd64/* ${JAVA_HOME} && \
    sed -i '/MAX_FAILURE_THRESHOLD = 5/ios.environ\[\"MPLCONFIGDIR\"\]=\"\/tmp\/matplotlib\"' /usr/local/lib/python3.7/dist-packages/ts/model_service_worker.py && \
    sed -i '/import json/iimport sys' /usr/local/lib/python3.7/dist-packages/ts/model_loader.py && \
    sed -i '/manifest_file\ =\ os.path.join/i\ \ \ \ \ \ \ \ sys.path.append(model_dir)' /usr/local/lib/python3.7/dist-packages/ts/model_loader.py && \
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

