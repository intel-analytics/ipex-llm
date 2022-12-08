ARG BASE_IMAGE_NAME
ARG BASE_IMAGE_TAG
ARG BIGDL_VERSION=2.2.0-SNAPSHOT
ARG TINI_VERSION=v0.18.0
ARG JDK_VERSION=11

#Stage.1 Torchserve Frontend
FROM $BASE_IMAGE_NAME:$BASE_IMAGE_TAG as temp
ARG http_proxy
ARG https_proxy
ARG JDK_VERSION
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

#Stage.1 Flask & Numpy & Pandas
FROM $BASE_IMAGE_NAME:$BASE_IMAGE_TAG
ARG http_proxy
ARG https_proxy
ARG JDK_VERSION
ARG TINI_VERSION
ENV JAVA_HOME				/opt/jdk${JDK_VERSION}
ENV TINI_VERSION                        $TINI_VERSION

ADD ./examples/ /ppml/examples
ADD ./entrypoint.sh /opt/entrypoint.sh
ADD ./start-scripts/ /ppml/work/start-scripts
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /sbin/tini
COPY --from=temp /ppml/torchserve/frontend.jar /ppml/torchserve/frontend.jar
COPY --from=temp /opt/jdk11/* /opt/jdk11/
RUN pip3 install --upgrade pip && \
    pip install --no-cache-dir flask && \
    pip install --no-cache-dir numpy && \
    pip install --no-cache-dir pandas && \
    pip install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir future cython requests pillow==9.0.1 captum packaging numpy nvgpu && \
    pip install --no-cache-dir torchserve==0.6.1 torch-model-archiver==0.6.1 torch-workflow-archiver==0.2.5 && \
    sed -i '/module = importlib.import_module/i\\tsys.path.append(model_dir)' /usr/local/lib/python3.7/dist-packages/ts/torch_handler/base_handler.py && \
    sed -i 's/SOCKET_ACCEPT_TIMEOUT = 30.0/SOCKET_ACCEPT_TIMEOUT = 300.0/' /usr/local/lib/python3.7/dist-packages/ts/model_service_worker.py && \
    mkdir -p /ppml/tests/numpy && \
    mkdir -p /ppml/tests/pandas && \
    cp /usr/local/lib/python3.7/dist-packages/ts/configs/metrics.yaml /ppml && \
    chmod +x /opt/entrypoint.sh && \
    chmod +x /sbin/tini && \
    chmod +x /ppml/work/start-scripts/start-python-flask-sgx.sh && \
    chmod +x /ppml/work/start-scripts/start-python-numpy-sgx.sh && \
    chmod +x /ppml/work/start-scripts/start-python-pandas-sgx.sh && \
    chmod +x /ppml/work/start-scripts/start-torchserve-sgx.sh && \
    chmod +x /ppml/work/start-scripts/start-frontend-sgx.sh && \
    chmod +x /ppml/work/start-scripts/start-backend-sgx.sh && \
    cp /sbin/tini /usr/bin/tini && \
    gramine-argv-serializer bash -c 'export TF_MKL_ALLOC_MAX_BYTES=10737418240 && export _SPARK_AUTH_SECRET=$_SPARK_AUTH_SECRET && $sgx_command' > /ppml/secured_argvs

WORKDIR /ppml

ENTRYPOINT [ "/opt/entrypoint.sh" ]

