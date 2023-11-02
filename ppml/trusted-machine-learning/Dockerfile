ARG BASE_IMAGE_NAME=intelanalytics/bigdl-ppml-gramine-base
ARG BASE_IMAGE_TAG=2.3.0-SNAPSHOT
ARG BIGDATA_IMAGE_NAME=intelanalytics/bigdl-ppml-trusted-bigdata-gramine-base
ARG BIGDATA_IMAGE_TAG=2.3.0-SNAPSHOT
ARG SPARK_VERSION=3.1.3
ARG TINI_VERSION=v0.18.0
ARG JDK_VERSION=8u192
ARG JDK_URL
ARG SPARK_JAR_REPO_URL
ARG SCALA_VERSION=2.12
ARG LGBM_NETWORK_MODE_BUILD

# refer to ppml bigdata image
FROM $BIGDATA_IMAGE_NAME:$BIGDATA_IMAGE_TAG as bigdata

# stage.1 gramine
FROM $BASE_IMAGE_NAME:$BASE_IMAGE_TAG

ARG SPARK_VERSION
ARG TINI_VERSION
ARG http_proxy
ARG https_proxy
ARG HTTP_PROXY_HOST
ARG HTTP_PROXY_PORT
ARG HTTPS_PROXY_HOST
ARG HTTPS_PROXY_PORT
ARG LGBM_NETWORK_MODE_BUILD
ENV SPARK_VERSION                       ${SPARK_VERSION}
ENV SPARK_HOME                          /ppml/spark-${SPARK_VERSION}
ENV LOCAL_IP                            127.0.0.1
ENV TINI_VERSION                        $TINI_VERSION
ENV LC_ALL                              C.UTF-8
ENV LANG                                C.UTF-8
ENV LGBM_NETWORK_MODE_BUILD             ${LGBM_NETWORK_MODE_BUILD}
ENV LGBM_HOME                           /ppml/LightGBM
ENV LOCAL_IP                            127.0.0.1

RUN mkdir -p /ppml/lib && \
    mkdir -p /ppml/keys && \
    mkdir -p /ppml/password && \
    mkdir -p /ppml/data && \
    mkdir -p /ppml/models && \
    mkdir -p /ppml/apps

COPY --from=bigdata  /ppml/spark-${SPARK_VERSION} /ppml/spark-${SPARK_VERSION}
COPY --from=bigdata  /ppml/examples/src/main/resources /ppml/examples/src/main/resources
COPY --from=bigdata  /usr/local/bin/gosu /usr/local/bin/gosu

COPY ./bash.manifest.template /ppml
COPY ./SparkML/bigdl-ppml-submit.sh /ppml/bigdl-ppml-submit.sh
COPY ./SparkML/scripts /ppml/scripts
COPY ./LightGBM/examples/encrypted_lightgbm_model_io.py /ppml/examples
COPY ./SparkML/spark-executor-template.yaml /ppml/spark-executor-template.yaml
COPY ./SparkML/spark-driver-template.yaml /ppml/spark-driver-template.yaml
COPY ./SparkML/entrypoint.sh /opt/entrypoint.sh
COPY ./LightGBM/make-lgbm.sh /ppml/make-lgbm.sh
COPY ./LightGBM/lgbm-training-entrypoint.sh /ppml/lgbm-training-entrypoint.sh
COPY ./LightGBM/trainer.conf /ppml/trainer.conf

ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /sbin/tini
RUN echo $http_proxy && \
    echo $https_proxy

RUN rm $SPARK_HOME/jars/okhttp-*.jar && \
    wget -P $SPARK_HOME/jars https://repo1.maven.org/maven2/com/squareup/okhttp3/okhttp/3.8.0/okhttp-3.8.0.jar && \
    wget -P $SPARK_HOME/jars https://github.com/xerial/sqlite-jdbc/releases/download/3.36.0.1/sqlite-jdbc-3.36.0.1.jar && \
    chmod +x /opt/entrypoint.sh && \
    chmod +x /sbin/tini && \
    chmod +x /ppml/bigdl-ppml-submit.sh && \
    cp /sbin/tini /usr/bin/tini && \
    gramine-argv-serializer bash -c 'export TF_MKL_ALLOC_MAX_BYTES=10737418240 && export _SPARK_AUTH_SECRET=$_SPARK_AUTH_SECRET && $sgx_command' > /ppml/secured_argvs && \
    wget -P $SPARK_HOME/jars https://repo1.maven.org/maven2/mysql/mysql-connector-java/8.0.28/mysql-connector-java-8.0.28.jar
#LightGBM
RUN git clone --recursive https://github.com/analytics-zoo/LightGBM.git && \
    apt-get install libssl-dev  && \
    bash /ppml/make-lgbm.sh && \
    pip install synapseml==0.11.1 && \
    cp /ppml/jars/*.jar ${SPARK_HOME}/jars && \
    rm -rf ${SPARK_HOME}/jars/lightgbmlib* && \
    cp ${LGBM_HOME}/build/lightgbmlib.jar ${SPARK_HOME}/jars/

COPY --from=bigdata /ppml/bigdl-ppml/src /ppml/bigdl-ppml/src
ENV PYTHONPATH   /usr/lib/python3.9:/usr/lib/python3.9/lib-dynload:/usr/local/lib/python3.9/dist-packages:/usr/lib/python3/dist-packages:/ppml/bigdl-ppml/src

ENTRYPOINT [ "/opt/entrypoint.sh" ]
