ARG BASE_IMAGE_NAME=intelanalytics/bigdl-ppml-gramine-base
ARG BASE_IMAGE_TAG=2.2.0-SNAPSHOT

# stage.1 flink
FROM eclipse-temurin:8-jre-jammy as flink

ARG FLINK_VERSION=1.15.3
ARG SCALA_VERSION=2.12
ARG FLINK_HOME=/opt/flink

#Grab gosu for easy step-down from root
ENV GOSU_VERSION 1.11
RUN set -ex; \
  wget -nv -O /usr/local/bin/gosu "https://github.com/tianon/gosu/releases/download/$GOSU_VERSION/gosu-$(dpkg --print-architecture)"; \
  chmod +x /usr/local/bin/gosu; \
  gosu nobody true

#Install Flink
RUN set -ex; \
  mkdir -p $FLINK_HOME; \
  cd $FLINK_HOME; \
  wget -nv -O flink.tgz "https://www.apache.org/dyn/closer.cgi?action=download&filename=flink/flink-${FLINK_VERSION}/flink-${FLINK_VERSION}-bin-scala_${SCALA_VERSION}.tgz"; \
  tar -xf flink.tgz --strip-components=1; \
  rm flink.tgz; \
  # Replace default REST/RPC endpoint bind address to use the container's network interface \
  sed -i 's/rest.address: localhost/rest.address: 0.0.0.0/g' $FLINK_HOME/conf/flink-conf.yaml; \
  sed -i 's/rest.bind-address: localhost/rest.bind-address: 0.0.0.0/g' $FLINK_HOME/conf/flink-conf.yaml; \
  sed -i 's/jobmanager.bind-host: localhost/jobmanager.bind-host: 0.0.0.0/g' $FLINK_HOME/conf/flink-conf.yaml; \
  sed -i 's/taskmanager.bind-host: localhost/taskmanager.bind-host: 0.0.0.0/g' $FLINK_HOME/conf/flink-conf.yaml; \
  sed -i '/taskmanager.host: localhost/d' $FLINK_HOME/conf/flink-conf.yaml;

FROM $BASE_IMAGE_NAME:$BASE_IMAGE_TAG
ARG SGX_MEM_SIZE     32G
ARG SGX_LOG_LEVEL    error

ENV LOCAL_IP                            127.0.0.1
ENV LC_ALL                              C.UTF-8
ENV LANG                                C.UTF-8
ENV FLINK_HOME                          /ppml/flink
ENV PATH                                $FLINK_HOME/bin:$PATH

# 1.1 add environment from flink layer
COPY --from=flink /opt/flink $FLINK_HOME
COPY --from=flink /usr/local/bin/gosu /usr/local/bin/gosu

RUN set -ex && \
    env DEBIAN_FRONTEND=noninteractive apt-get install -y libsnappy1v5 gettext-base libjemalloc-dev && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd --system --gid=9999 flink && \
    useradd --system --home-dir ${FLINK_HOME} --uid=9999 --gid=flink flink && \
    chmod +x /usr/local/bin/gosu && \
    chmod -R 777 ${FLINK_HOME} && \
    chown -R flink:flink ${FLINK_HOME}

# 2.1 make SGX and sign in a temp image
ADD ./enclave-key.pem /root/.config/gramine/enclave-key.pem

RUN cd /ppml && \
# generate secured_argvs
    gramine-argv-serializer bash -c 'export TF_MKL_ALLOC_MAX_BYTES=10737418240 && $sgx_command' > /ppml/secured_argvs && \
# make sgx
    echo SGX_MEM_SIZE:$SGX_MEM_SIZE && \
    echo SGX_LOG_LEVEL:$SGX_LOG_LEVEL && \
    make SGX=1 DEBUG=1 THIS_DIR=/ppml  SPARK_USER=root G_SGX_SIZE=$SGX_MEM_SIZE G_LOG_LEVEL=$SGX_LOG_LEVEL

# 2.2 output mr_enclave and mr_signer to customer
RUN cd /ppml && \
    echo "[INFO] Use the below hash values of mr_enclave and mr_signer to register enclave:" && \
    gramine-sgx-quote-dump -m bash.sig|grep mr_enclave && \
    gramine-sgx-quote-dump -m bash.sig|grep mr_signer


ADD ./entrypoint.sh /docker-entrypoint.sh
ADD ./k8s-template.yaml /ppml/k8s-template.yaml

RUN chmod +x /docker-entrypoint.sh

# Configure container
EXPOSE 6123 8081
WORKDIR /ppml
ENTRYPOINT [ "/docker-entrypoint.sh" ]
