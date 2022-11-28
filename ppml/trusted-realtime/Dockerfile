ARG BASE_IMAGE_NAME=intelanalytics/bigdl-ppml-gramine-base
ARG BASE_IMAGE_TAG=2.2.0-SNAPSHOT

# stage 1: flink
FROM eclipse-temurin:8-jre-jammy as flink

ARG FLINK_VERSION=1.15.2
ARG SCALA_VERSION=2.12

# Grab gosu for easy step-down from root
ENV GOSU_VERSION 1.11
RUN set -ex; \
  wget -nv -O /usr/local/bin/gosu "https://github.com/tianon/gosu/releases/download/$GOSU_VERSION/gosu-$(dpkg --print-architecture)"; \
  chmod +x /usr/local/bin/gosu; \
  gosu nobody true

# Install Flink
RUN set -ex; \
  mkdir /ppml/flink; \
  export FLINK_HOME=/ppml/flink; \
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


# stage.2 gramine
FROM $BASE_IMAGE_NAME:$BASE_IMAGE_TAG

ENV LOCAL_IP                            127.0.0.1
ENV LC_ALL                              C.UTF-8
ENV LANG                                C.UTF-8
ENV FLINK_HOME                          /ppml/flink
ENV PATH                                $FLINK_HOME/bin:$PATH

RUN mkdir -p /ppml/lib && \
    mkdir -p /ppml/keys && \
    mkdir -p /ppml/password && \
    mkdir -p /ppml/data && \
    mkdir -p /ppml/models && \
    mkdir -p /ppml/apps

# add environment from flink layer
COPY --from=flink /opt/flink /ppml/flink
COPY --from=flink /usr/local/bin/gosu /usr/local/bin/gosu

RUN set -ex && \
    env DEBIAN_FRONTEND=noninteractive apt-get install -y libsnappy1v5 gettext-base libjemalloc-dev
    
RUN set -ex && \ 
    groupadd --system --gid=9999 flink && \
    useradd --system --home-dir $FLINK_HOME --uid=9999 --gid=flink flink && \ 
    chmod +x /usr/local/bin/gosu; \
    chown -R flink:flink /ppml; \
    gosu nobody true
    chmod -R 777 ${FLINK_HOME}


ADD ./entrypoint.sh /docker-entrypoint.sh


# Configure container
EXPOSE 6123 8081
ENTRYPOINT [ "/docker-entrypoint.sh" ]
