ARG BASE_IMAGE_NAME=intelanalytics/bigdl-ppml-gramine-base
ARG BASE_IMAGE_TAG=2.2.0-SNAPSHOT
ARG TINI_VERSION=v0.18.0

# stage 1: flink
FROM eclipse-temurin:8-jre-jammy as flink

# Install dependencies
RUN set -ex; \
  apt-get update; \
  apt-get -y install gpg libsnappy1v5 gettext-base libjemalloc-dev; \
  rm -rf /var/lib/apt/lists/*

# Grab gosu for easy step-down from root
ENV GOSU_VERSION 1.11
RUN set -ex; \
  wget -nv -O /usr/local/bin/gosu "https://github.com/tianon/gosu/releases/download/$GOSU_VERSION/gosu-$(dpkg --print-architecture)"; \
  chmod +x /usr/local/bin/gosu; \
  gosu nobody true

# Configure Flink version
ENV FLINK_TGZ_URL=https://www.apache.org/dyn/closer.cgi?action=download&filename=flink/flink-1.15.2/flink-1.15.2-bin-scala_2.12.tgz \
    FLINK_ASC_URL=https://www.apache.org/dist/flink/flink-1.15.2/flink-1.15.2-bin-scala_2.12.tgz.asc \
    GPG_KEY=0F79F2AFB2351BC29678544591F9C1EC125FD8DB \
    CHECK_GPG=true

# Prepare environment
# ENV FLINK_HOME=/opt/flink
# ENV PATH=$FLINK_HOME/bin:$PATH
# RUN groupadd --system --gid=9999 flink && \
#     useradd --system --home-dir $FLINK_HOME --uid=9999 --gid=flink flink
# WORKDIR $FLINK_HOME

# Install Flink
RUN set -ex; \
  cd /opt/flink; \
  wget -nv -O flink.tgz "$FLINK_TGZ_URL"; \
  tar -xf flink.tgz --strip-components=1; \
  rm flink.tgz; \
  \
  # Replace default REST/RPC endpoint bind address to use the container's network interface \
  sed -i 's/rest.address: localhost/rest.address: 0.0.0.0/g' $FLINK_HOME/conf/flink-conf.yaml; \
  sed -i 's/rest.bind-address: localhost/rest.bind-address: 0.0.0.0/g' $FLINK_HOME/conf/flink-conf.yaml; \
  sed -i 's/jobmanager.bind-host: localhost/jobmanager.bind-host: 0.0.0.0/g' $FLINK_HOME/conf/flink-conf.yaml; \
  sed -i 's/taskmanager.bind-host: localhost/taskmanager.bind-host: 0.0.0.0/g' $FLINK_HOME/conf/flink-conf.yaml; \
  sed -i '/taskmanager.host: localhost/d' $FLINK_HOME/conf/flink-conf.yaml;


# stage.2 gramine
FROM $BASE_IMAGE_NAME:$BASE_IMAGE_TAG

ARG TINI_VERSION

ENV LOCAL_IP                            127.0.0.1
ENV TINI_VERSION                        $TINI_VERSION
ENV LC_ALL                              C.UTF-8
ENV LANG                                C.UTF-8

RUN mkdir -p /ppml/lib && \
    mkdir -p /ppml/keys && \
    mkdir -p /ppml/password && \
    mkdir -p /ppml/data && \
    mkdir -p /ppml/models && \
    mkdir -p /ppml/apps

# add environment from flink layer
COPY --from=flink /opt/flink /ppml/flink
COPY --from=flink /usr/local/bin/gosu /usr/local/bin/gosu
ENV FLINK_HOME=/ppml/flink
ENV PATH=$FLINK_HOME/bin:$PATH
RUN set -ex && \ 
    groupadd --system --gid=9999 flink && \
    useradd --system --home-dir $FLINK_HOME --uid=9999 --gid=flink flink && \
    chown -R flink:flink /ppml/flink && \
    chmod +x /usr/local/bin/gosu; \
    gosu nobody true


# ADD ./bigdl-ppml-submit.sh /ppml/bigdl-ppml-submit.sh
# ADD ./spark-executor-template.yaml /ppml/spark-executor-template.yaml
# ADD ./spark-driver-template.yaml /ppml/spark-driver-template.yaml
ADD ./entrypoint.sh /opt/entrypoint.sh
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /sbin/tini
RUN rm $SPARK_HOME/jars/okhttp-*.jar && \
    wget -P $SPARK_HOME/jars https://repo1.maven.org/maven2/com/squareup/okhttp3/okhttp/3.8.0/okhttp-3.8.0.jar && \
    wget -P $SPARK_HOME/jars https://github.com/xerial/sqlite-jdbc/releases/download/3.36.0.1/sqlite-jdbc-3.36.0.1.jar && \
    chmod +x /opt/entrypoint.sh && \
    chmod +x /sbin/tini && \
    chmod +x /ppml/bigdl-ppml-submit.sh && \
    cp /sbin/tini /usr/bin/tini && \
    gramine-argv-serializer bash -c 'export TF_MKL_ALLOC_MAX_BYTES=10737418240 && export _SPARK_AUTH_SECRET=$_SPARK_AUTH_SECRET && $sgx_command' > /ppml/secured_argvs


# Configure container

EXPOSE 6123 8081
ENTRYPOINT [ "/opt/entrypoint.sh" ]