ARG BASE_IMAGE_NAME=intelanalytics/bigdl-ppml-gramine-base
ARG BASE_IMAGE_TAG=2.3.0-SNAPSHOT
ARG BIGDL_VERSION=2.3.0-SNAPSHOT
ARG SPARK_VERSION=3.1.3
ARG TINI_VERSION=v0.18.0
ARG JDK_VERSION=8u192
ARG JDK_URL
ARG SPARK_JAR_REPO_URL
ARG FLINK_VERSION=1.15.3
ARG SCALA_VERSION=2.12


# Stage.1 Spark & Hadoop & Hive & Flink
FROM ubuntu:20.04 as bigdata
ARG HTTP_PROXY_HOST
ARG HTTP_PROXY_PORT
ARG HTTPS_PROXY_HOST
ARG HTTPS_PROXY_PORT
ARG SPARK_VERSION
ARG JDK_VERSION
ARG JDK_URL
ARG SPARK_JAR_REPO_URL
ARG FLINK_VERSION
ARG SCALA_VERSION

ENV SPARK_VERSION                       ${SPARK_VERSION}
ENV JAVA_HOME                           /opt/jdk${JDK_VERSION}
ENV PATH                                ${JAVA_HOME}/bin:${PATH}
ENV FLINK_HOME                          /opt/flink
ENV GOSU_VERSION                        1.11

RUN apt-get update --fix-missing && \
    env DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y tzdata apt-utils wget unzip patch zip git maven nasm
# java
RUN wget $JDK_URL && \
    gunzip jdk-$JDK_VERSION-linux-x64.tar.gz && \
    tar -xf jdk-$JDK_VERSION-linux-x64.tar -C /opt && \
    rm jdk-$JDK_VERSION-linux-x64.tar && \
    mv /opt/jdk* /opt/jdk$JDK_VERSION && \
    ln -s /opt/jdk$JDK_VERSION /opt/jdk

# spark
RUN cd /opt && \
    wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.2.tgz && \
    tar -zxvf spark-${SPARK_VERSION}-bin-hadoop3.2.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop3.2 spark-${SPARK_VERSION} && \
    rm spark-${SPARK_VERSION}-bin-hadoop3.2.tgz && \
    cp spark-${SPARK_VERSION}/conf/log4j.properties.template spark-${SPARK_VERSION}/conf/log4j.properties && \
    echo $'\nlog4j.logger.io.netty=ERROR' >> spark-${SPARK_VERSION}/conf/log4j.properties \
    rm spark-${SPARK_VERSION}/python/lib/pyspark.zip && \
    rm spark-${SPARK_VERSION}/jars/spark-core_2.12-$SPARK_VERSION.jar && \
    rm spark-${SPARK_VERSION}/jars/spark-launcher_2.12-$SPARK_VERSION.jar && \
    rm spark-${SPARK_VERSION}/jars/spark-kubernetes_2.12-$SPARK_VERSION.jar && \
    rm spark-${SPARK_VERSION}/jars/spark-network-common_2.12-$SPARK_VERSION.jar && \
    rm spark-${SPARK_VERSION}/examples/jars/spark-examples_2.12-$SPARK_VERSION.jar && \
    rm spark-${SPARK_VERSION}/jars/hadoop-common-3.2.0.jar && \
    rm spark-${SPARK_VERSION}/jars/hive-exec-2.3.7-core.jar
ADD ./log4j2.xml /opt/spark-${SPARK_VERSION}/conf/log4j2.xml
# spark modification
RUN cd /opt && \
    wget $SPARK_JAR_REPO_URL/spark-core_2.12-$SPARK_VERSION.jar && \
    wget $SPARK_JAR_REPO_URL/spark-kubernetes_2.12-$SPARK_VERSION.jar && \
    wget $SPARK_JAR_REPO_URL/spark-network-common_2.12-$SPARK_VERSION.jar && \
    wget $SPARK_JAR_REPO_URL/spark-examples_2.12-$SPARK_VERSION.jar && \
    wget $SPARK_JAR_REPO_URL/spark-launcher_2.12-$SPARK_VERSION.jar && \
    wget $SPARK_JAR_REPO_URL/pyspark.zip && \
    mv /opt/spark-core_2.12-$SPARK_VERSION.jar  /opt/spark-${SPARK_VERSION}/jars/spark-core_2.12-$SPARK_VERSION.jar && \
    mv /opt/spark-launcher_2.12-$SPARK_VERSION.jar /opt/spark-${SPARK_VERSION}/jars/spark-launcher_2.12-$SPARK_VERSION.jar && \
    mv /opt/spark-kubernetes_2.12-$SPARK_VERSION.jar /opt/spark-${SPARK_VERSION}/jars/spark-kubernetes_2.12-$SPARK_VERSION.jar && \
    mv /opt/spark-network-common_2.12-$SPARK_VERSION.jar /opt/spark-${SPARK_VERSION}/jars/spark-network-common_2.12-$SPARK_VERSION.jar && \
    mv /opt/spark-examples_2.12-$SPARK_VERSION.jar /opt/spark-${SPARK_VERSION}/examples/jars/spark-examples_2.12-$SPARK_VERSION.jar && \
    mv /opt/pyspark.zip /opt/spark-${SPARK_VERSION}/python/lib/pyspark.zip && \
    sed -i 's/\#\!\/usr\/bin\/env bash/\#\!\/usr\/bin\/env bash\nset \-x/' /opt/spark-${SPARK_VERSION}/bin/spark-class && \
    rm -f /opt/spark-${SPARK_VERSION}/jars/log4j-1.2.17.jar && \
    rm -f /opt/spark-${SPARK_VERSION}/jars/slf4j-log4j12-1.7.16.jar && \
    rm -f /opt/spark-${SPARK_VERSION}/jars/apache-log4j-extras-1.2.17.jar && \
    rm -r /opt/spark-${SPARK_VERSION}/jars/slf4j-log4j12-1.7.30.jar && \
    wget -P /opt/spark-${SPARK_VERSION}/jars/ https://repo1.maven.org/maven2/org/apache/logging/log4j/log4j-1.2-api/2.17.1/log4j-1.2-api-2.17.1.jar && \
    wget -P /opt/spark-${SPARK_VERSION}/jars/ https://repo1.maven.org/maven2/org/slf4j/slf4j-reload4j/1.7.35/slf4j-reload4j-1.7.35.jar && \
    wget -P /opt/spark-${SPARK_VERSION}/jars/ https://repo1.maven.org/maven2/org/apache/logging/log4j/log4j-api/2.17.1/log4j-api-2.17.1.jar && \
    wget -P /opt/spark-${SPARK_VERSION}/jars/ https://repo1.maven.org/maven2/org/apache/logging/log4j/log4j-core/2.17.1/log4j-core-2.17.1.jar && \
    wget -P /opt/spark-${SPARK_VERSION}/jars/ https://repo1.maven.org/maven2/org/apache/logging/log4j/log4j-slf4j-impl/2.17.1/log4j-slf4j-impl-2.17.1.jar && \
    wget -P /opt/spark-${SPARK_VERSION}/jars/ https://repo1.maven.org/maven2/org/wildfly/openssl/wildfly-openssl/1.0.7.Final/wildfly-openssl-1.0.7.Final.jar && \
    wget -P /opt/spark-${SPARK_VERSION}/jars/ https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-azure/3.2.0/hadoop-azure-3.2.0.jar && \
    wget -P /opt/spark-${SPARK_VERSION}/jars/ https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-azure-datalake/3.2.0/hadoop-azure-datalake-3.2.0.jar && \
    wget -P /opt/spark-${SPARK_VERSION}/jars/ https://repo1.maven.org/maven2/com/microsoft/azure/azure-storage/7.0.0/azure-storage-7.0.0.jar && \
    wget -P /opt/spark-${SPARK_VERSION}/jars/ https://repo1.maven.org/maven2/com/microsoft/azure/azure-data-lake-store-sdk/2.2.9/azure-data-lake-store-sdk-2.2.9.jar
# hadoop
RUN cd /opt && \
    apt-get update --fix-missing && \
    apt-get install -y build-essential && \
    wget https://github.com/protocolbuffers/protobuf/releases/download/v2.5.0/protobuf-2.5.0.tar.bz2 && \
    tar jxvf protobuf-2.5.0.tar.bz2 && \
    cd protobuf-2.5.0 && \
    ./configure && \
    make && \
    make check && \
    export LD_LIBRARY_PATH=/usr/local/lib && \
    make install && \
    protoc --version && \
    cd /opt && \
    git clone https://github.com/analytics-zoo/hadoop.git && \
    cd hadoop && \
    git checkout branch-3.2.0-ppml && \
    cd hadoop-common-project/hadoop-common && \
    export MAVEN_OPTS="-Xmx2g -XX:ReservedCodeCacheSize=512m \
        -Dhttp.proxyHost=$HTTP_PROXY_HOST \
        -Dhttp.proxyPort=$HTTP_PROXY_PORT \
        -Dhttps.proxyHost=$HTTPS_PROXY_HOST \
        -Dhttps.proxyPort=$HTTPS_PROXY_PORT" && \
    mvn -T 16 -DskipTests=true clean package && \
    mv /opt/hadoop/hadoop-common-project/hadoop-common/target/hadoop-common-3.2.0.jar /opt/spark-${SPARK_VERSION}/jars/hadoop-common-3.2.0.jar
# hive
RUN cd /opt && \
    git clone https://github.com/analytics-zoo/hive.git && \
    cd hive && \
    git checkout branch-2.3.7-ppml && \
    cd ql && \
    export MAVEN_OPTS="-Xmx2g -XX:ReservedCodeCacheSize=512m \
        -Dhttp.proxyHost=$HTTP_PROXY_HOST \
        -Dhttp.proxyPort=$HTTP_PROXY_PORT \
        -Dhttps.proxyHost=$HTTPS_PROXY_HOST \
        -Dhttps.proxyPort=$HTTPS_PROXY_PORT" && \
    mvn -T 16 -DskipTests=true clean package && \
    mv /opt/hive/ql/target/hive-exec-2.3.7-core.jar /opt/spark-${SPARK_VERSION}/jars/hive-exec-2.3.7-core.jar

# flink
RUN wget -nv -O /usr/local/bin/gosu "https://github.com/tianon/gosu/releases/download/$GOSU_VERSION/gosu-$(dpkg --print-architecture)" && \
  chmod +x /usr/local/bin/gosu && \
  gosu nobody true && \
  mkdir -p $FLINK_HOME && \
  cd $FLINK_HOME && \
  wget -nv -O flink.tgz "https://www.apache.org/dyn/closer.cgi?action=download&filename=flink/flink-${FLINK_VERSION}/flink-${FLINK_VERSION}-bin-scala_${SCALA_VERSION}.tgz" && \
  tar -xf flink.tgz --strip-components=1 && \
  rm flink.tgz && \
# Replace default REST/RPC endpoint bind address to use the container's network interface
  sed -i 's/rest.address: localhost/rest.address: 0.0.0.0/g' $FLINK_HOME/conf/flink-conf.yaml && \
  sed -i 's/rest.bind-address: localhost/rest.bind-address: 0.0.0.0/g' $FLINK_HOME/conf/flink-conf.yaml && \
  sed -i 's/jobmanager.bind-host: localhost/jobmanager.bind-host: 0.0.0.0/g' $FLINK_HOME/conf/flink-conf.yaml && \
  sed -i 's/taskmanager.bind-host: localhost/taskmanager.bind-host: 0.0.0.0/g' $FLINK_HOME/conf/flink-conf.yaml && \
  sed -i '/taskmanager.host: localhost/d' $FLINK_HOME/conf/flink-conf.yaml
RUN ls -al /opt

# Stage.2 BigDL
FROM ubuntu:20.04 as bigdl
ARG BIGDL_VERSION
ARG SPARK_VERSION
ENV SPARK_VERSION               ${SPARK_VERSION}
ENV BIGDL_VERSION               ${BIGDL_VERSION}
ENV BIGDL_HOME                  /bigdl-${BIGDL_VERSION}
RUN apt-get update --fix-missing && \
    apt-get install -y apt-utils curl wget unzip git
RUN wget https://raw.githubusercontent.com/intel-analytics/analytics-zoo/bigdl-2.0/docker/hyperzoo/download-bigdl.sh && \
    chmod a+x ./download-bigdl.sh
RUN ./download-bigdl.sh && \
    rm bigdl*.zip

# stage.3 gramine
FROM $BASE_IMAGE_NAME:$BASE_IMAGE_TAG

ARG SPARK_VERSION
ARG TINI_VERSION

ENV FLINK_HOME                          /ppml/flink
ENV SPARK_VERSION                       ${SPARK_VERSION}
ENV SPARK_HOME                          /ppml/spark-${SPARK_VERSION}
ENV LOCAL_IP                            127.0.0.1
ENV TINI_VERSION                        $TINI_VERSION
ENV LC_ALL                              C.UTF-8
ENV LANG                                C.UTF-8
ENV PATH                                $FLINK_HOME/bin:$PATH
ENV BIGDL_HOME                          /ppml/bigdl-${BIGDL_VERSION}
ENV PYSPARK_PYTHON                      /usr/bin/python

RUN mkdir -p /ppml/lib && \
    mkdir -p /ppml/keys && \
    mkdir -p /ppml/password && \
    mkdir -p /ppml/data && \
    mkdir -p /ppml/models && \
    mkdir -p /ppml/apps

COPY --from=bigdata /opt/spark-${SPARK_VERSION} /ppml/spark-${SPARK_VERSION}
COPY --from=bigdata /opt/spark-${SPARK_VERSION}/examples/src/main/resources /ppml/examples/src/main/resources
COPY --from=bigdata /opt/flink $FLINK_HOME
COPY --from=bigdata /usr/local/bin/gosu /usr/local/bin/gosu
COPY --from=bigdl /bigdl-${BIGDL_VERSION} ${BIGDL_HOME}

ADD ./bigdl-ppml-submit.sh /ppml/bigdl-ppml-submit.sh
ADD ./scripts /ppml/scripts
ADD ./spark-executor-template.yaml /ppml/spark-executor-template.yaml
ADD ./spark-driver-template.yaml /ppml/spark-driver-template.yaml
ADD ./entrypoint.sh /opt/entrypoint.sh
ADD ./flink-entrypoint.sh /opt/flink-entrypoint.sh
ADD ./flink-k8s-template.yaml /ppml/flink-k8s-template.yaml
ADD ./examples /ppml/examples
ADD ./zeppelin /ppml/zeppelin
ADD ./spark-executor-template-for-tdxvm.yaml /ppml/spark-executor-template-for-tdxvm.yaml
ADD ./spark-driver-template-for-tdxvm.yaml /ppml/spark-driver-template-for-tdxvm.yaml

ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /sbin/tini
RUN rm $SPARK_HOME/jars/okhttp-*.jar && \
    wget -P $SPARK_HOME/jars https://repo1.maven.org/maven2/com/squareup/okhttp3/okhttp/3.8.0/okhttp-3.8.0.jar && \
    wget -P $SPARK_HOME/jars https://github.com/xerial/sqlite-jdbc/releases/download/3.36.0.1/sqlite-jdbc-3.36.0.1.jar && \
    chmod +x /opt/entrypoint.sh && \
    chmod +x /sbin/tini && \
    chmod +x /ppml/bigdl-ppml-submit.sh && \
    cp /sbin/tini /usr/bin/tini && \
    gramine-argv-serializer bash -c 'export TF_MKL_ALLOC_MAX_BYTES=10737418240 && export _SPARK_AUTH_SECRET=$_SPARK_AUTH_SECRET && $sgx_command' > /ppml/secured_argvs && \
    wget -P $SPARK_HOME/jars https://repo1.maven.org/maven2/mysql/mysql-connector-java/8.0.28/mysql-connector-java-8.0.28.jar && \
    chmod a+x /ppml/scripts/* && \
#flink
    env DEBIAN_FRONTEND=noninteractive apt-get install -y libsnappy1v5 gettext-base libjemalloc-dev && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd --system --gid=9999 flink && \
    useradd --system --home-dir ${FLINK_HOME} --uid=9999 --gid=flink flink && \
    chmod +x /usr/local/bin/gosu && \
    chmod +x /opt/flink-entrypoint.sh && \
    chmod -R 777 ${FLINK_HOME} && \
    chown -R flink:flink ${FLINK_HOME} && \
# Python packages
    pip3 install numpy pandas pyarrow&& \
    cp ${BIGDL_HOME}/jars/bigdl-ppml-spark_${SPARK_VERSION}-${BIGDL_VERSION}.jar ${SPARK_HOME}/jars/ && \
    cp ${BIGDL_HOME}/jars/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}.jar ${SPARK_HOME}/jars/ && \
# zeppelin
    chmod +x /ppml/zeppelin/deploy.sh && \
    chmod +x /ppml/zeppelin/delete.sh && \
# Azure support
    apt purge -y libsgx-dcap-default-qpl && \
    echo "deb [arch=amd64] https://packages.microsoft.com/ubuntu/20.04/prod focal main" | tee /etc/apt/sources.list.d/msprod.list && \
    wget -qO - https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
    apt update && \
    apt install -y az-dcap-client && \
    curl -sL https://aka.ms/InstallAzureCLIDeb | bash && \
    apt-get install bsdmainutils && \
    curl -LO https://dl.k8s.io/release/v1.25.0/bin/linux/amd64/kubectl  && \
    install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
ADD azure /ppml/azure
RUN chmod a+x /ppml/azure/create-aks.sh && \
    chmod a+x /ppml/azure/generate-keys-az.sh && \
    chmod a+x /ppml/azure/generate-password-az.sh && \
    chmod a+x /ppml/azure/kubeconfig-secret.sh && \
    chmod a+x /ppml/azure/submit-spark-sgx-az.sh && \
    wget -P /ppml/lib https://sourceforge.net/projects/analytics-zoo/files/analytics-zoo-data/libhadoop.so


ENTRYPOINT [ "/opt/entrypoint.sh" ]
