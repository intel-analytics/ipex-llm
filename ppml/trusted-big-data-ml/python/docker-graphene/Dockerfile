# stage.1 graphene
FROM ubuntu:18.04 AS graphene
ARG GRAPHENE_BRANCH=branch-210913
RUN env DEBIAN_FRONTEND=noninteractive apt-get update && \
    env DEBIAN_FRONTEND=noninteractive apt-get install -y \
        autoconf bison build-essential coreutils gawk git wget \
        python3 python3-pip libcurl4-openssl-dev \
        libprotobuf-c-dev protobuf-c-compiler python3-protobuf wget
RUN git clone https://github.com/analytics-zoo/graphene.git /graphene
RUN cd /graphene && \
    git fetch origin $GRAPHENE_BRANCH && \
    git checkout $GRAPHENE_BRANCH
RUN pip3 install ninja meson && \
    python3 -m pip install toml==0.10.2 click jinja2
RUN cd /graphene/Pal/src/host/Linux-SGX && \
    git clone https://github.com/intel/SGXDataCenterAttestationPrimitives.git linux-sgx-driver && \
    cd linux-sgx-driver && \
    git checkout DCAP_1.7 && \
    cp -r driver/linux/* .
RUN cd /graphene && \
    make && \
    ISGX_DRIVER_PATH=/graphene/Pal/src/host/Linux-SGX/linux-sgx-driver make -s -j4 SGX=1 && \
    meson setup build/ --prefix="/graphene/meson_build_output" \ 
    --buildtype=release -Ddirect=enabled -Dsgx=enabled  && \
    ninja -C build/ && \
    ninja -C build/ install
RUN for f in $(find /graphene/Runtime -type l); do cp --remove-destination $(realpath $f) $f; done

# stage.2 jdk & scala & spark & hadoop
FROM ubuntu:18.04 as spark
ARG HTTP_PROXY_HOST
ARG HTTP_PROXY_PORT
ARG HTTPS_PROXY_HOST
ARG HTTPS_PROXY_PORT
ARG SPARK_VERSION=3.1.2
ARG JDK_VERSION=8u192
ARG JDK_URL=your_jdk_url
ENV SPARK_VERSION	${SPARK_VERSION}
ENV JAVA_HOME           /opt/jdk$JDK_VERSION
ENV PATH                ${JAVA_HOME}/bin:${PATH}

RUN apt-get update --fix-missing && \
    apt-get install -y apt-utils wget unzip patch zip git maven
ADD pyspark.rdd.patch /opt/rdd.patch
ADD pyspark.shuffle.patch /opt/shuffle.patch
ADD spark-3.1.2.patch /opt/spark-3.1.2.patch
ADD hadoop-common-shell.patch /opt/hadoop-common-shell.patch

# java
RUN wget $JDK_URL && \
    gunzip jdk-$JDK_VERSION-linux-x64.tar.gz && \
    tar -xf jdk-$JDK_VERSION-linux-x64.tar -C /opt && \
    rm jdk-$JDK_VERSION-linux-x64.tar && \
    mv /opt/jdk* /opt/jdk$JDK_VERSION && \
    ln -s /opt/jdk$JDK_VERSION /opt/jdk
# scala
RUN cd / && wget -c https://downloads.lightbend.com/scala/2.11.8/scala-2.11.8.tgz && \
    (cd / && gunzip < scala-2.11.8.tgz)|(cd /opt && tar -xvf -) && \
    rm /scala-2.11.8.tgz
# maven
RUN cd /opt && \
    wget https://archive.apache.org/dist/maven/maven-3/3.6.3/binaries/apache-maven-3.6.3-bin.tar.gz && \
    tar -zxvf apache-maven-3.6.3-bin.tar.gz
# spark
RUN cd /opt && \
    wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.2.tgz && \
    tar -zxvf spark-${SPARK_VERSION}-bin-hadoop3.2.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop3.2 spark-${SPARK_VERSION} && \
    rm spark-${SPARK_VERSION}-bin-hadoop3.2.tgz && \
    cp spark-${SPARK_VERSION}/conf/log4j.properties.template spark-${SPARK_VERSION}/conf/log4j.properties && \
    echo $'\nlog4j.logger.io.netty=ERROR' >> spark-${SPARK_VERSION}/conf/log4j.properties && \
    wget -O spark-network-common_2.12-${SPARK_VERSION}.jar https://master.dl.sourceforge.net/project/analytics-zoo/analytics-zoo-data/spark-network-common_2.12-${SPARK_VERSION}.jar && \
    mv spark-network-common_2.12-${SPARK_VERSION}.jar spark-${SPARK_VERSION}/jars && \
    mv spark-${SPARK_VERSION}/python/lib/pyspark.zip spark-${SPARK_VERSION}/python/lib/pyspark.zip.bac && \
    patch spark-${SPARK_VERSION}/python/pyspark/rdd.py /opt/rdd.patch && \
    patch spark-${SPARK_VERSION}/python/pyspark/shuffle.py /opt/shuffle.patch && \
    cd spark-${SPARK_VERSION}/python && \
    zip -r lib/pyspark.zip pyspark
# spark modification
# org.apache.spark.util.Utils to disable chmod fork
RUN cd /opt && \
    git clone https://github.com/apache/spark.git && \
    cd spark && \
    git checkout tags/v3.1.2 -b branch-3.1.2 && \
    git apply /opt/spark-3.1.2.patch && \
    git status && \
    export MAVEN_OPTS="-Xmx2g -XX:ReservedCodeCacheSize=512m \
        -Dhttp.proxyHost=$HTTP_PROXY_HOST \
        -Dhttp.proxyPort=$HTTP_PROXY_PORT \
        -Dhttps.proxyHost=$HTTPS_PROXY_HOST \
        -Dhttps.proxyPort=$HTTPS_PROXY_PORT" && \
    /opt/apache-maven-3.6.3/bin/mvn -T 16 -DskipTests=true clean install -pl core && \
    cd resource-managers/kubernetes/core && \
    /opt/apache-maven-3.6.3/bin/mvn -T 16 -DskipTests=true clean package && \
    ls /opt/spark/core/target/spark-core_2.12-$SPARK_VERSION.jar && \
    ls /opt/spark/resource-managers/kubernetes/core/target/spark-kubernetes_2.12-3.1.2.jar
# hadoop
# org.apache.hadoop.util.Shell.java to disable setsid
RUN cd /opt && \
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
    git clone https://github.com/apache/hadoop.git && \
    cd hadoop && \
    git checkout rel/release-3.2.0 -b branch-3.2.0 && \
    cd hadoop-common-project/hadoop-common && \
    patch src/main/java/org/apache/hadoop/util/Shell.java /opt/hadoop-common-shell.patch && \
    export MAVEN_OPTS="-Xmx2g -XX:ReservedCodeCacheSize=512m \
        -Dhttp.proxyHost=$HTTP_PROXY_HOST \
        -Dhttp.proxyPort=$HTTP_PROXY_PORT \
        -Dhttps.proxyHost=$HTTPS_PROXY_HOST \
        -Dhttps.proxyPort=$HTTPS_PROXY_PORT" && \
    /opt/apache-maven-3.6.3/bin/mvn -T 16 -DskipTests=true clean package && \
    ls /opt/hadoop/hadoop-common-project/hadoop-common/target/hadoop-common-3.2.0.jar

# stage.3 bigdl
FROM ubuntu:18.04 AS bigdl
ARG HTTP_PROXY_HOST
ARG HTTP_PROXY_PORT
ARG HTTPS_PROXY_HOST
ARG HTTPS_PROXY_PORT
ENV JAVA_HOME           /opt/jdk8
ENV PATH                ${JAVA_HOME}/bin:${PATH}

ADD bigdl.lenet.training.patch /bigdl.lenet.training.patch
ADD bigdl.serde.patch /bigdl.serde.patch
COPY --from=spark /opt/jdk  /opt/jdk8

RUN apt-get update --fix-missing && \
    apt-get install -y apt-utils maven git

#bigdl
RUN git clone https://github.com/intel-analytics/BigDL.git && \
    cd BigDL && \
    git apply /bigdl.lenet.training.patch && \
    echo $'\nlog4j.logger.io.netty=ERROR' >> ./spark/dl/src/main/resources/log4j.properties && \
    git status && \
    export MAVEN_OPTS="-Xmx2g -XX:ReservedCodeCacheSize=512m \
        -Dhttp.proxyHost=$HTTP_PROXY_HOST \
        -Dhttp.proxyPort=$HTTP_PROXY_PORT \
        -Dhttps.proxyHost=$HTTPS_PROXY_HOST \
        -Dhttps.proxyPort=$HTTPS_PROXY_PORT" && \
    bash make-dist.sh -P spark_2.x

# stage.4 analytics-zoo
FROM ubuntu:18.04 as analytics-zoo
ARG ANALYTICS_ZOO_VERSION=0.12.0-SNAPSHOT
ARG BIGDL_VERSION=0.13.0
ARG SPARK_VERSION=3.1.2
ENV ANALYTICS_ZOO_VERSION	${ANALYTICS_ZOO_VERSION}
ENV SPARK_VERSION		${SPARK_VERSION}
ENV BIGDL_VERSION		${BIGDL_VERSION}
ENV ANALYTICS_ZOO_HOME		/analytics-zoo-${ANALYTICS_ZOO_VERSION}
ADD orca.data.patch		/opt/orca.data.patch
ADD orca.learn.tf.text.patch	/opt/orca.learn.tf.text.patch
ADD orca.learn.tf.transfer.patch /opt/orca.learn.tf.transfer.patch
RUN apt-get update --fix-missing && \
    apt-get install -y apt-utils curl wget unzip git
RUN wget https://raw.githubusercontent.com/intel-analytics/analytics-zoo/master/docker/zoo/download-analytics-zoo.sh && \
    chmod a+x ./download-analytics-zoo.sh
RUN ./download-analytics-zoo.sh && \
    cd /opt && git clone --depth 1 https://github.com/intel-analytics/analytics-zoo.git
RUN patch /opt/analytics-zoo/pyzoo/zoo/examples/orca/data/spark_pandas.py /opt/orca.data.patch && \
    patch /opt/analytics-zoo/pyzoo/zoo/examples/orca/learn/tf/basic_text_classification/basic_text_classification.py /opt/orca.learn.tf.text.patch && \
    patch /opt/analytics-zoo/pyzoo/zoo/examples/orca/learn/tf/transfer_learning/transfer_learning.py /opt/orca.learn.tf.transfer.patch

# stage.5 az ppml
FROM ubuntu:18.04
ARG ANALYTICS_ZOO_VERSION=0.12.0-SNAPSHOT
ARG SPARK_VERSION=3.1.2
ARG TINI_VERSION=v0.18.0
ENV ANALYTICS_ZOO_VERSION		${ANALYTICS_ZOO_VERSION}
ENV SPARK_VERSION			${SPARK_VERSION}
ENV SPARK_HOME				/ppml/trusted-big-data-ml/work/spark-${SPARK_VERSION}
ENV ANALYTICS_ZOO_HOME			/ppml/trusted-big-data-ml/work/analytics-zoo-${ANALYTICS_ZOO_VERSION}
ENV JAVA_HOME				/opt/jdk8
ENV PATH				/graphene/meson_build_output/bin:${JAVA_HOME}/bin:${PATH}
ENV LOCAL_IP				127.0.0.1
ENV SGX_MEM_SIZE			32G
ENV SPARK_MASTER_IP			127.0.0.1
ENV SPARK_MASTER_PORT			7077
ENV SPARK_MASTER_WEBUI_PORT		8080
ENV SPARK_MASTER			spark://$SPARK_MASTER_IP:$SPARK_MASTER_PORT
ENV SPARK_WORKER_PORT			8082
ENV SPARK_WORKER_WEBUI_PORT		8081
ENV SPARK_DRIVER_PORT			10027
ENV SPARK_DRIVER_BLOCK_MANAGER_PORT	10026
ENV SPARK_DRIVER_IP			$LOCAL_IP
ENV SPARK_BLOCK_MANAGER_PORT		10025
ENV TINI_VERSION                        $TINI_VERSION
ENV PYTHONPATH                          ${PYTHONPATH}:/graphene/meson_build_output/lib/python3.6/site-packages
ENV LC_ALL                              C.UTF-8
ENV LANG                                C.UTF-8

RUN mkdir -p /graphene && \
    mkdir -p /graphene/Runtime && \
    mkdir -p /graphene/python && \
    mkdir -p /graphene/Tools && \
    mkdir -p /graphene/Pal/src && \
    mkdir -p /graphene/meson_build_output && \
    mkdir -p /ppml/trusted-big-data-ml/work && \
    mkdir -p /ppml/trusted-big-data-ml/work/lib && \
    mkdir -p /ppml/trusted-big-data-ml/work/keys && \
    mkdir -p /ppml/trusted-big-data-ml/work/password && \
    mkdir -p /ppml/trusted-big-data-ml/work/data && \
    mkdir -p /ppml/trusted-big-data-ml/work/models && \
    mkdir -p /ppml/trusted-big-data-ml/work/apps && \
    mkdir -p /ppml/trusted-big-data-ml/work/examples/bigdl && \
    mkdir -p /root/.keras/datasets && \
    mkdir -p /root/.zinc && \
    mkdir -p /root/.m2 && \
    mkdir -p /graphene/Pal/src/host/Linux-SGX/signer

COPY --from=graphene /graphene/Scripts /graphene/Scripts
COPY --from=graphene /graphene/Runtime/ /graphene/Runtime
COPY --from=graphene /graphene/python /graphene/python
COPY --from=graphene /graphene/Pal /graphene/Pal
COPY --from=graphene /graphene/Pal/src/host/Linux-SGX/generated_offsets.py /graphene/python/
COPY --from=graphene /graphene/Tools/argv_serializer /graphene/Tools
COPY --from=graphene /graphene/meson_build_output /graphene/meson_build_output
COPY --from=spark /opt/jdk  /opt/jdk8
COPY --from=spark /opt/scala-2.11.8  /opt/scala-2.11.8
COPY --from=spark /opt/spark-${SPARK_VERSION} /ppml/trusted-big-data-ml/work/spark-${SPARK_VERSION}
COPY --from=spark /opt/spark/core/target/spark-core_2.12-$SPARK_VERSION.jar /opt/spark-core_2.12-$SPARK_VERSION.jar
COPY --from=spark /opt/spark/resource-managers/kubernetes/core/target/spark-kubernetes_2.12-$SPARK_VERSION.jar /opt/spark-kubernetes_2.12-$SPARK_VERSION.jar
COPY --from=spark /opt/hadoop/hadoop-common-project/hadoop-common/target/hadoop-common-3.2.0.jar /opt/hadoop-common-3.2.0.jar
COPY --from=bigdl /BigDL/dist/lib/bigdl-*-jar-with-dependencies.jar /ppml/trusted-big-data-ml/work/bigdl-jar-with-dependencies.jar
COPY --from=bigdl /BigDL/dist/lib/bigdl-*-python-api.zip /ppml/trusted-big-data-ml/work/bigdl-python-api.zip
COPY --from=bigdl /BigDL/pyspark/bigdl/examples /ppml/trusted-big-data-ml/work/examples/bigdl
COPY --from=analytics-zoo /analytics-zoo-${ANALYTICS_ZOO_VERSION} /ppml/trusted-big-data-ml/work/analytics-zoo-${ANALYTICS_ZOO_VERSION}
COPY --from=analytics-zoo /opt/analytics-zoo/pyzoo/zoo/examples /ppml/trusted-big-data-ml/work/examples/pyzoo

RUN apt-get update --fix-missing && \
    apt-get install -y apt-utils vim curl nano wget unzip maven git tree zip && \
    apt-get install -y libsm6 make build-essential && \
    apt-get install -y autoconf gawk bison libcurl4-openssl-dev python3-protobuf libprotobuf-c-dev protobuf-c-compiler && \
    apt-get install -y netcat net-tools

#python
RUN apt-get install -y python3-minimal && \
    apt-get install -y build-essential python3 python3-setuptools python3-dev python3-pip && \
    pip3 install --upgrade pip && \
    pip install --upgrade setuptools && \
    pip install --no-cache-dir numpy scipy && \
    pip install --no-cache-dir --no-binary pandas -I pandas && \
    pip install --no-cache-dir scikit-learn matplotlib seaborn jupyter wordcloud moviepy requests h5py opencv-python tensorflow==1.15.0 && \
    pip install --no-cache-dir torch==1.8.1 torchvision==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    #Fix tornado await process
    pip uninstall -y -q tornado && \
    pip install --no-cache-dir tornado && \
    pip install --no-cache-dir filelock && \
    pip install --no-cache-dir tensorflow_datasets==3.2.0 h5py==2.10.0 && \
    pip install --no-cache-dir pyarrow && \
    pip install --no-cache-dir ninja meson && \
    pip install --no-cache-dir psutil && \
    python3 -m pip install toml==0.10.2 click jinja2 && \
    python3 -m ipykernel.kernelspec

ADD ./bash.manifest.template /ppml/trusted-big-data-ml/bash.manifest.template
ADD ./Makefile /ppml/trusted-big-data-ml/Makefile
ADD ./init.sh /ppml/trusted-big-data-ml/init.sh
ADD ./clean.sh /ppml/trusted-big-data-ml/clean.sh
ADD ./examples /ppml/trusted-big-data-ml/work/examples
ADD ./start-spark-local-train-sgx.sh /ppml/trusted-big-data-ml/start-spark-local-train-sgx.sh
ADD ./start-spark-standalone-master-sgx.sh /ppml/trusted-big-data-ml/start-spark-standalone-master-sgx.sh
ADD ./start-spark-standalone-worker-sgx.sh /ppml/trusted-big-data-ml/start-spark-standalone-worker-sgx.sh
ADD ./start-spark-standalone-driver-sgx.sh /ppml/trusted-big-data-ml/start-spark-standalone-driver-sgx.sh
ADD ./ppml-spark-submit.sh /ppml/trusted-big-data-ml/ppml-spark-submit.sh
ADD ./check-status.sh /ppml/trusted-big-data-ml/check-status.sh
ADD ./tracker.py ./tracker.py
ADD ./start-scripts /ppml/trusted-big-data-ml/work/start-scripts
ADD ./_dill.py.patch ./_dill.py.patch
ADD ./python-uuid.patch ./python-uuid.patch

RUN zip -u /ppml/trusted-big-data-ml/work/analytics-zoo-0.12.0-SNAPSHOT/lib/analytics-zoo-bigdl_0.13.0-spark_$SPARK_VERSION-0.12.0-SNAPSHOT-jar-with-dependencies.jar ./tracker.py && \
    unzip /ppml/trusted-big-data-ml/work/analytics-zoo-0.12.0-SNAPSHOT/lib/analytics-zoo-bigdl_0.13.0-spark_$SPARK_VERSION-0.12.0-SNAPSHOT-jar-with-dependencies.jar log4j.properties && \
    echo $'\nlog4j.logger.io.netty=ERROR' >> log4j.properties && \
    zip -u /ppml/trusted-big-data-ml/work/analytics-zoo-0.12.0-SNAPSHOT/lib/analytics-zoo-bigdl_0.13.0-spark_$SPARK_VERSION-0.12.0-SNAPSHOT-jar-with-dependencies.jar log4j.properties && \
    patch /usr/local/lib/python3.6/dist-packages/dill/_dill.py ./_dill.py.patch && \
    patch /usr/lib/python3.6/uuid.py ./python-uuid.patch && \
    #rm $SPARK_HOME/jars/flatbuffers*.jar && \
    #wget -P $SPARK_HOME/jars https://repo1.maven.org/maven2/com/google/flatbuffers/flatbuffers-java/1.9.0/flatbuffers-java-1.9.0.jar && \
    #rm $SPARK_HOME/jars/arrow*.jar &&\
    #wget -P $SPARK_HOME/jars https://repo1.maven.org/maven2/org/apache/arrow/arrow-format/0.11.0/arrow-format-0.11.0.jar && \
    #wget -P $SPARK_HOME/jars https://repo1.maven.org/maven2/org/apache/arrow/arrow-memory/0.11.0/arrow-memory-0.11.0.jar && \
    #wget -P $SPARK_HOME/jars https://repo1.maven.org/maven2/org/apache/arrow/arrow-vector/0.11.0/arrow-vector-0.11.0.jar && \
    rm $SPARK_HOME/jars/spark-core_2.12-$SPARK_VERSION.jar && \
    rm $SPARK_HOME/jars/spark-kubernetes_2.12-$SPARK_VERSION.jar && \
    rm $SPARK_HOME/jars/hadoop-common-3.2.0.jar && \
    mv /opt/spark-core_2.12-$SPARK_VERSION.jar $SPARK_HOME/jars/spark-core_2.11-$SPARK_VERSION.jar && \
    mv /opt/spark-kubernetes_2.12-$SPARK_VERSION.jar $SPARK_HOME/jars/spark-kubernetes_2.12-$SPARK_VERSION.jar && \
    mv /opt/hadoop-common-3.2.0.jar $SPARK_HOME/jars/hadoop-common-3.2.0.jar && \
    wget -P /ppml/trusted-big-data-ml/work/lib https://sourceforge.net/projects/analytics-zoo/files/analytics-zoo-data/libhadoop.so && \
    cp /usr/lib/x86_64-linux-gnu/libpython3.6m.so /usr/lib/libpython3.6m.so && \
    chmod a+x /ppml/trusted-big-data-ml/init.sh && \
    chmod a+x /ppml/trusted-big-data-ml/clean.sh && \
    chmod a+x /ppml/trusted-big-data-ml/start-spark-local-train-sgx.sh && \
    chmod a+x /ppml/trusted-big-data-ml/start-spark-standalone-master-sgx.sh && \
    chmod a+x /ppml/trusted-big-data-ml/start-spark-standalone-worker-sgx.sh && \
    chmod a+x /ppml/trusted-big-data-ml/start-spark-standalone-driver-sgx.sh && \
    chmod a+x /ppml/trusted-big-data-ml/ppml-spark-submit.sh && \
    chmod a+x /ppml/trusted-big-data-ml/check-status.sh && \
    chmod a+x /ppml/trusted-big-data-ml/work/start-scripts/start-python-helloworld-sgx.sh && \
    chmod a+x /ppml/trusted-big-data-ml/work/start-scripts/start-python-numpy-sgx.sh && \
    chmod a+x /ppml/trusted-big-data-ml/work/start-scripts/start-spark-local-bigdl-sgx.sh && \
    chmod a+x /ppml/trusted-big-data-ml/work/start-scripts/start-spark-local-pi-sgx.sh && \
    chmod a+x /ppml/trusted-big-data-ml/work/start-scripts/start-spark-local-sql-sgx.sh && \
    chmod a+x /ppml/trusted-big-data-ml/work/start-scripts/start-spark-local-wordcount-sgx.sh && \
    chmod a+x /ppml/trusted-big-data-ml/work/start-scripts/start-spark-local-xgboost-regressor-sgx.sh && \
    chmod a+x /ppml/trusted-big-data-ml/work/start-scripts/start-spark-local-xgboost-classifier-sgx.sh && \
    chmod a+x /ppml/trusted-big-data-ml/work/start-scripts/start-spark-local-orca-data-sgx.sh && \
    chmod a+x /ppml/trusted-big-data-ml/work/start-scripts/start-spark-local-orca-tf-text.sh 

# kuberenetes support
ADD ./spark-executor-template.yaml /ppml/trusted-big-data-ml/spark-executor-template.yaml
ADD ./entrypoint.sh /opt/entrypoint.sh
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /sbin/tini
RUN rm $SPARK_HOME/jars/okhttp-*.jar && \
    wget -P $SPARK_HOME/jars https://repo1.maven.org/maven2/com/squareup/okhttp3/okhttp/3.8.0/okhttp-3.8.0.jar && \
    chmod +x /opt/entrypoint.sh && \
    chmod +x /sbin/tini && \
    cp /sbin/tini /usr/bin/tini

WORKDIR /ppml/trusted-big-data-ml

ENTRYPOINT [ "/opt/entrypoint.sh" ]
