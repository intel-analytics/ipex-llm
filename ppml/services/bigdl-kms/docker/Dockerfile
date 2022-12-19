ARG BIGDL_VERSION=2.1.0-SNAPSHOT
ARG SPARK_VERSION=3.1.2

# jdk & scala
FROM ubuntu:20.04 as scala
ARG SCALA_VERSION=2.12
ARG SCALA_SUB_VERSION_NUM=10
ARG JDK_VERSION=8u192
ARG JDK_URL=your_jdk_url
ENV SCALA_VERSION ${SCALA_VERSION}
ENV SCALA_SUB_VERSION_NUM ${SCALA_SUB_VERSION_NUM}

RUN apt-get update --fix-missing && \
    env DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y tzdata apt-utils wget unzip
# java
RUN wget $JDK_URL && \
    gunzip jdk-$JDK_VERSION-linux-x64.tar.gz && \
    tar -xf jdk-$JDK_VERSION-linux-x64.tar -C /opt && \
    rm jdk-$JDK_VERSION-linux-x64.tar && \
    mv /opt/jdk* /opt/jdk$JDK_VERSION && \
    ln -s /opt/jdk$JDK_VERSION /opt/jdk
# scala
RUN cd / && wget -c https://downloads.lightbend.com/scala/${SCALA_VERSION}.${SCALA_SUB_VERSION_NUM}/scala-${SCALA_VERSION}.${SCALA_SUB_VERSION_NUM}.tgz && \
    (cd / && gunzip < scala-${SCALA_VERSION}.${SCALA_SUB_VERSION_NUM}.tgz)|(cd /opt && tar -xvf -) && \
    rm /scala-${SCALA_VERSION}.${SCALA_SUB_VERSION_NUM}.tgz
RUN ls -al /opt

# bigdl
FROM ubuntu:20.04 as bigdl
ARG BIGDL_VERSION
ARG SPARK_VERSION
ENV SPARK_VERSION		${SPARK_VERSION}
ENV BIGDL_VERSION		${BIGDL_VERSION}
ENV BIGDL_HOME		        /bigdl-${BIGDL_VERSION}
ADD ./download-bigdl-serving.sh /opt/download-bigdl-serving.sh
ADD ./download-bigdl-keywhiz.sh /opt/download-bigdl-keywhiz.sh
RUN apt-get update --fix-missing && \
    apt-get install -y apt-utils curl wget unzip git
RUN wget https://raw.githubusercontent.com/intel-analytics/analytics-zoo/bigdl-2.0/docker/hyperzoo/download-bigdl.sh && \
    chmod a+x ./download-bigdl.sh && \
    chmod a+x /opt/download-bigdl-serving.sh
RUN ./download-bigdl.sh && \
    rm bigdl*.zip && \
    cd /opt && \
    ./download-bigdl-serving.sh && \
    ./download-bigdl-keywhiz.sh

# ppml
FROM ubuntu:20.04
ARG BIGDL_VERSION
ARG SPARK_VERSION
ARG SCALA_VERSION=2.12
ARG SCALA_SUB_VERSION_NUM=10
ENV SPARK_VERSION			${SPARK_VERSION}
ENV BIGDL_VERSION			${BIGDL_VERSION}
ENV JAVA_HOME				/opt/jdk8
ENV CORE_NUM                            2
ENV XMX_SIZE                            10g 
ENV SCALA_VERSION                       ${SCALA_VERSION}
ENV SCALA_SUB_VERSION_NUM               ${SCALA_SUB_VERSION_NUM}

COPY --from=scala /opt/jdk  /opt/jdk8
COPY --from=scala /opt/scala-${SCALA_VERSION}.${SCALA_SUB_VERSION_NUM}  /opt/scala-${SCALA_VERSION}.${SCALA_SUB_VERSION_NUM}
COPY --from=bigdl /opt/bigdl-serving*-jar-with-dependencies.jar /opt/bigdl-serving-jar-with-dependencies.jar
COPY --from=bigdl /opt/bigdl-serving*-http.jar /opt/bigdl-serving-http.jar

RUN apt-get update --fix-missing && \
    env DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y tzdata && \
    apt-get install -y apt-utils vim curl nano wget unzip git tree && \
    apt-get install -y libsm6 make build-essential && \
    apt-get install -y autoconf gawk bison libcurl4-openssl-dev python3-protobuf libprotobuf-c-dev protobuf-c-compiler && \
    apt-get install -y openssl libssl-dev pkg-config && \
    apt-get install -y netcat net-tools && \
    apt-get install -y python3-minimal && \
    apt-get install -y build-essential python3 python3-setuptools python3-dev python3-pip && \
    pip3 install --upgrade pip && \
    pip install --upgrade setuptools && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip install --no-cache-dir ninja meson && \
    python3 -m pip install toml==0.10.2 click jinja2


ADD ./start-https-frontend.sh /opt/start-http-frontend.sh

RUN chmod a+x /opt/start-https-frontend.sh

ENV LC_ALL	C.UTF-8
ENV LANG	C.UTF-8
