FROM ubuntu:18.04

MAINTAINER The Analytics Zoo Authors https://github.com/intel-analytics/analytics-zoo

WORKDIR /opt/work

ARG SPARK_VERSION=2.4.6
ARG BIGDL_VERSION=0.13.0
ARG ANALYTICS_ZOO_VERSION=0.12.0-SNAPSHOT
ARG FLINK_VERSION=1.11.2
ARG JDK_VERSION=8u192
ARG JDK_URL=your_jdk_url

ENV FLINK_HOME			/opt/work/flink-${FLINK_VERSION}
ENV JAVA_HOME			/opt/jdk
ENV REDIS_HOME 		/opt/work/redis-5.0.5
ENV ANALYTICS_ZOO_VERSION 	${ANALYTICS_ZOO_VERSION}
ENV CS_HOME  	/opt/work/cluster-serving
ENV PATH			${JAVA_HOME}/bin:${ANALYTICS_ZOO_HOME}/bin/cluster-serving:${PATH}

RUN apt-get update && \
    apt-get install -y apt-utils vim curl nano wget unzip maven git numactl net-tools && \
    apt-get install -y gcc g++ make && \
    apt-get install -y libsm6 libxext6 libxrender-dev libgl1-mesa-dev

#python
RUN apt-get install -y python3-minimal && \
    apt-get install -y build-essential python3 python3-setuptools python3-dev python3-pip && \
    pip3 install --upgrade pip==20.2.4 && \
    pip install --upgrade setuptools && \
    pip install analytics-zoo-serving

#java
RUN wget $JDK_URL && \
    gunzip jdk-$JDK_VERSION-linux-x64.tar.gz && \
    tar -xf jdk-$JDK_VERSION-linux-x64.tar -C /opt && \
    rm jdk-$JDK_VERSION-linux-x64.tar && \
    mv /opt/jdk* /opt/jdk$JDK_VERSION && \
    ln -s /opt/jdk$JDK_VERSION /opt/jdk

#redis
RUN wget http://download.redis.io/releases/redis-5.0.5.tar.gz && \
    tar xzf redis-5.0.5.tar.gz && \
    rm redis-5.0.5.tar.gz && \
    cd redis-5.0.5 && \
    make && \
    echo bind 0.0.0.0 >> /opt/work/redis-5.0.5/redis.conf && \
    cd ..

#flink
RUN wget https://archive.apache.org/dist/flink/flink-1.11.2/flink-${FLINK_VERSION}-bin-scala_2.11.tgz && \
    tar xzf flink-*.tgz && \
    rm flink-*.tgz


ADD resources/ /opt/work/cluster-serving/resources/
ADD *.py /opt/work/cluster-serving/
ADD perf/ /opt/work/perf/
ADD freeze_checkpoint.py /opt/work

RUN cd cluster-serving && \
    wget https://repo1.maven.org/maven2/com/intel/analytics/zoo/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}/${ANALYTICS_ZOO_VERSION}/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-serving.jar && \
    mv *-serving.jar zoo.jar

ENV OMP_NUM_THREADS	all

RUN sed -i "s/jobmanager\.heap\.size:.*/jobmanager\.heap\.size: 8g/g" $FLINK_HOME/conf/flink-conf.yaml && \
    sed -i "s/taskmanager\.memory\.process\.size:.*/taskmanager\.memory\.process\.size: 8g/g" $FLINK_HOME/conf/flink-conf.yaml

CMD /bin/bash -c $FLINK_HOME/bin/start-cluster.sh && tail -f /dev/null


