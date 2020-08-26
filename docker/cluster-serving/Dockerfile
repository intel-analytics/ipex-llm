FROM ubuntu:18.04

MAINTAINER The Analytics Zoo Authors https://github.com/intel-analytics/analytics-zoo

WORKDIR /opt/work

ARG SPARK_VERSION=2.4.3 
ARG BIGDL_VERSION=0.11.1
ARG ANALYTICS_ZOO_VERSION=0.9.0-SNAPSHOT
ARG FLINK_VERSION=1.10.0

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
    pip3 install --upgrade pip && \
    pip install --upgrade setuptools && \
    pip install opencv-python && \
    pip install pyyaml && \
    pip install redis && \
    pip install pyarrow && \
    pip install httpx

#java
RUN wget http://enos.itcollege.ee/~jpoial/allalaadimised/jdk8/jdk-8u241-linux-x64.tar.gz && \
    gunzip jdk-8u241-linux-x64.tar.gz && \
    tar -xf jdk-8u241-linux-x64.tar -C /opt && \
    rm jdk-8u241-linux-x64.tar && \
    ln -s /opt/jdk1.8.0_241 /opt/jdk

#redis
RUN wget http://download.redis.io/releases/redis-5.0.5.tar.gz && \
    tar xzf redis-5.0.5.tar.gz && \
    rm redis-5.0.5.tar.gz && \
    cd redis-5.0.5 && \
    make && \
    echo bind 0.0.0.0 >> /opt/work/redis-5.0.5/redis.conf && \
    cd ..

#flink
RUN wget https://archive.apache.org/dist/flink/flink-1.10.0/flink-${FLINK_VERSION}-bin-scala_2.11.tgz && \
    tar xzf flink-*.tgz && \
    rm flink-*.tgz


ADD *.sh /opt/work/scripts/
RUN chmod a+x /opt/work/scripts/*.*

RUN /opt/work/scripts/download-cluster-serving-all-zip.sh && \
    rm analytics-*-cluster-serving-all.zip && \
    cd cluster-serving && \
    ./cluster-serving-setup.sh

ADD resources/ /opt/work/cluster-serving/resources/
ADD *.py /opt/work/cluster-serving/
ADD perf/ /opt/work/perf/
ADD freeze_checkpoint.py /opt/work

RUN chmod a+x /opt/work/cluster-serving/*

ENV PYTHONPATH 		${CS_HOME}/serving-python.zip

ENV OMP_NUM_THREADS	all

RUN sed -i "s/jobmanager\.heap\.size:.*/jobmanager\.heap\.size: 8g/g" $FLINK_HOME/conf/flink-conf.yaml && \
    sed -i "s/taskmanager\.memory\.process\.size:.*/taskmanager\.memory\.process\.size: 8g/g" $FLINK_HOME/conf/flink-conf.yaml

CMD /bin/bash -c $FLINK_HOME/bin/start-cluster.sh && tail -f /dev/null


