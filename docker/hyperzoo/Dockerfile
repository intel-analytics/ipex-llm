FROM ubuntu:18.04

MAINTAINER The Analytics-Zoo Authors https://github.com/intel-analytics/analytics-zoo

ARG ANALYTICS_ZOO_VERSION=0.9.0-SNAPSHOT
ARG BIGDL_VERSION=0.10.0
ARG SPARK_VERSION=2.4.3

ENV ANALYTICS_ZOO_VERSION               ${ANALYTICS_ZOO_VERSION}
ENV SPARK_VERSION                       ${SPARK_VERSION}
ENV BIGDL_VERSION                       ${BIGDL_VERSION}

ENV OMP_NUM_THREADS                     4
ENV NOTEBOOK_PORT                       12345
ENV NOTEBOOK_TOKEN                      1234qwer

ENV RUNTIME_SPARK_MASTER                local[4]
ENV RUNTIME_K8S_SERVICE_ACCOUNT         spark
ENV RUNTIME_K8S_SPARK_IMAGE             intelanalytics/hyper-zoo:0.8.0-SNAPSHOT-2.4.3
ENV RUNTIME_DRIVER_HOST                 localhost
ENV RUNTIME_DRIVER_PORT                 54321
ENV RUNTIME_EXECUTOR_CORES              4
ENV RUNTIME_EXECUTOR_MEMORY             20g
ENV RUNTIME_EXECUTOR_INSTANCES          1
ENV RUNTIME_TOTAL_EXECUTOR_CORES        4
ENV RUNTIME_DRIVER_CORES                4
ENV RUNTIME_DRIVER_MEMORY               10g
ENV RUNTIME_PERSISTENT_VOLUME_CLAIM     myvolumeclaim

ENV SPARK_HOME                          /opt/spark
ENV HADOOP_CONF_DIR                     /opt/hadoop-conf
ENV ANALYTICS_ZOO_HOME                  /opt/analytics-zoo-${ANALYTICS_ZOO_VERSION}
ENV JAVA_HOME                           /opt/jdk
ENV REDIS_HOME                          /opt/redis-5.0.5
ENV PYTHONPATH                          ${ANALYTICS_ZOO_HOME}/lib/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-python-api.zip:/opt/models/research/slim
ENV PATH                                ${ANALYTICS_ZOO_HOME}/bin/cluster-serving:${JAVA_HOME}/bin:${PATH}

RUN apt-get update --fix-missing && \
    apt-get install -y apt-utils vim curl nano wget unzip maven git && \
    apt-get install -y gcc g++ make && \
    apt-get install -y libsm6 libxext6 libxrender-dev && \
    rm /bin/sh && \
    ln -sv /bin/bash /bin/sh && \
    echo "auth required pam_wheel.so use_uid" >> /etc/pam.d/su && \
    chgrp root /etc/passwd && chmod ug+rw /etc/passwd

#java
RUN wget http://enos.itcollege.ee/~jpoial/allalaadimised/jdk8/jdk-8u241-linux-x64.tar.gz && \
    gunzip jdk-8u241-linux-x64.tar.gz && \
    tar -xf jdk-8u241-linux-x64.tar -C /opt && \
    rm jdk-8u241-linux-x64.tar && \
    ln -s /opt/jdk1.8.0_241 /opt/jdk

#spark
RUN wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop2.7.tgz && \
    tar -zxvf spark-${SPARK_VERSION}-bin-hadoop2.7.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop2.7 /opt/spark && \
    rm spark-${SPARK_VERSION}-bin-hadoop2.7.tgz && \
    cp /opt/spark/kubernetes/dockerfiles/spark/entrypoint.sh /opt
ENV TINI_VERSION v0.18.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /sbin/tini
RUN chmod +x /sbin/tini

#python
RUN apt-get install -y python3-minimal && \
    apt-get install -y build-essential python3 python3-setuptools python3-dev python3-pip && \
    pip3 install --upgrade pip && \
    pip install --upgrade setuptools && \
    pip install numpy scipy && \
    pip install --no-binary pandas==0.23.4 -I pandas && \
    pip install scikit-learn matplotlib seaborn jupyter wordcloud moviepy requests h5py opencv-python tensorflow==1.15.0 && \
    pip install torch==1.1.0 torchvision==0.3.0 -f https://download.pytorch.org/whl/torch_stable.html && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    #Fix tornado await process
    pip uninstall -y -q tornado && \
    pip install tornado==5.1.1 && \
    python3 -m ipykernel.kernelspec && \
    pip install tensorboard && \
    pip install opencv-python && \
    pip install pyyaml && \
    pip install redis && \
    pip install keras && \
    pip install mxnet-mkl==1.6.0b20191025 && \
    pip install ray==0.7.2 && \
    pip install Pillow==6.1 && \
    pip install psutil aiohttp && \
    pip install gym gym[atari] && \
    pip install lz4 && \
    pip install ray[rllib] && \
    git clone https://github.com/tensorflow/models/ /opt/models

#redis
RUN wget http://download.redis.io/releases/redis-5.0.5.tar.gz && \
    tar xzf redis-5.0.5.tar.gz -C /opt && \
    rm redis-5.0.5.tar.gz && \
    cd /opt/redis-5.0.5 && \
    make && \
    echo "bind 0.0.0.0" >> /opt/redis-5.0.5/redis.conf && \
    cd ..

#zoo
ADD ./download-analytics-zoo.sh /opt
ADD ./start-notebook-spark.sh /opt
ADD ./start-notebook-k8s.sh /opt
RUN chmod a+x /opt/download-analytics-zoo.sh && \
    chmod a+x /opt/start-notebook-spark.sh && \
    chmod a+x /opt/start-notebook-k8s.sh && \
    mkdir -p /opt/analytics-zoo-examples/scala && \
    mkdir -p /opt/analytics-zoo-examples/python
RUN /opt/download-analytics-zoo.sh && \
    rm analytics-zoo-bigdl*.zip && \
    unzip $ANALYTICS_ZOO_HOME/lib/*.zip 'zoo/examples/*' -d /opt/analytics-zoo-examples/python && \
    mv /opt/analytics-zoo-examples/python/zoo/examples/* /opt/analytics-zoo-examples/python && \
    rm -rf /opt/analytics-zoo-examples/python/zoo/examples

#cluster-serving
RUN mkdir -p /opt/cluster-serving && \
    wget https://oss.sonatype.org/content/repositories/public/com/redislabs/spark-redis/2.4.0/spark-redis-2.4.0-jar-with-dependencies.jar && \
    mv spark-redis-2.4.0-jar-with-dependencies.jar /opt/cluster-serving && \
    mkdir -p /opt/cluster-serving/model && \
    mkdir -p /opt/cluster-serving/test_image && \
    unzip -p $ANALYTICS_ZOO_HOME/lib/*.zip zoo/serving/quick_start.py > /opt/cluster-serving/quick_start.py
ADD https://raw.githubusercontent.com/intel-analytics/analytics-zoo/master/pyzoo/test/zoo/resources/serving_quick_start/cat1.jpeg /opt/cluster-serving/test_image 
ADD https://raw.githubusercontent.com/intel-analytics/analytics-zoo/master/pyzoo/test/zoo/resources/serving_quick_start/dog1.jpeg /opt/cluster-serving/test_image
ADD https://raw.githubusercontent.com/intel-analytics/analytics-zoo/master/pyzoo/test/zoo/resources/serving_quick_start/fish1.jpeg /opt/cluster-serving/test_image 
ADD https://analytics-zoo-models.s3-ap-southeast-1.amazonaws.com/tf-res50/frozen_inference_graph.pb /opt/cluster-serving/model/tf_res50/frozen_inference_graph.pb
ADD https://analytics-zoo-models.s3-ap-southeast-1.amazonaws.com/tf-res50/graph_meta.json /opt/cluster-serving/model/tf_res50/graph_meta.json
RUN chmod a+x ${ANALYTICS_ZOO_HOME}/bin/cluster-serving/*
RUN cd /opt/cluster-serving && cluster-serving-init && cd ..

WORKDIR /opt/spark/work-dir

ENTRYPOINT [ "/opt/entrypoint.sh" ]
