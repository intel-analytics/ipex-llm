FROM krallin/ubuntu-tini AS tini
FROM occlum/occlum:0.24.0-ubuntu18.04 AS occlum

ENV SPARK_HOME /opt/spark

RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
        openjdk-8-jdk \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* 
RUN echo "auth required pam_wheel.so use_uid" >> /etc/pam.d/su && \
    chgrp root /etc/passwd && chmod ug+rw /etc/passwd 

COPY --from=tini /usr/local/bin/tini /sbin/tini

ARG SPARK_VERSION=3.1.2
ENV SPARK_VERSION	${SPARK_VERSION}

# Download & prepare Spark 3.0.0
WORKDIR /opt
RUN wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop2.7.tgz -P /opt/ && \
    tar -zxvf /opt/spark-${SPARK_VERSION}-bin-hadoop2.7.tgz && \
    mv /opt/spark-${SPARK_VERSION}-bin-hadoop2.7 /opt/spark && \
    rm /opt/spark-${SPARK_VERSION}-bin-hadoop2.7.tgz && \
    cp -r /opt/spark/examples /bin/examples && \
    cp -r /opt/spark/kubernetes/tests /opt/spark/tests

# Remove fork with libhadoop.so and spark-network-common_2.12-3.0.0.jar
RUN wget https://sourceforge.net/projects/analytics-zoo/files/analytics-zoo-data/libhadoop.so -P /lib/ && \
    rm -f /opt/spark/jars/spark-network-common_2.12-${SPARK_VERSION}.jar && \
    wget -O /opt/spark/jars/spark-network-common_2.12-${SPARK_VERSION}.jar https://master.dl.sourceforge.net/project/analytics-zoo/analytics-zoo-data/spark-network-common_2.12-${SPARK_VERSION}.jar

COPY ./entrypoint.sh /opt/
COPY ./init.sh /opt/

RUN chmod a+x /opt/entrypoint.sh && \
    chmod a+x /opt/init.sh

WORKDIR /opt/

ENTRYPOINT [ "/opt/entrypoint.sh" ]
