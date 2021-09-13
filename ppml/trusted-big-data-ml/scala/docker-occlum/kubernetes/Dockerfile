#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
FROM krallin/ubuntu-tini AS tini
FROM occlum/occlum:0.23.7-ubuntu18.04 AS occlum

# Before building the docker image, first build and make a Spark distribution following
# the instructions in http://spark.apache.org/docs/latest/building-spark.html.
# If this docker file is being used in the context of building your images from a Spark
# distribution, the docker build command should be invoked from the top level directory
# of the Spark distribution. E.g.:
# docker build -t spark:latest -f kubernetes/dockerfiles/spark/Dockerfile .

RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
        openjdk-8-jdk \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* 
RUN echo "auth required pam_wheel.so use_uid" >> /etc/pam.d/su && \
    chgrp root /etc/passwd && chmod ug+rw /etc/passwd 


WORKDIR /root
RUN git clone https://github.com/occlum/occlum.git && \
    cd occlum && \
    source /opt/intel/sgxsdk/environment && \
    git checkout -b 0.23.7 0.23.7 && \
    make submodule && \
    OCCLUM_RELEASE_BUILD=1 make && \
    make install && \
    cp -r demos /root/demos && \
    rm -rf /root/occlum

COPY --from=tini /usr/local/bin/tini /sbin/tini

RUN wget https://sourceforge.net/projects/analytics-zoo/files/analytics-zoo-data/libhadoop.so -P /lib/
RUN wget https://archive.apache.org/dist/spark/spark-3.0.0/spark-3.0.0-bin-hadoop2.7.tgz -P /opt/
RUN wget https://sourceforge.net/projects/analytics-zoo/files/analytics-zoo-data/spark-network-common_2.12-3.0.0.jar/download -P /opt/
RUN cd /opt && tar -xvzf /opt/spark-3.0.0-bin-hadoop2.7.tgz && mv /opt/spark-3.0.0-bin-hadoop2.7 /opt/spark
RUN rm /opt/spark-3.0.0-bin-hadoop2.7.tgz
RUN cp /opt/download /opt/spark/jars/spark-network-common_2.12-3.0.0.jar

#COPY ./entrypoint.sh /opt/
RUN cp -r /opt/spark/examples /bin/examples
RUN cp -r /opt/spark/kubernetes/tests /opt/spark/tests
COPY build_spark_instance.sh /opt/
ENV SPARK_HOME /opt/spark

WORKDIR /opt/
#RUN ./build_spark_instance.sh
#COPY ./run_deadloop.sh /opt/occlum_spark

#ENTRYPOINT [ "/opt/entrypoint.sh" ]
