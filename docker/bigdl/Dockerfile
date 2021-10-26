#
# Copyright 2016 The Analytics-Zoo Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

FROM ubuntu:18.04

MAINTAINER The Analytics-Zoo Authors https://github.com/intel-analytics/analytics-zoo

WORKDIR /opt/work

ARG ANALYTICS_ZOO_VERSION=0.12.0-SNAPSHOT
ARG BIGDL_VERSION=0.13.0
ARG SPARK_VERSION=2.4.6
ARG RUNTIME_SPARK_MASTER=local[4]
ARG RUNTIME_DRIVER_CORES=4
ARG RUNTIME_DRIVER_MEMORY=20g
ARG RUNTIME_EXECUTOR_CORES=4
ARG RUNTIME_EXECUTOR_MEMORY=20g
ARG RUNTIME_TOTAL_EXECUTOR_CORES=4
ARG JDK_VERSION=8u192
ARG JDK_URL=your_jdk_url
ENV ANALYTICS_ZOO_VERSION           ${ANALYTICS_ZOO_VERSION}
ENV SPARK_VERSION                   ${SPARK_VERSION}
ENV BIGDL_VERSION                   ${BIGDL_VERSION}
ENV RUNTIME_SPARK_MASTER            ${RUNTIME_SPARK_MASTER}
ENV RUNTIME_DRIVER_CORES            ${RUNTIME_DRIVER_CORES}
ENV RUNTIME_DRIVER_MEMORY           ${RUNTIME_DRIVER_MEMORY}
ENV RUNTIME_EXECUTOR_CORES          ${RUNTIME_EXECUTOR_CORES}
ENV RUNTIME_EXECUTOR_MEMORY         ${RUNTIME_EXECUTOR_MEMORY}
ENV RUNTIME_TOTAL_EXECUTOR_CORES    ${RUNTIME_TOTAL_EXECUTOR_CORES}
ENV SPARK_HOME                      /opt/work/spark-${SPARK_VERSION}
ENV ANALYTICS_ZOO_HOME              /opt/work/analytics-zoo-${ANALYTICS_ZOO_VERSION}
ENV JAVA_HOME                       /opt/jdk
ENV PATH                            ${JAVA_HOME}/bin:${PATH}

RUN apt-get update --fix-missing && \
    apt-get install -y apt-utils vim curl nano wget unzip maven git && \
    apt-get install -y libsm6

#java
RUN wget $JDK_URL && \
    gunzip jdk-$JDK_VERSION-linux-x64.tar.gz && \
    tar -xf jdk-$JDK_VERSION-linux-x64.tar -C /opt && \
    rm jdk-$JDK_VERSION-linux-x64.tar && \
    mv /opt/jdk* /opt/jdk$JDK_VERSION && \
    ln -s /opt/jdk$JDK_VERSION /opt/jdk

#python
RUN apt-get install -y python3-minimal && \
    apt-get install -y build-essential python3 python3-setuptools python3-dev python3-pip && \
    pip3 install --upgrade pip==20.2.4 && \
    pip install --upgrade setuptools && \
    pip install numpy scipy && \
    pip install --no-binary pandas -I pandas && \
    pip install scikit-learn matplotlib seaborn jupyter notebook==6.1.4 jupyterlab==2.3.1 wordcloud moviepy requests h5py opencv-python tensorflow==1.15.0 && \
    pip install torch==1.1.0 torchvision==0.3.0 -f https://download.pytorch.org/whl/torch_stable.html && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    #Fix tornado await process
    pip uninstall -y -q tornado && \
    pip install --no-cache-dir tornado==5.1.1 && \
    python3 -m ipykernel.kernelspec

#spark
RUN wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop2.7.tgz && \
    tar -zxvf spark-${SPARK_VERSION}-bin-hadoop2.7.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop2.7 spark-${SPARK_VERSION} && \
    rm spark-${SPARK_VERSION}-bin-hadoop2.7.tgz

ADD ./download-analytics-zoo.sh /opt/work
ADD ./start-notebook.sh /opt/work
RUN chmod a+x /opt/work/download-analytics-zoo.sh && \
    chmod a+x /opt/work/start-notebook.sh
RUN /opt/work/download-analytics-zoo.sh

CMD ["/opt/work/start-notebook.sh"]
