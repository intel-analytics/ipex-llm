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

FROM ubuntu:14.04

MAINTAINER The Analytics-Zoo Authors https://github.com/intel-analytics/analytics-zoo

WORKDIR /opt/work

ARG ANALYTICS_ZOO_VERSION=0.2.0-SNAPSHOT
ARG SPARK_VERSION=2.2.0
ENV ANALYTICS_ZOO_VERSION_ENV   ${ANALYTICS_ZOO_VERSION}
ENV SPARK_VERSION_ENV           ${SPARK_VERSION}
ENV SPARK_HOME                  /opt/work/spark-${SPARK_VERSION}
ENV ANALYTICS_ZOO_HOME          /opt/work/analytics-zoo-${ANALYTICS_ZOO_VERSION}
ENV JAVA_HOME                   /opt/jdk
ENV PATH                        ${JAVA_HOME}/bin:${PATH}

RUN apt-get update && \
    apt-get install -y vim curl nano wget unzip maven git
#java
RUN wget http://ftp.osuosl.org/pub/funtoo/distfiles/oracle-java/jdk-8u152-linux-x64.tar.gz && \
    gunzip jdk-8u152-linux-x64.tar.gz && \
    tar -xf jdk-8u152-linux-x64.tar -C /opt && \
    rm jdk-8u152-linux-x64.tar && \
    ln -s /opt/jdk1.8.0_152 /opt/jdk
#python
RUN apt-get install -y software-properties-common python-software-properties python-pkg-resources && \
    add-apt-repository -y ppa:jonathonf/python-2.7 && \
    apt-get update && \
    apt-get install -y build-essential python python-setuptools python-dev && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python2 get-pip.py && \
    pip2 install --upgrade setuptools && \
    pip2 install numpy scipy pandas scikit-learn matplotlib seaborn jupyter wordcloud moviepy requests && \
    python2 -m pip install ipykernel && \
    python2 -m ipykernel install --user
#spark
RUN wget https://d3kbcqa49mib13.cloudfront.net/spark-${SPARK_VERSION}-bin-hadoop2.7.tgz && \
    tar -zxvf spark-${SPARK_VERSION}-bin-hadoop2.7.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop2.7 spark-${SPARK_VERSION} && \
    rm spark-${SPARK_VERSION}-bin-hadoop2.7.tgz
#analytics-zoo
RUN git config --global http.proxy $http_proxy && \ 
    git config --global https.proxy $https_proxy &&\
    git clone https://github.com/intel-analytics/analytics-zoo.git

ADD ./download-analytics-zoo.sh /opt/work
ADD ./start-notebook.sh /opt/work
RUN chmod a+x download-analytics-zoo.sh && \
    chmod a+x start-notebook.sh && \
    ./download-analytics-zoo.sh

CMD ["/opt/work/start-notebook.sh"]
