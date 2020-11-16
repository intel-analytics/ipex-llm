#
# Copyright 2016 The BigDL Authors.
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

MAINTAINER The BigDL Authors https://github.com/intel-analytics/BigDL

WORKDIR /opt/work

ARG BIGDL_VERSION=0.13.0-SNAPSHOT
ARG SPARK_VERSION=2.1.1
ENV BIGDL_VERSION_ENV		${BIGDL_VERSION}
ENV SPARK_VERSION_ENV		${SPARK_VERSION}
ENV SPARK_HOME			/opt/work/spark-${SPARK_VERSION}
ENV BIGDL_HOME			/opt/work/bigdl-${BIGDL_VERSION}
ENV BIGDL_TUTORIALS_HOME	/opt/work/BigDL-Tutorials
ENV JAVA_HOME 			/opt/jdk
ENV PATH 			${JAVA_HOME}/bin:${PATH}

RUN apt-get update && \
    apt-get install -y vim curl nano wget unzip maven git
#java
RUN wget https://build.funtoo.org/distfiles/oracle-java/jdk-8u152-linux-x64.tar.gz && \
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
    pip install numpy scipy pandas scikit-learn matplotlib seaborn jupyter wordcloud && \
    python2 -m pip install ipykernel && \
    python2 -m ipykernel install --user
#spark
RUN wget https://d3kbcqa49mib13.cloudfront.net/spark-${SPARK_VERSION}-bin-hadoop2.7.tgz && \
    tar -zxvf spark-${SPARK_VERSION}-bin-hadoop2.7.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop2.7 spark-${SPARK_VERSION} && \
    rm spark-${SPARK_VERSION}-bin-hadoop2.7.tgz
#bigdl
RUN git clone https://github.com/intel-analytics/BigDL-Tutorials.git 

ADD ./start-notebook.sh /opt/work
ADD ./download-bigdl.sh /opt/work
RUN chmod a+x start-notebook.sh && \
    chmod a+x download-bigdl.sh 
RUN ./download-bigdl.sh

CMD ["/opt/work/start-notebook.sh"]
