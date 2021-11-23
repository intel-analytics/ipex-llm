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

FROM ubuntu:18.04

MAINTAINER The BigDL Authors https://github.com/intel-analytics/BigDL

WORKDIR /opt/work

ARG BIGDL_VERSION=0.13.1-SNAPSHOT
ARG SPARK_VERSION=2.1.1
ARG JDK_VERSION=8u192
ARG JDK_URL=your_jdk_url
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
    pip install scikit-learn matplotlib seaborn jupyter wordcloud moviepy requests h5py opencv-python tensorflow==1.15.0 && \
    pip install torch==1.1.0 torchvision==0.3.0 -f https://download.pytorch.org/whl/torch_stable.html && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    #Fix tornado await process
    pip uninstall -y -q tornado && \
    pip install tornado==5.1.1 && \
    python3 -m ipykernel.kernelspec
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
