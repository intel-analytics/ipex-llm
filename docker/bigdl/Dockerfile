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

ARG SPARK_VERSION=2.4.6
ARG SPARK_HOME=/opt/spark
ARG JDK_VERSION=8u192
ARG JDK_URL=your_jdk_url
ARG BIGDL_VERSION=0.14.0-SNAPSHOT

# stage.1 jdk & spark
FROM ubuntu:18.04 as spark
ARG SPARK_VERSION
ARG JDK_VERSION
ARG JDK_URL
ARG SPARK_HOME
ENV SPARK_VERSION                       ${SPARK_VERSION}
ENV SPARK_HOME                          ${SPARK_HOME}
RUN apt-get update --fix-missing && \
    apt-get install -y apt-utils vim curl nano wget unzip maven git && \
# java
    wget $JDK_URL && \
    gunzip jdk-$JDK_VERSION-linux-x64.tar.gz && \
    tar -xf jdk-$JDK_VERSION-linux-x64.tar -C /opt && \
    rm jdk-$JDK_VERSION-linux-x64.tar && \
    mv /opt/jdk* /opt/jdk$JDK_VERSION && \
    ln -s /opt/jdk$JDK_VERSION /opt/jdk && \
# spark
    wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop2.7.tgz && \
    tar -zxvf spark-${SPARK_VERSION}-bin-hadoop2.7.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop2.7 /opt/spark && \
    rm spark-${SPARK_VERSION}-bin-hadoop2.7.tgz && \
    cp /opt/spark/kubernetes/dockerfiles/spark/entrypoint.sh /opt

RUN ln -fs /bin/bash /bin/sh
RUN if [ $SPARK_VERSION = "3.1.2" ]; then \
        rm $SPARK_HOME/jars/okhttp-*.jar && \
        wget -P $SPARK_HOME/jars https://repo1.maven.org/maven2/com/squareup/okhttp3/okhttp/3.8.0/okhttp-3.8.0.jar; \
    elif [ $SPARK_VERSION = "2.4.6" ]; then \
        rm $SPARK_HOME/jars/kubernetes-client-*.jar && \
        wget -P $SPARK_HOME/jars https://repo1.maven.org/maven2/io/fabric8/kubernetes-client/4.4.2/kubernetes-client-4.4.2.jar; \
    fi

# stage.2 bigdl
FROM ubuntu:18.04 as bigdl
ARG SPARK_VERSION
ARG BIGDL_VERSION
ENV SPARK_VERSION               ${SPARK_VERSION}
ENV BIGDL_VERSION               ${BIGDL_VERSION}
ENV BIGDL_HOME                  /opt/bigdl-${BIGDL_VERSION}

RUN apt-get update --fix-missing && \
    apt-get install -y apt-utils vim curl nano wget unzip maven git
ADD ./download-bigdl.sh /opt

RUN chmod a+x /opt/download-bigdl.sh && \
    mkdir -p /opt/bigdl-examples/python
RUN /opt/download-bigdl.sh && \
    rm bigdl*.zip

# stage.3 bigdl-tutorials
ARG BIGDL_VERSION
ARG SPARK_VERSION
ARG RUNTIME_SPARK_MASTER=local[4]
ARG RUNTIME_DRIVER_CORES=4
ARG RUNTIME_DRIVER_MEMORY=20g
ARG RUNTIME_EXECUTOR_CORES=4
ARG RUNTIME_EXECUTOR_MEMORY=20g
ARG RUNTIME_TOTAL_EXECUTOR_CORES=4
ENV SPARK_VERSION                   ${SPARK_VERSION}
ENV BIGDL_VERSION                   ${BIGDL_VERSION}
ENV RUNTIME_SPARK_MASTER            ${RUNTIME_SPARK_MASTER}
ENV RUNTIME_DRIVER_CORES            ${RUNTIME_DRIVER_CORES}
ENV RUNTIME_DRIVER_MEMORY           ${RUNTIME_DRIVER_MEMORY}
ENV RUNTIME_EXECUTOR_CORES          ${RUNTIME_EXECUTOR_CORES}
ENV RUNTIME_EXECUTOR_MEMORY         ${RUNTIME_EXECUTOR_MEMORY}
ENV RUNTIME_TOTAL_EXECUTOR_CORES    ${RUNTIME_TOTAL_EXECUTOR_CORES}
ENV SPARK_HOME                      /opt/work/spark-${SPARK_VERSION}
ENV BIGDL_HOME                      /opt/work/bigdl-${BIGDL_VERSION}
ENV JAVA_HOME                       /opt/jdk
ENV PATH                            ${JAVA_HOME}/bin:${PATH}

WORKDIR /opt/work

COPY --from=spark /opt/jdk /opt/jdk
COPY --from=spark /opt/spark /opt/work/spark-${SPARK_VERSION}
COPY --from=bigdl /opt/bigdl-${BIGDL_VERSION} /opt/work/bigdl-${BIGDL_VERSION}

RUN apt-get update --fix-missing && \
    apt-get install -y apt-utils vim curl nano wget unzip maven git && \
    apt-get install -y gcc g++ make && \
    apt-get install -y libsm6 libxext6 libxrender-dev && \
    rm /bin/sh && \
    ln -sv /bin/bash /bin/sh && \
    echo "auth required pam_wheel.so use_uid" >> /etc/pam.d/su && \
    chgrp root /etc/passwd && chmod ug+rw /etc/passwd && \
# python
    apt-get install -y python3-minimal && \
    apt-get install -y build-essential python3 python3-setuptools python3-dev python3-pip && \
    pip3 install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade setuptools && \
    pip install --no-cache-dir numpy==1.18.1 scipy && \
    pip install --no-cache-dir pandas==1.0.3 && \
    pip install --no-cache-dir scikit-learn matplotlib seaborn jupyter jupyterlab requests h5py && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    #Fix tornado await process
    pip uninstall -y -q tornado && \
    pip install --no-cache-dir tornado && \
    python3 -m ipykernel.kernelspec && \
    pip install --no-cache-dir tensorboard && \
    pip install --no-cache-dir jep && \
    pip install --no-cache-dir cloudpickle && \
    pip install --no-cache-dir opencv-python && \
    pip install --no-cache-dir pyyaml && \
    pip install --no-cache-dir redis && \
    pip install --no-cache-dir ray[tune]==1.2.0 && \
    pip install --no-cache-dir Pillow==6.2 && \
    pip install --no-cache-dir psutil aiohttp && \
    pip install --no-cache-dir py4j && \
    pip install --no-cache-dir cmake==3.16.3 && \
    pip install --no-cache-dir torch==1.7.1 torchvision==0.8.2 && \
    pip install --no-cache-dir horovod==0.19.2 && \
    pip install --no-cache-dir xgboost && \
#tf2
    pip install --no-cache-dir pyarrow && \
    pip install opencv-python==4.2.0.34 && \
    pip install aioredis==1.1.0 && \
    pip install tensorflow==2.4.0

ADD ./start-notebook.sh /opt/work
RUN chmod a+x /opt/work/start-notebook.sh

CMD ["/opt/work/start-notebook.sh"]
