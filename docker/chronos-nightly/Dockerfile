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


FROM ubuntu:20.04
WORKDIR /opt/work
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

ADD ./docker/chronos-nightly/install-python-env.sh /opt
ADD ./python/chronos/colab-notebook /opt/work/colab-notebook
RUN chmod a+x /opt/install-python-env.sh

RUN apt-get update --fix-missing && \
    apt-get install -y apt-utils vim curl nano wget unzip git && \
    apt-get install -y gcc g++ make && \
    apt-get install -y libsm6 libxext6 libxrender-dev && \
    apt-get install -y openjdk-8-jre && \
    rm /bin/sh && \
    ln -sv /bin/bash /bin/sh && \
    echo "auth required pam_wheel.so use_uid" >> /etc/pam.d/su && \
    chgrp root /etc/passwd && chmod ug+rw /etc/passwd && \
# Install Miniconda
    wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh && \
    chmod +x Miniconda3-py37_4.12.0-Linux-x86_64.sh && \
    ./Miniconda3-py37_4.12.0-Linux-x86_64.sh -b -f -p /usr/local && \
    rm Miniconda3-py37_4.12.0-Linux-x86_64.sh && \
# python
    /opt/install-python-env.sh

RUN echo "source activate chronos" > ~/.bashrc
RUN echo "export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/" >> ~/.bashrc
ENV PATH /usr/local/envs/chronos/bin:$PATH

