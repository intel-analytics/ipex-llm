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
SHELL ["/bin/bash", "-c"]

# model: select from pytorch, tensorflow, prophet, arima, ml (default)
ARG model=ml

# auto_tuning: y (for yes) or n (default, for no)
ARG auto_tuning=n

# hardware: single (default) or cluster
ARG hardware=single

# inference: install dependencies for inference optimization (e.g. onnxruntime, openvino, ...)
#            y (for yes) or n (default, for no)
ARG inference=n

# extra_dep: install extra dependencies (e.g. jupyter, matplotlib, ...)
#            y (for yes) or n (default, for no)
ARG extra_dep=n

COPY ./docker/chronos-nightly/install-python-env.sh /opt
COPY ./python/chronos/colab-notebook /opt/work/colab-notebook
COPY ./python/chronos/use-case/AIOps /opt/work/use-case/AIOps
RUN chmod a+x /opt/install-python-env.sh

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc wget && \
    rm -rf /var/lib/apt/lists/*
SHELL ["/bin/bash", "-c"]
RUN echo "auth required pam_wheel.so use_uid" >> /etc/pam.d/su && \
    chgrp root /etc/passwd && chmod ug+rw /etc/passwd && \
    # Install Miniconda
    wget --progress=dot:giga --no-check-certificate https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh && \
    chmod +x Miniconda3-py37_4.12.0-Linux-x86_64.sh && \
    ./Miniconda3-py37_4.12.0-Linux-x86_64.sh -b -f -p /usr/local && \
    rm Miniconda3-py37_4.12.0-Linux-x86_64.sh && \
    # python
    conda create -y -n chronos python=3.7 setuptools=58.0.4 && \
    /opt/install-python-env.sh ${model} ${auto_tuning} ${hardware} ${inference} ${extra_dep}

RUN echo "source activate chronos" > ~/.bashrc
ENV PATH /usr/local/envs/chronos/bin:$PATH
