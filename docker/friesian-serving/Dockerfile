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

ARG SPARK_VERSION=2.4.6
ARG BIGDL_VERSION=2.1.0-SNAPSHOT

WORKDIR /opt/work

ENV LD_LIBRARY_PATH /opt/work/lib
ENV SERVING_JAR_PATH /opt/work/bigdl-friesian-serving.jar

RUN apt-get update --fix-missing && \
    apt-get install -y openjdk-8-jdk libgomp1 vim net-tools openssh-server wget

RUN mkdir logs

ADD download-friesian-serving.sh /opt/work

RUN chmod a+x download-friesian-serving.sh
RUN /opt/work/download-friesian-serving.sh

ADD lib/* /opt/work/lib/
ADD start_service.sh /opt/work/
RUN chmod a+x /opt/work/start_service.sh

ENTRYPOINT ["/opt/work/start_service.sh"]
