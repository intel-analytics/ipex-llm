
FROM krallin/ubuntu-tini AS tini
FROM ubuntu:20.04 as bigdl


ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64

ARG HTTP_PROXY_HOST
ARG HTTP_PROXY_PORT
ARG HTTPS_PROXY_HOST
ARG HTTPS_PROXY_PORT
ENV HTTP_PROXY=http://$HTTP_PROXY_HOST:$HTTP_PROXY_PORT
ENV HTTPS_PROXY=http://$HTTPS_PROXY_HOST:$HTTPS_PROXY_PORT

RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
        openjdk-8-jdk build-essential wget git unzip zip swig libssl-dev cmake && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ADD install_llm_with_conda.sh /opt
RUN cd /opt && bash ./install_llm_with_conda.sh

FROM occlum/occlum:0.29.7-ubuntu20.04 as ppml

MAINTAINER The BigDL Authors https://github.com/intel-analytics/BigDL
ARG BIGDL_VERSION=2.5.0-SNAPSHOT
ENV BIGDL_VERSION=${BIGDL_VERSION}
ENV BIGDL_HOME=/opt/bigdl-${BIGDL_VERSION}
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
ENV SGX_MEM_SIZE=20GB

ARG HTTP_PROXY_HOST
ARG HTTP_PROXY_PORT
ARG HTTPS_PROXY_HOST
ARG HTTPS_PROXY_PORT

RUN echo "auth required pam_wheel.so use_uid" >> /etc/pam.d/su && \
    chgrp root /etc/passwd && chmod ug+rw /etc/passwd 

COPY --from=tini /usr/local/bin/tini /sbin/tini

#RUN echo 'deb [arch=amd64] https://download.01.org/intel-sgx/sgx_repo/ubuntu focal main' | sudo tee /etc/apt/sources.list.d/intel-sgx.list && \
#    wget -qO - https://download.01.org/intel-sgx/sgx_repo/ubuntu/intel-sgx-deb.key | apt-key add -

RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
        openjdk-8-jdk && \
    apt-get clean

# Prepare BigDL
#RUN cd /opt && \
#    wget https://raw.githubusercontent.com/intel-analytics/analytics-zoo/bigdl-2.0/docker/hyperzoo/download-bigdl.sh && \
#    chmod a+x ./download-bigdl.sh && \
#    ./download-bigdl.sh && \
#    rm bigdl*.zip

# Copy scripts & other files
ADD ./run_llm_on_occlum_glibc.sh /opt/run_llm_on_occlum_glibc.sh

# For get_quote C compile
RUN rm /root/demos/remote_attestation/dcap/c_app/dcap_c_test.c
ADD ./attestation/dcap_c_test.c /root/demos/remote_attestation/dcap/c_app/
ADD ./attestation/dcap-ppml.yaml /root/demos/remote_attestation/dcap/
ADD ./attestation/get_quote_on_ppml.sh /root/demos/remote_attestation/dcap
#ADD ./attestation/attestation_cli.py /opt/attestation_cli.py
ADD ./attestation/attestation.sh /opt/attestation.sh

# For attestation, add ppml libs
#ADD ./verify-attestation-service.sh /opt
ADD ./attestation/print_enclave_signer.sh /opt
#ADD ./register.sh /opt

COPY ./entrypoint.sh /opt/
#COPY ./ehsm_entry.sh /opt/
COPY ./spark.yaml /opt/

RUN chmod a+x /opt/entrypoint.sh && \
#    chmod a+x /opt/ehsm_entry.sh && \
    chmod a+x /opt/run_llm_on_occlum_glibc.sh && \
#    chmod a+x /opt/verify-attestation-service.sh && \
    chmod a+x /opt/print_enclave_signer.sh && \
#    chmod a+x /opt/register.sh && \
    chmod a+x /root/demos/remote_attestation/dcap/get_quote_on_ppml.sh && \
    chmod a+x /root/demos/remote_attestation/dcap/c_app/dcap_c_test.c && \
    chmod a+x /opt/attestation.sh && \
    sed -i '/aesm/d' /root/.bashrc && \
    rm -rf /var/run/aesmd

# Add python lib
ADD ./python-glibc.yaml /opt/python-glibc.yaml
COPY --from=bigdl /opt/python-occlum /opt/python-occlum
COPY --from=bigdl /opt/occlum_spark/data1/FastChat /opt/occlum_spark/data1/FastChat
ADD ./bench.py /opt/occlum_spark/data1/bench.py
RUN cd /opt && \
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash ./Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda
ENV PATH=$PATH:/opt/miniconda/bin

#patch for utils.py and cpuinfo.py
ADD ./utils.diff /opt/utils.diff
ADD ./cpuinfo.diff /opt/cpuinfo.diff
RUN patch /opt/python-occlum/lib/python3.9/site-packages/pydub/utils.py /opt/utils.diff && \
    patch /opt/python-occlum/lib/python3.9/site-packages/cpuinfo/cpuinfo.py /opt/cpuinfo.diff

RUN cd /root && \
    git clone https://github.com/intel-analytics/BigDL.git && \
    mv /root/BigDL/python/ppml/src/bigdl/ppml/ /opt/python-occlum/lib/python3.9/site-packages/bigdl/ && \
    cp  /opt/python-occlum/lib/python3.9/site-packages/bigdl/ppml/attestation/attestation_cli.py /opt/ && \
    rm -rf /root/BigDL
#ADD ./py-examples /opt/py-examples

# Prepare Bigdl python libs
#RUN unzip -o $BIGDL_HOME/python/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip -d $BIGDL_HOME/python-lib && \
#    unzip -o $BIGDL_HOME/python/bigdl-friesian-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip -d $BIGDL_HOME/python-lib && \
#    unzip -o $BIGDL_HOME/python/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip -d $BIGDL_HOME/python-lib && \
#    unzip -o $BIGDL_HOME/python/bigdl-ppml-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip -d $BIGDL_HOME/python-lib && \
#    unzip -o $BIGDL_HOME/python/bigdl-serving-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip -d $BIGDL_HOME/python-lib && \
#    unzip -o $BIGDL_HOME/python/bigdl-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip -d $BIGDL_HOME/python-lib

#RUN cd $BIGDL_HOME/python-lib/ && \
#    zip -r bigdl.zip ./bigdl && \
#    mv bigdl.zip /opt/py-examples/

ENTRYPOINT [ "/opt/entrypoint.sh" ]
