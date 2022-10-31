ARG BASE_IMAGE_NAME
ARG BASE_IMAGE_TAG

ARG BIGDL_VERSION=2.2.0-SNAPSHOT
ARG TINI_VERSION=v0.18.0

#Stage.1 Flask & Numpy & Pandas
FROM $BASE_IMAGE_NAME:$BASE_IMAGE_TAG
RUN pip3 install --upgrade pip && \
    pip install --no-cache-dir flask && \
    pip install --no-cache-dir numpy && \
    pip install --no-cache-dir pandas && \
    mkdir /ppml/examples && \
    mkdir -p /ppml/work/start-scripts

ADD ./examples/ /ppml/examples
ADD ./start-scripts/ /ppml/work/start-scripts

ADD ./entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

#TINI
ARG TINI_VERSION
ENV TINI_VERSION			$TINI_VERSION
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /sbin/tini

RUN chmod +x /sbin/tini && \
    cp /sbin/tini /usr/bin/tini && \
    gramine-argv-serializer bash -c 'export TF_MKL_ALLOC_MAX_BYTES=10737418240 && export _SPARK_AUTH_SECRET=$_SPARK_AUTH_SECRET && $sgx_command' > /ppml/secured_argvs && \

WORKDIR /ppml

ENTRYPOINT [ "/opt/entrypoint.sh" ]

