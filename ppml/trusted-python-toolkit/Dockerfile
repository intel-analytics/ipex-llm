ARG BASE_IMAGE_NAME
ARG BASE_IMAGE_TAG
ARG BIGDL_VERSION=2.2.0-SNAPSHOT
ARG TINI_VERSION=v0.18.0

#Stage.1 Flask & Numpy & Pandas
FROM $BASE_IMAGE_NAME:$BASE_IMAGE_TAG
ARG http_proxy
ARG https_proxy
ARG TINI_VERSION
ENV TINI_VERSION                        $TINI_VERSION

ADD ./examples/ /ppml/examples
ADD ./entrypoint.sh /opt/entrypoint.sh
ADD ./start-scripts/ /ppml/work/start-scripts
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /sbin/tini
RUN pip3 install --upgrade pip && \
    pip install --no-cache-dir flask && \
    pip install --no-cache-dir numpy && \
    pip install --no-cache-dir pandas && \
    mkdir -p /ppml/tests/numpy && \
    mkdir -p /ppml/tests/pandas && \
    chmod +x /opt/entrypoint.sh && \
    chmod +x /sbin/tini && \
    chmod +x /ppml/work/start-scripts/start-python-flask-sgx.sh && \
    chmod +x /ppml/work/start-scripts/start-python-numpy-sgx.sh && \
    chmod +x /ppml/work/start-scripts/start-python-pandas-sgx.sh && \
    cp /sbin/tini /usr/bin/tini && \
    gramine-argv-serializer bash -c 'export TF_MKL_ALLOC_MAX_BYTES=10737418240 && export _SPARK_AUTH_SECRET=$_SPARK_AUTH_SECRET && $sgx_command' > /ppml/secured_argvs

WORKDIR /ppml

ENTRYPOINT [ "/opt/entrypoint.sh" ]

