ARG BASE_IMAGE_NAME=intelanalytics/bigdl-ppml-trusted-bigdata-gramine-base
ARG BASE_IMAGE_TAG=2.5.0-SNAPSHOT

FROM $BASE_IMAGE_NAME:$BASE_IMAGE_TAG

ADD ./bigdl-as-entrypoint.sh /ppml/bigdl-as-entrypoint.sh
RUN mkdir -p /opt/bigdl-as/data && \
    chmod 777 /ppml/bigdl-as-entrypoint.sh

RUN apt-get update --fix-missing && \
    apt-get install -y libtdx-attest libtdx-attest-dev libsgx-dcap-quote-verify libsgx-dcap-quote-verify-dev libsgx-dcap-default-qpl

ENTRYPOINT [ "/ppml/bigdl-as-entrypoint.sh" ]
