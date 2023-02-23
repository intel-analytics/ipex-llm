ARG BASE_IMAGE_NAME=intelanalytics/bigdl-ppml-trusted-bigdata-gramine-base
ARG BASE_IMAGE_TAG=2.3.0-SNAPSHOT

FROM $BASE_IMAGE_NAME:$BASE_IMAGE_TAG as attestation-service-base

RUN mkdir -p /opt/bigdl-as/data
ADD ./bigdl-as-entrypoint.sh /ppml/bigdl-as-entrypoint.sh
RUN chmod 777 /ppml/bigdl-as-entrypoint.sh

# stage 1. generate SGX secrets and prepare KMS env
FROM attestation-service-base as temp

ARG SGX_MEM_SIZE=32G
ARG SGX_LOG_LEVEL=error

ADD ./enclave-key.pem /root/.config/gramine/enclave-key.pem

# 1.1 make SGX and sign in a temp image
RUN cd /ppml && \
    echo SGX_MEM_SIZE:$SGX_MEM_SIZE && \
    echo SGX_LOG_LEVEL:$SGX_LOG_LEVEL && \
    make SGX=1 DEBUG=1 THIS_DIR=/ppml  SPARK_USER=root G_SGX_SIZE=$SGX_MEM_SIZE G_LOG_LEVEL=$SGX_LOG_LEVEL

# stage 2. copy sign etc. secrets from temp into target image and generate AS secrets
FROM attestation-service-base

# 2.1 copy sign etc. secrets from temp into target image
COPY --from=temp /ppml/bash.manifest.sgx /ppml/bash.manifest.sgx
COPY --from=temp /ppml/bash.sig /ppml/bash.sig
COPY --from=temp /ppml/bash.manifest /ppml/bash.manifest

# 2.2 output mr_enclave and mr_signer to customer
RUN cd /ppml && \
    echo "[INFO] Use the below hash values of mr_enclave and mr_signer to register enclave:" && \
    gramine-sgx-quote-dump -m bash.sig|grep mr_enclave && \
    gramine-sgx-quote-dump -m bash.sig|grep mr_signer

ENTRYPOINT [ "/ppml/bigdl-as-entrypoint.sh" ]