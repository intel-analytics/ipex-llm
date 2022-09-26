ARG BIGDL_IMAGE_NAME
ARG BIGDL_IMAGE_VERSION

# stage 1. generate SGX secrets and prepare KMS env
FROM $BIGDL_IMAGE_NAME:$BIGDL_IMAGE_VERSION as temp

ARG LOCAL_IP
ARG SGX_MEM_SIZE
ARG SGX_LOG_LEVEL

ADD ./enclave-key.pem /root/.config/gramine/enclave-key.pem

# 1.1 make SGX and sign in a temp image
RUN cd /ppml/trusted-big-data-ml && \
    echo LOCAL_IP:$LOCAL_IP && \
    echo SGX_MEM_SIZE:$SGX_MEM_SIZE && \
    echo SGX_LOG_LEVEL:$SGX_LOG_LEVEL && \
    make SGX=1 DEBUG=1 THIS_DIR=/ppml/trusted-big-data-ml  SPARK_LOCAL_IP=$LOCAL_IP SPARK_USER=root G_SGX_SIZE=$SGX_MEM_SIZE G_LOG_LEVEL=$SGX_LOG_LEVEL

# stage 2. copy sign etc. secrets from temp into target image and generate AS secrets
FROM $BIGDL_IMAGE_NAME:$BIGDL_IMAGE_VERSION

# 2.1 copy sign etc. secrets from temp into target image
ARG http_proxy
ARG https_proxy

COPY --from=temp /ppml/trusted-big-data-ml/bash.manifest.sgx /ppml/trusted-big-data-ml/bash.manifest.sgx
COPY --from=temp /ppml/trusted-big-data-ml/bash.sig /ppml/trusted-big-data-ml/bash.sig
COPY --from=temp /ppml/trusted-big-data-ml/bash.manifest /ppml/trusted-big-data-ml/bash.manifest

# 2.2 output mr_enclave and mr_signer to customer
RUN cd /ppml/trusted-big-data-ml && \
    echo "[INFO] Use the below hash values of mr_enclave and mr_signer to register enclave:" && \
    gramine-sgx-quote-dump -m bash.sig|grep mr_enclave && \
    gramine-sgx-quote-dump -m bash.sig|grep mr_signer

WORKDIR /ppml/trusted-big-data-ml

ENTRYPOINT [ "/opt/entrypoint.sh" ]
