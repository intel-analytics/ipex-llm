ARG BASE_IMAGE_NAME
ARG BASE_IMAGE_TAG

# stage 1. generate SGX secrets and prepare KMS env
FROM $BASE_IMAGE_NAME:$BASE_IMAGE_TAG as temp

ARG SGX_MEM_SIZE     16G 
ARG SGX_LOG_LEVEL    error
ARG ENABLE_DCAP_ATTESTATION true

ADD ./enclave-key.pem /root/.config/gramine/enclave-key.pem
ADD ./make-sgx.sh /ppml/make-sgx.sh

# 1.1 make SGX and sign in a temp image
RUN cd /ppml && \
    echo SGX_MEM_SIZE:$SGX_MEM_SIZE && \
    echo SGX_LOG_LEVEL:$SGX_LOG_LEVEL && \
    echo ENABLE_DCAP_ATTESTATION:$ENABLE_DCAP_ATTESTATION && \
    chmod a+x make-sgx.sh && \
    ./make-sgx.sh

# stage 2. copy sign etc. secrets from temp into target image and generate AS secrets
FROM $BASE_IMAGE_NAME:$BASE_IMAGE_TAG

# 2.1 copy sign etc. secrets from temp into target image
COPY --from=temp /ppml/bash.manifest.sgx /ppml/bash.manifest.sgx
COPY --from=temp /ppml/bash.sig /ppml/bash.sig
COPY --from=temp /ppml/bash.manifest /ppml/bash.manifest


# 2.2 output mr_enclave and mr_signer to customer
RUN cd /ppml && \
    gramine-sgx-get-token --output bash.token --sig bash.sig

WORKDIR /ppml

ENTRYPOINT [ "/ppml/torchserve/frontend-entrypoint.sh" ]
