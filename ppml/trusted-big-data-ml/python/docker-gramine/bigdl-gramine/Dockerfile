ARG BASE_IMAGE_NAME
ARG BASE_IMAGE_TAG

# stage 1. generate SGX secrets and prepare KMS env
FROM $BASE_IMAGE_NAME:$BASE_IMAGE_TAG as temp

ARG SGX_MEM_SIZE
ARG SGX_LOG_LEVEL
ARG ENABLE_DCAP_ATTESTATION

ADD ./enclave-key.pem /root/.config/gramine/enclave-key.pem
ADD ./make-sgx.sh /ppml/trusted-big-data-ml/make-sgx.sh

# 1.1 make SGX and sign in a temp image
RUN cd /ppml/trusted-big-data-ml && \
    chmod a+x make-sgx.sh && \
    ./make-sgx.sh

# stage 2. copy sign etc. secrets from temp into target image and generate AS secrets
FROM $BASE_IMAGE_NAME:$BASE_IMAGE_TAG

# 2.1 copy sign etc. secrets from temp into target image
COPY --from=temp /ppml/trusted-big-data-ml/bash.manifest.sgx /ppml/trusted-big-data-ml/bash.manifest.sgx
COPY --from=temp /ppml/trusted-big-data-ml/bash.sig /ppml/trusted-big-data-ml/bash.sig
COPY --from=temp /ppml/trusted-big-data-ml/bash.manifest /ppml/trusted-big-data-ml/bash.manifest

# 2.2 output mr_enclave and mr_signer to customer
RUN cd /ppml/trusted-big-data-ml && \
    gramine-sgx-get-token --output bash.token --sig bash.sig

WORKDIR /ppml/trusted-big-data-ml

ENTRYPOINT [ "/opt/entrypoint.sh" ]
