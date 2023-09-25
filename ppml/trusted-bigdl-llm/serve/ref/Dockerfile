ARG BASE_IMAGE_NAME
ARG BASE_IMAGE_TAG

# stage 1. generate SGX secrets and prepare KMS env
FROM $BASE_IMAGE_NAME:$BASE_IMAGE_TAG as temp

ARG SGX_MEM_SIZE="32G"
ARG SGX_LOG_LEVEL="error"

ADD ./enclave-key.pem /root/.config/gramine/enclave-key.pem

# 1.1 make SGX and sign in a temp image
RUN cd /ppml && \
    echo SGX_MEM_SIZE:$SGX_MEM_SIZE && \
    echo SGX_LOG_LEVEL:$SGX_LOG_LEVEL && \
    sed -i 's#loader.env.PYTHONPATH = "/usr/lib/python3.9:/usr/lib/python3.9/lib-dynload:/usr/local/lib/python3.9/dist-packages:/usr/lib/python3/dist-packages:/ppml/bigdl-ppml/src"#loader.env.PYTHONPATH = "/usr/lib/python3.9:/usr/lib/python3.9/lib-dynload:/usr/local/lib/python3.9/dist-packages:/usr/lib/python3/dist-packages:/ppml/bigdl-ppml/src:/ppml/FastChat"#' /ppml/bash.manifest.template && \
    make SGX=1 DEBUG=1 THIS_DIR=/ppml  SPARK_USER=root G_SGX_SIZE=$SGX_MEM_SIZE G_LOG_LEVEL=$SGX_LOG_LEVEL

# stage 2. copy sign etc. secrets from temp into target image and generate AS secrets
FROM $BASE_IMAGE_NAME:$BASE_IMAGE_TAG

# 2.1 copy sign etc. secrets from temp into target image
COPY --from=temp /ppml/bash.manifest.sgx /ppml/bash.manifest.sgx
COPY --from=temp /ppml/bash.sig /ppml/bash.sig
COPY --from=temp /ppml/bash.manifest /ppml/bash.manifest


# 2.2 output mr_enclave and mr_signer to customer
RUN cd /ppml && \
    sed -i 's#loader.env.PYTHONPATH = "/usr/lib/python3.9:/usr/lib/python3.9/lib-dynload:/usr/local/lib/python3.9/dist-packages:/usr/lib/python3/dist-packages:/ppml/bigdl-ppml/src"#loader.env.PYTHONPATH = "/usr/lib/python3.9:/usr/lib/python3.9/lib-dynload:/usr/local/lib/python3.9/dist-packages:/usr/lib/python3/dist-packages:/ppml/bigdl-ppml/src:/ppml/FastChat"#' /ppml/bash.manifest.template && \
    gramine-sgx-get-token --output bash.token --sig bash.sig

WORKDIR /ppml
ENTRYPOINT [ "/opt/entrypoint.sh" ]

