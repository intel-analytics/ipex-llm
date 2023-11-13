ARG BASE_IMAGE_NAME
ARG BASE_IMAGE_TAG
ARG BIGDL_VERSION=2.5.0-SNAPSHOT

FROM $BASE_IMAGE_NAME:$BASE_IMAGE_TAG as temp
ARG http_proxy
ARG https_proxy


ADD ./examples/ /ppml/examples
ADD ./entrypoint.sh /opt/entrypoint.sh
ADD ./scripts/ /ppml/scripts
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir flask && \
    pip3 install --no-cache-dir numpy && \
    pip3 install --no-cache-dir pandas && \
    pip3 install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip3 install --no-cache-dir lightgbm && \
    pip install --no-cache-dir --upgrade tensorflow && \
    pip install --no-cache-dir -U scikit-learn && \
    pip install --no-cache-dir shap && \
    pip install --no-cache-dir -U matplotlib && \
    pip install --upgrade "protobuf<=3.20.1" && \
    chmod +x /opt/entrypoint.sh && \
    chmod +x /ppml/scripts/* && \
    gramine-argv-serializer bash -c 'export TF_MKL_ALLOC_MAX_BYTES=10737418240 && export _SPARK_AUTH_SECRET=$_SPARK_AUTH_SECRET && $sgx_command' > /ppml/secured_argvs

WORKDIR /ppml

ENTRYPOINT [ "/opt/entrypoint.sh" ]

