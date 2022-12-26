ARG http_proxy
ARG https_proxy
ARG BASE_IMAGE_NAME
ARG BASE_IMAGE_TAG
ARG BIGDL_VERSION=2.2.0-SNAPSHOT
ARG TINI_VERSION=v0.18.0
ARG JDK_VERSION=11
ARG TRITON_VERSION=2.27.0
ARG TRITON_CONTAINER_VERSION=22.10

#Stage.2 Triton Server
FROM nvcr.io/nvidia/tritonserver:22.10-py3 as tritonserver

#Stage.3 Flask & Numpy & Pandas & Torchserve & Triton Server
FROM $BASE_IMAGE_NAME:$BASE_IMAGE_TAG
ARG http_proxy
ARG https_proxy
ARG JDK_VERSION
ARG TINI_VERSION
ENV JDK_HOME				/opt/jdk${JDK_VERSION}
ENV JAVA_HOME				/opt/jdk${JDK_VERSION}
ENV TINI_VERSION                        $TINI_VERSION
ARG TRITON_VERSION
ARG TRITON_CONTAINER_VERSION
ENV TRITON_SERVER_VERSION ${TRITON_VERSION}
ENV NVIDIA_TRITON_SERVER_VERSION ${TRITON_CONTAINER_VERSION}
ENV PATH /opt/tritonserver/bin:${PATH}
ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH /usr/local/cuda/targets/x86_64-linux/lib:/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}
ENV PYTHONPATH /usr/lib/python3.8:/usr/lib/python3.8/lib-dynload:/usr/local/lib/python3.8/dist-packages:/usr/lib/python3/dist-packages

ENV TF_ADJUST_HUE_FUSED         1
ENV TF_ADJUST_SATURATION_FUSED  1
ENV TF_ENABLE_WINOGRAD_NONFUSED 1
ENV TF_AUTOTUNE_THRESHOLD       2
ENV TRITON_SERVER_GPU_ENABLED    0
ENV TRITON_SERVER_CPU_ONLY      1



ADD ./examples/ /ppml/examples
ADD ./entrypoint.sh /opt/entrypoint.sh
ADD ./start-scripts/ /ppml/work/start-scripts
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /sbin/tini
COPY --from=tritonserver /usr/local/cuda/lib64/stubs/libcusparse.so /usr/local/cuda/lib64/stubs/libcusparse.so.11
COPY --from=tritonserver /usr/local/cuda/lib64/stubs/libcusolver.so /usr/local/cuda/lib64/stubs/libcusolver.so.11
COPY --from=tritonserver /usr/local/cuda/lib64/stubs/libcurand.so /usr/local/cuda/lib64/stubs/libcurand.so.10
COPY --from=tritonserver /usr/local/cuda/lib64/stubs/libcufft.so /usr/local/cuda/lib64/stubs/libcufft.so.10
COPY --from=tritonserver /usr/local/cuda/lib64/stubs/libcublas.so /usr/local/cuda/lib64/stubs/libcublas.so.11
COPY --from=tritonserver /usr/local/cuda/lib64/stubs/libcublasLt.so /usr/local/cuda/lib64/stubs/libcublasLt.so.11
COPY --from=tritonserver /usr/local/cuda-11.8/targets/x86_64-linux/lib/libcudart.so.11.0 /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.11.0
COPY --from=tritonserver /usr/local/cuda-11.8/targets/x86_64-linux/lib/libcupti.so.11.8 /usr/local/cuda/targets/x86_64-linux/lib/libcupti.so.11.8
COPY --from=tritonserver /usr/local/cuda-11.8/targets/x86_64-linux/lib/libnvToolsExt.so.1 /usr/local/cuda/targets/x86_64-linux/lib/libnvToolsExt.so.1
COPY --from=tritonserver /usr/lib/x86_64-linux-gnu/libcudnn.so.8 /usr/lib/x86_64-linux-gnu/libcudnn.so.8
COPY --from=tritonserver /usr/lib/x86_64-linux-gnu/libnccl.so.2 /usr/lib/x86_64-linux-gnu/libnccl.so.2
COPY --from=tritonserver /opt/nvidia/ /opt/nvidia/
COPY --from=tritonserver /opt/tritonserver/ /opt/tritonserver/
COPY --from=tritonserver /usr/bin/ /usr/bin/

# Ensure apt-get won't prompt for selecting options

# Common dependencies. 
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common libb64-0d libcurl4-openssl-dev libre2-5 git dirmngr libnuma-dev curl libgomp1 && \
    rm -rf /var/lib/apt/lists/* && \

# patchelf is needed to add deps of libcublasLt.so.11 to libtorch_cuda.so
    apt-get update && \
    apt-get install -y --no-install-recommends openmpi-bin patchelf && \
    patchelf --add-needed /usr/local/cuda/lib64/stubs/libcublasLt.so.11 /opt/tritonserver/backends/pytorch/libtorch_cuda.so && \

    pip3 install --upgrade pip && \
    pip install --no-cache-dir numpy && \
    cp -r /usr/local/lib/python3.7/dist-packages/* /usr/local/lib/python3.8/dist-packages/ && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.8 /usr/bin/python3 && \
    chmod +x /opt/entrypoint.sh && \
    chmod +x /sbin/tini && \
    cp /sbin/tini /usr/bin/tini && \
    gramine-argv-serializer bash -c 'export TF_MKL_ALLOC_MAX_BYTES=10737418240 && export _SPARK_AUTH_SECRET=$_SPARK_AUTH_SECRET && $sgx_command' > /ppml/secured_argvs


WORKDIR /ppml

ENTRYPOINT [ "/opt/entrypoint.sh" ]


