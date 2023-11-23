ARG FINAL_NAME
FROM ${FINAL_NAME} as final
FROM krallin/ubuntu-tini AS tini
COPY --from=final /opt/occlum_spark /opt/occlum_spark
# remove image dir to reduce image size
RUN rm -rf /opt/occlum_spark/image
FROM ubuntu:20.04

MAINTAINER The BigDL Authors https://github.com/intel-analytics/BigDL
ARG BIGDL_VERSION=2.5.0-SNAPSHOT
ARG SPARK_VERSION=3.1.3
ARG HADOOP_VERSION=3.2.0
ARG SPARK_SCALA_VERSION=2.12
ENV SPARK_SCALA_VERSION=${SPARK_SCALA_VERSION}
ENV HADOOP_VERSION=${HADOOP_VERSION}
ENV SPARK_VERSION=${SPARK_VERSION}
ENV BIGDL_VERSION=${BIGDL_VERSION}
ENV SPARK_HOME=/opt/spark
ENV BIGDL_HOME=/opt/bigdl-${BIGDL_VERSION}
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
ENV SGX_MEM_SIZE=20GB

ARG HTTP_PROXY_HOST
ARG HTTP_PROXY_PORT
ARG HTTPS_PROXY_HOST
ARG HTTPS_PROXY_PORT
ARG HTTP_PROXY=http://$HTTP_PROXY_HOST:$HTTP_PROXY_PORT
ARG HTTPS_PROXY=http://$HTTPS_PROXY_HOST:$HTTPS_PROXY_PORT

#copy occlum runable instance
COPY --from=tini /opt/occlum_spark /opt/occlum_spark

# Configure sgx and occlum deb repo
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends vim ca-certificates gnupg2 jq make gdb wget libfuse-dev libtool tzdata && \
                echo 'deb [arch=amd64] https://download.01.org/intel-sgx/sgx_repo/ubuntu focal main' | tee /etc/apt/sources.list.d/intel-sgx.list && \
                wget -qO - https://download.01.org/intel-sgx/sgx_repo/ubuntu/intel-sgx-deb.key | apt-key add -

RUN echo 'deb [arch=amd64] https://occlum.io/occlum-package-repos/debian focal main' | tee /etc/apt/sources.list.d/occlum.list && \
                wget -qO - https://occlum.io/occlum-package-repos/debian/public.key | apt-key add -

#Install sgx dependencies and occlum
RUN apt-get update && apt-cache policy occlum
RUN apt-get install -y occlum
RUN apt-get install -y libsgx-uae-service libsgx-dcap-ql
# glibc
RUN apt install -y occlum-toolchains-glibc
RUN echo "source /etc/profile" >> $HOME/.bashrc

#COPY bash files
COPY --from=final /opt/*.sh /opt/
COPY --from=final /opt/sqlSuites /opt
COPY --from=final /opt/spark /opt/spark
COPY --from=final /var/run/secrets /var/run/secrets
COPY --from=final /opt/intel /opt/intel
COPY --from=tini /usr/local/bin/tini /sbin/tini
#COPY --from=final /etc /etc

# Add jdk and attestation ppml jars
COPY --from=final /usr/lib/jvm/java-8-openjdk-amd64 /usr/lib/jvm/java-8-openjdk-amd64
COPY --from=final ${BIGDL_HOME}/jars/bigdl* ${BIGDL_HOME}/jars/
COPY --from=final ${BIGDL_HOME}/jars/all-* ${BIGDL_HOME}/jars/

#Add quote attest lib
COPY --from=final /usr/lib/x86_64-linux-gnu/ /usr/lib/x86_64-linux-gnu/
RUN mkdir -p /etc/occlum_attestation/

# useful etc
COPY --from=final  /etc/java-8-openjdk  /etc/java-8-openjdk
COPY --from=final  /etc/hosts /etc/hosts
COPY --from=final  /etc/hostname  /etc/hostname
COPY --from=final  /etc/ssl /etc/ssl
COPY --from=final  /etc/passwd /etc/passwd
COPY --from=final  /etc/group /etc/group
COPY --from=final  /etc/nsswitch.conf /etc/nsswitch.conf


ENV PATH="/opt/occlum/build/bin:/usr/lib/jvm/java-8-openjdk-amd64/bin:/usr/local/occlum/bin:$PATH"
#ENV PATH="/opt/occlum/build/bin:/usr/local/occlum/bin:$PATH"

ENTRYPOINT [ "/opt/entrypoint.sh" ]
