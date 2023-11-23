ARG FINAL_NAME
FROM ${FINAL_NAME} as final
FROM krallin/ubuntu-tini AS tini
COPY --from=final /opt/occlum_spark /opt/occlum_spark
# remove image dir to reduce image size
RUN rm -rf /opt/occlum_spark/image
FROM ubuntu:20.04

MAINTAINER The BigDL Authors https://github.com/intel-analytics/BigDL
ARG BIGDL_VERSION=2.5.0-SNAPSHOT
ENV BIGDL_VERSION=${BIGDL_VERSION}
ENV BIGDL_HOME=/opt/bigdl-${BIGDL_VERSION}
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
ENV SGX_MEM_SIZE=32GB

#copy occlum runable instance
COPY --from=tini /opt/occlum_spark /opt/occlum_spark

ARG http_proxy
ARG https_proxy

# Configure sgx and occlum deb repo
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends vim ca-certificates gnupg2 jq make gdb wget libfuse-dev libtool tzdata
RUN echo 'deb [arch=amd64] https://download.01.org/intel-sgx/sgx_repo/ubuntu focal main' | tee /etc/apt/sources.list.d/intel-sgx.list && \
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
COPY --from=final /var/run/secrets /var/run/secrets
COPY --from=final /opt/intel /opt/intel
COPY --from=tini /usr/local/bin/tini /sbin/tini
#COPY --from=final /etc /etc

# Add jdk and attestation ppml jars
COPY --from=final /usr/lib/jvm/java-8-openjdk-amd64 /usr/lib/jvm/java-8-openjdk-amd64

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
