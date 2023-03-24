ARG BIGDL_VERSION=2.3.0-SNAPSHOT
ARG SPARK_VERSION=3.1.3
ARG GRAMINE_BRANCH=devel-v1.3.1-2022-12-28

FROM ubuntu:20.04
ARG http_proxy
ARG https_proxy
ARG no_proxy
ARG BIGDL_VERSION
ARG GRAMINE_BRANCH
ARG JDK_VERSION=8u192
ARG JDK_URL
ARG SPARK_VERSION

ENV BIGDL_VERSION                       ${BIGDL_VERSION}
ENV LOCAL_IP                            127.0.0.1
ENV LC_ALL                              C.UTF-8
ENV LANG                                C.UTF-8
ENV JAVA_HOME                           /opt/jdk8
ENV PATH                                ${JAVA_HOME}/bin:${PATH}

ENV MALLOC_ARENA_MAX                    4

RUN mkdir -p /ppml/

ADD ./bash.manifest.template /ppml/bash.manifest.template
ADD ./Makefile /ppml/Makefile
ADD ./init.sh /ppml/init.sh
ADD ./clean.sh /ppml/clean.sh

# These files ared used for attestation service and register MREnclave
ADD ./register-mrenclave.py /ppml/register-mrenclave.py
ADD ./verify-attestation-service.sh /ppml/verify-attestation-service.sh

ADD ./_dill.py.patch /_dill.py.patch
ADD ./python-uuid.patch /python-uuid.patch
ADD ./python-pslinux.patch /python-pslinux.patch

# Python3.7
RUN env DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install -y python3.7-minimal build-essential python3.7-distutils python3.7 python3-setuptools python3.7-dev python3-wheel python3-pip libpython3.7 && \
# Python3.8
    apt-get install -y python3.8 python3.8-dev python3.8-distutils && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.8 /usr/bin/python3 && \
    pip3 install --upgrade pip && \
    pip3 install --no-cache requests argparse cryptography==3.3.2 urllib3 && \
    pip3 install --upgrade requests && \
    pip install setuptools==58.4.0 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
# Gramine
    apt-get update --fix-missing && \
    env DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y apt-utils wget unzip protobuf-compiler libgmp3-dev libmpfr-dev libmpfr-doc libmpc-dev && \
    pip install --no-cache-dir meson==0.63.2 cmake==3.24.1.1 toml==0.10.2 pyelftools cffi dill==0.3.4 psutil && \
# Create a link to cmake, upgrade cmake version so that it is compatible with latest meson build (require version >= 3.17)
    ln -s /usr/local/bin/cmake /usr/bin/cmake && \
    env DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential autoconf bison gawk git ninja-build python3-click python3-jinja2 wget pkg-config cmake libcurl4-openssl-dev libprotobuf-c-dev protobuf-c-compiler python3-cryptography python3-pip python3-protobuf nasm && \
    git clone https://github.com/analytics-zoo/gramine.git /gramine && \
    git clone https://github.com/intel/SGXDataCenterAttestationPrimitives.git /opt/intel/SGXDataCenterAttestationPrimitives && \
    cd /gramine && \
    git checkout ${GRAMINE_BRANCH} && \
# Also create the patched gomp
    meson setup build/ --buildtype=release  -Dsgx=enabled   -Dsgx_driver=dcap1.10 -Dlibgomp=enabled && \
    ninja -C build/ && \
    ninja -C build/ install && \
    cd /ppml/ && \
    mkdir -p /ppml/lib/ && \
    mv /usr/local/lib/x86_64-linux-gnu/gramine/runtime/glibc/libgomp.so.1 /ppml/lib/ && \
# meson will copy the original file instead of the symlink, which enable us to delete the gramine directory entirely
    rm -rf /gramine && \
    apt-get update --fix-missing && \
    env DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y tzdata && \
    apt-get install -y apt-utils vim curl nano wget unzip git tree zip && \
    apt-get install -y libsm6 make build-essential && \
    apt-get install -y autoconf gawk bison libcurl4-openssl-dev python3-protobuf libprotobuf-c-dev protobuf-c-compiler && \
    apt-get install -y netcat net-tools && \
    patch /usr/local/lib/python3.8/dist-packages/dill/_dill.py /_dill.py.patch && \
    patch /usr/lib/python3.8/uuid.py /python-uuid.patch && \
    patch /usr/local/lib/python3.8/dist-packages/psutil/_pslinux.py /python-pslinux.patch && \
    cp /usr/lib/x86_64-linux-gnu/libpython3.8.so.1 /usr/lib/libpython3.8.so.1 && \
    chmod a+x /ppml/init.sh && \
    chmod a+x /ppml/clean.sh && \
    chmod +x /ppml/verify-attestation-service.sh && \
# Install sgxsdk and dcap, which is used for remote attestation
    mkdir -p /opt/intel/ && \
    cd /opt/intel && \
    wget https://download.01.org/intel-sgx/sgx-dcap/1.16/linux/distro/ubuntu20.04-server/sgx_linux_x64_sdk_2.19.100.3.bin && \
    chmod a+x ./sgx_linux_x64_sdk_2.19.100.3.bin && \
    printf "no\n/opt/intel\n"|./sgx_linux_x64_sdk_2.19.100.3.bin && \
    . /opt/intel/sgxsdk/environment && \
    cd /opt/intel && \
    wget https://download.01.org/intel-sgx/sgx-dcap/1.16/linux/distro/ubuntu20.04-server/sgx_debian_local_repo.tgz && \
    tar xzf sgx_debian_local_repo.tgz && \
    echo 'deb [trusted=yes arch=amd64] file:///opt/intel/sgx_debian_local_repo focal main' | tee /etc/apt/sources.list.d/intel-sgx.list && \
    wget -qO - https://download.01.org/intel-sgx/sgx_repo/ubuntu/intel-sgx-deb.key | apt-key add - && \
    apt-get update && \
    apt-get install -y libsgx-enclave-common-dev  libsgx-ae-qe3 libsgx-ae-qve libsgx-urts libsgx-dcap-ql libsgx-dcap-default-qpl libsgx-dcap-quote-verify-dev libsgx-dcap-ql-dev libsgx-dcap-default-qpl-dev libsgx-quote-ex-dev libsgx-uae-service libsgx-ra-network libsgx-ra-uefi libtdx-attest libtdx-attest-dev && \
# java
    cd /opt && \
    wget $JDK_URL && \
    gunzip jdk-$JDK_VERSION-linux-x64.tar.gz && \
    tar -xf jdk-$JDK_VERSION-linux-x64.tar -C /opt && \
    rm jdk-$JDK_VERSION-linux-x64.tar && \
    mv /opt/jdk* /opt/jdk8 && \
    ln -s /opt/jdk8 /opt/jdk && \
# related dirs required by bash.manifest
    mkdir -p /root/.cache/ && \
    mkdir -p /root/.keras/datasets && \
    mkdir -p /root/.zinc && \
    mkdir -p /root/.m2 && \
    mkdir -p /root/.kube/ && \
    mkdir -p /ppml/encrypted-fs && \
# folder required for distributed encrypted file system
    mkdir -p /ppml/encrypted-fsd && \
# folder to contain the keys (primary key and data key)
    mkdir -p /ppml/encrypted_keys && \
# Install bigdl-ppml lib
    cd /ppml/ && \
    git clone https://github.com/intel-analytics/BigDL.git && \
    cd BigDL && \
    mkdir -p /ppml/bigdl-ppml/ && \
    cp -r /ppml/BigDL/python/ppml/src/ /ppml/bigdl-ppml && \
    rm -rf /ppml/BigDL && \
    cd /ppml


ADD ./download_jars.sh /ppml
ADD ./attestation.sh /ppml
ADD ./encrypted-fsd.sh /ppml

RUN cd /ppml && \
    bash ./download_jars.sh

ENV PYTHONPATH   /usr/lib/python3.8:/usr/lib/python3.8/lib-dynload:/usr/local/lib/python3.8/dist-packages:/usr/lib/python3/dist-packages:/ppml/bigdl-ppml/src
WORKDIR /ppml
ENTRYPOINT [ "/bin/bash" ]
