ARG BIGDL_VERSION=2.2.0-SNAPSHOT

FROM ubuntu:20.04
ARG http_proxy
ARG https_proxy
ARG no_proxy
ARG BIGDL_VERSION
ENV BIGDL_VERSION                       ${BIGDL_VERSION}
ENV LOCAL_IP                            127.0.0.1
ENV SGX_MEM_SIZE                        32G
ENV SGX_LOG_LEVEL                       error
ENV LC_ALL                              C.UTF-8
ENV LANG                                C.UTF-8



# TODO: remove these directories
# RUN mkdir -p /ppml/trusted-big-data-ml/work && \
#     mkdir -p /ppml/trusted-big-data-ml/work/lib && \
#     mkdir -p /ppml/trusted-big-data-ml/work/keys && \
#     mkdir -p /ppml/trusted-big-data-ml/work/password && \
#     mkdir -p /ppml/trusted-big-data-ml/work/data && \
#     mkdir -p /ppml/trusted-big-data-ml/work/models && \
#     mkdir -p /ppml/trusted-big-data-ml/work/apps && \
#     mkdir -p /ppml/trusted-big-data-ml/work/examples/bigdl && \
#     mkdir -p /root/.keras/datasets && \
#     mkdir -p /root/.zinc && \
#     mkdir -p /root/.m2 && \
#     mkdir -p /root/.kube/
#     mkdir -p /ppml/trusted-big-data-ml/fl

# SOLVED: change to use /ppml
RUN mkdir -p /ppml/

# /ppml/work/lib is the folder for gramine to load libraries, check loader.env.LD_LIBRARY_PATH

# SOLVED: check these four files, resolve any bad paths
ADD ./bash.manifest.template /ppml/bash.manifest.template
ADD ./Makefile /ppml/Makefile
ADD ./init.sh /ppml/init.sh
ADD ./clean.sh /ppml/clean.sh

ADD ./_dill.py.patch /_dill.py.patch
ADD ./python-uuid.patch /python-uuid.patch
ADD ./python-pslinux.patch /python-pslinux.patch

# Python3.7
RUN env DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install -y python3.7-minimal build-essential python3.7-distutils python3.7 python3-setuptools python3.7-dev python3-wheel python3-pip libpython3.7 && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.7 /usr/bin/python3 && \
    pip3 install --upgrade pip && \
    pip install setuptools==58.4.0 && \
    ln -s /usr/bin/python3 /usr/bin/python && \
# Gramine
    apt-get update --fix-missing && \
    env DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y apt-utils wget unzip protobuf-compiler && \
    pip install --no-cache-dir meson==0.63.2 cmake==3.24.1.1 toml==0.10.2 pyelftools cffi dill==0.3.4 psutil && \
# Create a link to cmake, upgrade cmake version so that it is compatible with latest meson build (require version >= 3.17)
    ln -s /usr/local/bin/cmake /usr/bin/cmake && \
    env DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential autoconf bison gawk git ninja-build python3-click python3-jinja2 wget pkg-config cmake libcurl4-openssl-dev libprotobuf-c-dev protobuf-c-compiler python3-cryptography python3-pip python3-protobuf nasm && \
    git clone https://github.com/analytics-zoo/gramine.git /gramine && \
    git clone https://github.com/intel/SGXDataCenterAttestationPrimitives.git /opt/intel/SGXDataCenterAttestationPrimitives && \
    cd /gramine && \
    git checkout devel-v1.3.1-2022-10-08 && \
    meson setup build/ --buildtype=release  -Dsgx=enabled   -Dsgx_driver=dcap1.10 && \
    ninja -C build/ && \
    ninja -C build/ install && \
    cd /ppml/ && \
# meson will copy the original file instead of the symlink, which enable us to delete the gramine directory entirely
    rm -rf /gramine && \
    apt-get update --fix-missing && \
    env DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y tzdata && \
    apt-get install -y apt-utils vim curl nano wget unzip git tree zip && \
    apt-get install -y libsm6 make build-essential && \
    apt-get install -y autoconf gawk bison libcurl4-openssl-dev python3-protobuf libprotobuf-c-dev protobuf-c-compiler && \
    apt-get install -y netcat net-tools && \
    patch /usr/local/lib/python3.7/dist-packages/dill/_dill.py /_dill.py.patch && \
    patch /usr/lib/python3.7/uuid.py /python-uuid.patch && \
    patch /usr/local/lib/python3.7/dist-packages/psutil/_pslinux.py /python-pslinux.patch && \
    cp /usr/lib/x86_64-linux-gnu/libpython3.7m.so.1 /usr/lib/libpython3.7m.so.1 && \
    chmod a+x /ppml/init.sh && \
    chmod a+x /ppml/clean.sh


ENV PYTHONPATH                          /usr/lib/python3.7:/usr/lib/python3.7/lib-dynload:/usr/local/lib/python3.7/dist-packages:/usr/lib/python3/dist-packages
# Change WORKDIR to /ppml/ later images can overwrite this
WORKDIR /ppml
ENTRYPOINT [ "/bin/bash" ]
