FROM ubuntu:20.04 AS builder

# DCAP version (github repo branch, tag or commit hash)
ARG DCAP_VERSION=DCAP_1.13
ARG http_proxy
ARG https_proxy

# update and install packages
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -yq && \
    apt-get upgrade -yq && \
    apt-get install -yq --no-install-recommends \
            build-essential \
            ca-certificates \
            curl \
            git \
            zip  \
            tzdata \
            debhelper \
            libcurl4-openssl-dev \
            libcurl4 \
            vim \
            wget \
            python \
            systemctl
# install sgxsdk
RUN mkdir /opt/intel && \
    cd /opt/intel && \
    wget https://download.01.org/intel-sgx/sgx-dcap/1.13/linux/distro/ubuntu20.04-server/sgx_linux_x64_sdk_2.16.100.4.bin && \
    chmod a+x ./sgx_linux_x64_sdk_2.16.100.4.bin && \
    printf "no\n/opt/intel\n"|./sgx_linux_x64_sdk_2.16.100.4.bin && \
    . /opt/intel/sgxsdk/environment

# install node.js
RUN curl -sL https://deb.nodesource.com/setup_16.x | bash -
RUN DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends nodejs

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# clone DCAP repo
RUN git clone https://github.com/Uxito-Ada/SGXDataCenterAttestationPrimitives.git -b ${DCAP_VERSION} --depth 1

# install MPA
RUN mkdir /etc/init && cd /SGXDataCenterAttestationPrimitives/tools/SGXPlatformRegistration && \
    make deb_pkg && \
    cd build/installer && \
    dpkg -i *.deb

# set PWD to PCKCertSelection dir
WORKDIR /SGXDataCenterAttestationPrimitives/tools/PCKCertSelection/

# build libPCKCertSelection library and copy to lib folder
RUN make && \
    mkdir -p ../../QuoteGeneration/pccs/lib && \
    cp ./out/libPCKCertSelection.so ../../QuoteGeneration/pccs/lib/ && \
    make clean

# set PWD to PCCS dir
WORKDIR /SGXDataCenterAttestationPrimitives/QuoteGeneration/pccs/

# build pccs
RUN npm config set proxy $http_proxy && \
    npm config set http-proxy $http_proxy && \
    npm config set https-proxy $https_proxy && \
    npm config set engine-strict true && \
    npm install

# build final image
FROM ubuntu:20.04

COPY --from=builder /usr/bin/node /usr/bin/node
COPY --from=builder /SGXDataCenterAttestationPrimitives/QuoteGeneration/pccs/ /opt/intel/pccs/
ADD ./default.json /opt/intel/pccs/config/default.json
ADD ./entrypoint.sh /opt/intel/pccs/entrypoint.sh
RUN chmod 777 /opt/intel/pccs/entrypoint.sh && \
    apt-get update -yq && \
    apt-get upgrade -yq && \
    apt-get install -yq --no-install-recommends openssl
WORKDIR /opt/intel/pccs/

# entrypoint to start pccs
ENTRYPOINT [ "/opt/intel/pccs/entrypoint.sh" ]
