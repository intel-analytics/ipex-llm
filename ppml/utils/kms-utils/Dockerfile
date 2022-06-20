FROM ubuntu:18.04

ARG EHSM_KMS_BRANCH_VERSION=main
ARG SPARK_JAR_REPO_URL
ARG HTTP_PROXY_HOST
ARG HTTP_PROXY_PORT
ARG HTTPS_PROXY_HOST
ARG HTTPS_PROXY_PORT

# Step 1. Install requirement tools
RUN apt-get update && apt-get install -y vim autoconf automake build-essential cmake curl debhelper git libcurl4-openssl-dev libprotobuf-dev libssl-dev libtool lsb-release ocaml ocamlbuild protobuf-compiler python wget libcurl4 libprotobuf10 libssl1.1 make module-init-tools g++ libjsoncpp-dev uuid-dev openjdk-8-jdk && \
# Step 2. Install Intel SGX SDK
    mkdir /opt/intel && cd /opt/intel && \
    wget https://download.01.org/intel-sgx/sgx-dcap/1.12.1/linux/distro/ubuntu18.04-server/sgx_linux_x64_sdk_2.15.101.1.bin && \
    chmod 777 sgx_linux_x64_sdk_2.15.101.1.bin && \
    sh -c 'echo yes | ./sgx_linux_x64_sdk_2.15.101.1.bin' && \
# Step 3. Install DCAP packages and set PCCS
# DCAP repository setup
    wget https://download.01.org/intel-sgx/sgx-dcap/1.12.1/linux/distro/ubuntu18.04-server/sgx_debian_local_repo.tgz && \
    tar xzf sgx_debian_local_repo.tgz && \
    echo 'deb [trusted=yes arch=amd64] file:///opt/intel/sgx_debian_local_repo bionic main' | tee /etc/apt/sources.list.d/intel-sgx.list && \
    wget -qO - https://download.01.org/intel-sgx/sgx_repo/ubuntu/intel-sgx-deb.key | apt-key add - && \
    apt-get update && \
    apt-get install -y libsgx-enclave-common-dev libsgx-ae-qe3 libsgx-ae-qve libsgx-urts libsgx-dcap-ql libsgx-dcap-default-qpl libsgx-dcap-quote-verify-dev libsgx-dcap-default-qpl-dev libsgx-quote-ex-dev libsgx-uae-service libsgx-ra-network libsgx-ra-uefi libsgx-dcap-ql-dev && \
# Step 4. Build enroll app from source
    cd /home && \
    git clone -b $EHSM_KMS_BRANCH_VERSION https://github.com/analytics-zoo/ehsm.git && cd ehsm && wget https://download.01.org/intel-sgx/sgx-linux/2.16/as.ld.objdump.r4.tar.gz && tar -zxf as.ld.objdump.r4.tar.gz && cp external/toolset/ubuntu18.04/* /usr/local/bin && make && cd out/ehsm-kms_enroll_app && ls ehsm-kms_enroll_app && \
# Step 5. Prepare bigdl ppml e2e environment
# scala
    cd /home && \
    wget $SPARK_JAR_REPO_URL/spark-encrypt-io-0.2-SNAPSHOT.jar && \
    mv spark-encrypt-io-0.2-SNAPSHOT.jar spark-encrypt-io.jar

ADD ./entrypoint.sh /home/entrypoint.sh
ADD ./pyspark-encrypt-io /home/pyspark-encrypt-io

# python
RUN apt-get install -y python3-minimal && \
    apt-get install -y build-essential python3 python3-setuptools python3-dev python3-pip && \
    pip3 install --upgrade pip && \
    pip install setuptools==58.4.0 && \
    pip install -r /home/pyspark-encrypt-io/src/bigdl/ppml/kms/requirement.yml

CMD ["sh", "sleep 10s"]
