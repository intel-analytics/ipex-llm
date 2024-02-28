export DOCKER_IMAGE=intelanalytics/bigdl-ppml-trusted-bigdl-llm-gramine-ref:2.5.0-SNAPSHOT
export DOCKER_NAME=gramine-test-fastchat
docker run -itd \
        --privileged \
        --net=host \
        --name=$DOCKER_NAME \
        --cpuset-cpus="0-47" \
         -v /mnt/sde/tpch-data:/ppml/data \
        --shm-size="16g" \
        --memory="64g" \
        -e LOCAL_IP=$LOCAL_IP \
        -e https_proxy=http://child-prc.intel.com:913/ \
        -e http_proxy=http://child-prc.intel.com:913/ \
        -v $NFS_INPUT_PATH:/ppml/data \
        --device=/dev/sgx/enclave \
        --device=/dev/sgx/provision \
        -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
        $DOCKER_IMAGE bash
