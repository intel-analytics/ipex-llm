# Clean up old container
sudo docker rm -f bigdl-ppml-trusted-big-data-ml-scala-occlum-llm-test

# Run new command in container
sudo docker run -itd \
        --net=host \
        --name=bigdl-ppml-trusted-big-data-ml-scala-occlum-llm-test \
        -e LOCAL_IP=$LOCAL_IP \
        --cpuset-cpus=0-15 \
        -v /data/occlum_data:/opt/occlum_spark/data \
        -e SGX_MEM_SIZE=60GB \
        -e SGX_THREAD=2048 \
        -e SGX_KERNEL_HEAP=3GB \
        --device=/dev/sgx/enclave \
        --device=/dev/sgx/provision \
        -e SGX_LOG_LEVEL=off \
        intelanalytics/bigdl-ppml-trusted-llm-fastchat-occlum:2.4.0-SNAPSHOT \
        bash
