export DATA_PATH=the_dir_path_of_your_prepared_data
export KEYS_PATH=the_dir_path_of_your_prepared_keys
export LOCAL_IP=your_local_ip_of_the_sgx_server

sudo docker pull intelanalytics/analytics-zoo-ppml-trusted-big-data-ml-scala-graphene:0.10-SNAPSHOT 

sudo docker run -itd \
    --privileged \
    --net=host \
    --cpuset-cpus="0-5" \
    --oom-kill-disable \
    --device=/dev/gsgx \
    --device=/dev/sgx/enclave \
    --device=/dev/sgx/provision \
    -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
    -v $DATA_PATH:/ppml/trusted-big-data-ml/work/data \
    -v $KEYS_PATH:/ppml/trusted-big-data-ml/work/keys \
    --name=spark-local \
    -e LOCAL_IP=$LOCAL_IP \
    -e SGX_MEM_SIZE=64G \
    intelanalytics/analytics-zoo-ppml-trusted-big-data-ml-scala-graphene:0.10-SNAPSHOT
