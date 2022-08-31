export KEYS_PATH=the_dir_path_of_your_prepared_keys
export SECURE_PASSWORD_PATH=the_dir_path_of_your_prepared_password
export LOCAL_IP=your_local_ip_of_the_sgx_server
sudo docker run -itd \
    -e REDIS_HOST=127.0.0.1 \
    --net=host \
    --cpuset-cpus="0-4" \
    --oom-kill-disable \
    --device=/dev/sgx/enclave \
    --device=/dev/sgx/provision \
    -v /var/run/aesmd:/var/run/aesmd \
    -v $KEYS_PATH:/opt/keys \
    -v $PWD/conf:/opt/conf \
    -v $SECURE_PASSWORD_PATH:/opt/password \
    --name=trusted-cluster-serving-local \
    -e LOCAL_IP=$LOCAL_IP \
    -e CORE_NUM=4 \
    -e SGX_THREAD=256 \
    -e SGX_HEAP=128MB \
    -e SGX_KERNEL_HEAP=512MB \
    -e SGX_MMAP=26000MB \
    intelanalytics/bigdl-ppml-trusted-realtime-ml-scala-occlum:2.1.0-SNAPSHOT \
    bash -c "export PATH=/opt/occlum/build/bin:$PATH && cd /opt/ && ./start-all.sh && tail -f /dev/null"
