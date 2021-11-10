sudo docker run -it \
	--net=host \
	--name=bigdl-ppml-trusted-big-data-ml-scala-occlum \
	--cpuset-cpus 10-14 \
	--device=/dev/sgx/enclave \
	--device=/dev/sgx/provision \
	-v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
	-v data:/opt/data \
	-e LOCAL_IP=$LOCAL_IP \
	-e SGX_MEM_SIZE=24GB \
	intelanalytics/bigdl-ppml-trusted-big-data-ml-scala-occlum:0.14.0-SNAPSHOT \
	bash /opt/run_spark_on_occlum_glibc.sh $1 && tail -f /dev/null
