sudo docker run -it \
	--net=host \
	--name=bigdl-ppml-trusted-big-data-ml-scala-occlum \
	--cpuset-cpus 10-14 \
	--device=/dev/sgx \
	-v data:/opt/data \
	-e LOCAL_IP=$LOCAL_IP \
	intelanalytics/bigdl-ppml-trusted-big-data-ml-scala-occlum:0.14.0-SNAPSHOT \
	bash /ppml/docker-occlum/run_spark_on_occlum_glibc.sh $1 && tail -f /dev/null
