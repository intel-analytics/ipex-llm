sudo docker run -it \
	--net=host \
	--name=occlum-spark-local \
	--cpuset-cpus 10-14 \
	--device=/dev/sgx \
	-v data:/ppml/docker-occlum/data \
	-e LOCAL_IP=$LOCAL_IP \
	intelanalytics/analytics-zoo-ppml-trusted-big-data-ml-scala-occlum:0.11-SNAPSHOT \
	bash /ppml/docker-occlum/run_spark_on_occlum_glibc.sh $1 && tail -f /dev/null
