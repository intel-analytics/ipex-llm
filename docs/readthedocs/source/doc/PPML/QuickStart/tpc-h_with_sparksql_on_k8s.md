## TPC-H with Trusted SparkSQL on Kubernetes ##

### Prerequisites ###
- Hardware that supports SGX
- A fully configured Kubernetes cluster 
- Intel SGX Device Plugin to use SGX in K8S cluster (install following instructions [here](https://bigdl.readthedocs.io/en/latest/doc/PPML/QuickStart/deploy_intel_sgx_device_plugin_for_kubernetes.html "here"))

### Prepare TPC-H kit and data ###
1. Download and compile tpc-h 
```
git clone https://github.com/intel-analytics/zoo-tutorials.git
cd zoo-tutorials/tpch-spark

sed -i 's/2.11.7/2.12.1/g' tpch.sbt
sed -i 's/2.4.0/3.1.2/g' tpch.sbt
sbt package

cd dbgen
make
```
2. Generate data

Generate input data with size ~100GB (user can adjust data size to need):
```
./dbgen -s 100
```

### Deploy PPML TPC-H on Kubernetes ###
1.  Pull docker image
```
sudo docker pull intelanalytics/bigdl-ppml-trusted-big-data-ml-python-graphene:0.14.0-SNAPSHOT
```
2. Prepare SGX keys, make sure keys and tpch-spark can be accessed on each K8S node
3. Start a bigdl-ppml enabled Spark K8S client container with configured local IP, key, tpch and kuberconfig path
```
export ENCLAVE_KEY=/root/keys/enclave-key.pem
export DATA_PATH=/root/zoo-tutorials/tpch-spark
export KEYS_PATH=/root/keys
export KUBERCONFIG_PATH=/root/kuberconfig
export LOCAL_IP=$local_ip
export DOCKER_IMAGE=intelanalytics/bigdl-ppml-trusted-big-data-ml-python-graphene:0.14.0-SNAPSHOT
sudo docker run -itd \
        --privileged \
        --net=host \
        --name=spark-local-k8s-client \
        --oom-kill-disable \
        --device=/dev/sgx/enclave \
        --device=/dev/sgx/provision \
        -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
        -v $ENCLAVE_KEY:/graphene/Pal/src/host/Linux-SGX/signer/enclave-key.pem \
        -v $DATA_PATH:/ppml/trusted-big-data-ml/work/tpch-spark \
        -v $KEYS_PATH:/ppml/trusted-big-data-ml/work/keys \
        -v $KUBERCONFIG_PATH:/root/.kube/config \
        -e RUNTIME_SPARK_MASTER=k8s://https://$LOCAL_IP:6443 \
        -e RUNTIME_K8S_SERVICE_ACCOUNT=spark \
        -e RUNTIME_K8S_SPARK_IMAGE=$DOCKER_IMAGE \
        -e RUNTIME_DRIVER_HOST=$LOCAL_IP \
        -e RUNTIME_DRIVER_PORT=54321 \
        -e RUNTIME_EXECUTOR_INSTANCES=1 \
        -e RUNTIME_EXECUTOR_CORES=4 \
        -e RUNTIME_EXECUTOR_MEMORY=20g \
        -e RUNTIME_TOTAL_EXECUTOR_CORES=4 \
        -e RUNTIME_DRIVER_CORES=4 \
        -e RUNTIME_DRIVER_MEMORY=10g \
        -e SGX_MEM_SIZE=64G \
        -e SGX_LOG_LEVEL=error \
        -e LOCAL_IP=$LOCAL_IP \
        $DOCKER_IMAGE bash
``` 
4. Attach to the client container
```
sudo docker exec -it spark-local-k8s-client bash
```
5. Modify `spark-executor-template.yaml`, add path of `enclave-key`, `tpch-spark` and `kuberconfig` on host
```
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: spark-executor
    securityContext:
      privileged: true
    volumeMounts:
	...
      - name: tpch
        mountPath: /ppml/trusted-big-data-ml/work/tpch-spark
      - name: kubeconf
        mountPath: /root/.kube/config
  volumes:
    - name: enclave-key
      hostPath:
        path:  /root/keys/enclave-key.pem
	...
    - name: tpch
      hostPath:
        path: /path/to/tpch-spark
    - name: kubeconf
      hostPath:
        path: /path/to/kuberconfig
```
6. Run PPML TPC-H
```
export TF_MKL_ALLOC_MAX_BYTES=10737418240 && \
export SPARK_LOCAL_IP=$LOCAL_IP && \
export HDFS_HOST=$hdfs_host_ip && \
export HDFS_PORT=$hdfs_port && \
export TPCH_DIR=/ppml/trusted-big-data-ml/work/tpch-spark \
export INPUT_DIR=$TPCH_DIR/dbgen \
export OUTPUT_DIR=hdfs://$HDFS_HOST:$HDFS_PORT/tpc-h/output \
  /opt/jdk8/bin/java \
    -cp '$TPCH_DIR/target/scala-2.12/spark-tpc-h-queries_2.12-1.0.jar:$TPCH_DIR/dbgen/*:/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*' \
    -Xmx10g \
    -Dbigdl.mklNumThreads=1 \
    org.apache.spark.deploy.SparkSubmit \
    --master $RUNTIME_SPARK_MASTER \
    --deploy-mode client \
    --name spark-tpch-sgx \
    --conf spark.driver.host=$LOCAL_IP \
    --conf spark.driver.port=54321 \
    --conf spark.driver.memory=10g \
    --conf spark.driver.blockManager.port=10026 \
    --conf spark.blockManager.port=10025 \
    --conf spark.scheduler.maxRegisteredResourcesWaitingTime=5000000 \
    --conf spark.worker.timeout=600 \
    --conf spark.python.use.daemon=false \
    --conf spark.python.worker.reuse=false \
    --conf spark.network.timeout=10000000 \
    --conf spark.starvation.timeout=250000 \
    --conf spark.rpc.askTimeout=600 \
    --conf spark.sql.autoBroadcastJoinThreshold=-1 \
    --conf spark.io.compression.codec=lz4 \
    --conf spark.sql.shuffle.partitions=8 \
    --conf spark.speculation=false \
    --conf spark.executor.heartbeatInterval=10000000 \
    --conf spark.executor.instances=24 \
    --executor-cores 8 \
    --total-executor-cores 192 \
    --executor-memory 16G \
    --properties-file /ppml/trusted-big-data-ml/work/bigdl-0.14.0-SNAPSHOT/conf/spark-bigdl.conf \
    --conf spark.kubernetes.authenticate.serviceAccountName=spark \
    --conf spark.kubernetes.container.image=$RUNTIME_K8S_SPARK_IMAGE \
    --conf spark.kubernetes.executor.podTemplateFile=/ppml/trusted-big-data-ml/spark-executor-template.yaml \
    --conf spark.kubernetes.executor.deleteOnTermination=false \
    --conf spark.kubernetes.executor.podNamePrefix=spark-tpch-sgx \
    --conf spark.kubernetes.sgx.enabled=true \
    --conf spark.kubernetes.sgx.mem=32g \
    --conf spark.kubernetes.sgx.jvm.mem=10g \
    --class main.scala.TpchQuery \
    --verbose \
    $TPCH_DIR/target/scala-2.12/spark-tpc-h-queries_2.12-1.0.jar \
    $INPUT_DIR $OUTPUT_DIR
```
