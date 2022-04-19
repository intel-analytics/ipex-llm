## TPC-DS with Trusted SparkSQL on Kubernetes

### Prerequisites

- Hardware that supports SGX
- A fully configured Kubernetes cluster
- Intel SGX Device Plugin to use SGX in K8S cluster (install following instructions [here](https://bigdl.readthedocs.io/en/latest/doc/PPML/QuickStart/deploy_intel_sgx_device_plugin_for_kubernetes.html "here"))

### Prepare TPC-DS kit and data

1. Download and compile tpc-ds

```bash
git clone --recursive https://github.com/intel-analytics/zoo-tutorials.git
cd /path/to/zoo-tutorials
git clone https://github.com/databricks/tpcds-kit.git
cd tpcds-kit/tools
make OS=LINUX
```

2. Generate data

```bash
cd /path/to/zoo-tutorials
cd tpcds-spark/spark-sql-perf
sbt "test:runMain com.databricks.spark.sql.perf.tpcds.GenTPCDSData -d <dsdgenDir> -s <scaleFactor> -l <dataDir> -f parquet"
```

`dsdgenDir` is the path of `tpcds-kit/tools`, `scaleFactor` is the size of the data, for example `-s 1` will generate 1G data, `dataDir` is the path to store generated data.

### Deploy PPML TPC-DS on Kubernetes

1. Compile Kit

```bash
cd zoo-tutorials/tpcds-spark
sbt package
```

2. Create external tables

```bash
$SPARK_HOME/bin/spark-submit \
        --class "createTables" \
        --master <spark-master> \
        --driver-memory 20G \
        --executor-cores <executor-cores> \
        --total-executor-cores <total-cores> \
        --executor-memory 20G \
        --jars spark-sql-perf/target/scala-2.12/spark-sql-perf_2.12-0.5.1-SNAPSHOT.jar \
        target/scala-2.12/tpcds-benchmark_2.12-0.1.jar <dataDir> <dsdgenDir> <scaleFactor>
```

3. Pull docker image

```bash
sudo docker pull intelanalytics/bigdl-ppml-trusted-big-data-ml-python-graphene:2.1.0-SNAPSHOT
```

4. Prepare SGX keys (following instructions [here](https://github.com/intel-analytics/BigDL/tree/main/ppml/trusted-big-data-ml/python/docker-graphene#11-prepare-the-keyspassworddataenclave-keypem "here")), make sure keys and tpcds-spark can be accessed on each K8S node
5. Start a bigdl-ppml enabled Spark K8S client container with configured local IP, key, tpc-ds and kuberconfig path

```bash
export ENCLAVE_KEY=/YOUR_DIR/keys/enclave-key.pem
export DATA_PATH=/YOUR_DIR/zoo-tutorials/tpcds-spark
export KEYS_PATH=/YOUR_DIR/keys
export SECURE_PASSWORD_PATH=/YOUR_DIR/password
export KUBERCONFIG_PATH=/YOUR_DIR/kuberconfig
export LOCAL_IP=$local_ip
export DOCKER_IMAGE=intelanalytics/bigdl-ppml-trusted-big-data-ml-python-graphene:2.1.0-SNAPSHOT
sudo docker run -itd \
        --privileged \
        --net=host \
        --name=spark-local-k8s-client \
        --oom-kill-disable \
        --device=/dev/sgx/enclave \
        --device=/dev/sgx/provision \
        -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
        -v $ENCLAVE_KEY:/graphene/Pal/src/host/Linux-SGX/signer/enclave-key.pem \
        -v $DATA_PATH:/ppml/trusted-big-data-ml/work/tpcds-spark \
        -v $KEYS_PATH:/ppml/trusted-big-data-ml/work/keys \
	-v $SECURE_PASSWORD_PATH:/ppml/trusted-big-data-ml/work/password \
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

6. Attach to the client container

```bash
sudo docker exec -it spark-local-k8s-client bash
```

7. Modify `spark-executor-template.yaml`, add path of `enclave-key`, `tpcds-spark` and `kuberconfig` on host

```yaml
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: spark-executor
    securityContext:
      privileged: true
    volumeMounts:
	...
      - name: tpcds
        mountPath: /ppml/trusted-big-data-ml/work/tpcds-spark
      - name: kubeconf
        mountPath: /root/.kube/config
  volumes:
    - name: enclave-key
      hostPath:
        path:  /root/keys/enclave-key.pem
	...
    - name: tpcds
      hostPath:
        path: /path/to/tpcds-spark
    - name: kubeconf
      hostPath:
        path: /path/to/kuberconfig
```

8. Execute TPC-DS queries

Optional argument `QUERY` is the query number to run. Multiple query numbers should be separated by space, e.g. `1 2 3`. If no query number is specified, all 1-99 queries would be executed.

```bash
secure_password=`openssl rsautl -inkey /ppml/trusted-big-data-ml/work/password/key.txt -decrypt </ppml/trusted-big-data-ml/work/password/output.bin` && \
export TF_MKL_ALLOC_MAX_BYTES=10737418240 && \
export SPARK_LOCAL_IP=$LOCAL_IP && \
export HDFS_HOST=$hdfs_host_ip && \
export HDFS_PORT=$hdfs_port && \
export TPCDS_DIR=/ppml/trusted-big-data-ml/work/tpcds-spark \
export OUTPUT_DIR=hdfs://$HDFS_HOST:$HDFS_PORT/tpc-ds/output \
export QUERY=3
  /opt/jdk8/bin/java \
    -cp '$TPCDS_DIR/target/scala-2.12/tpcds-benchmark_2.12-0.1.jar:/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*' \
    -Xmx10g \
    -Dbigdl.mklNumThreads=1 \
    org.apache.spark.deploy.SparkSubmit \
    --master $RUNTIME_SPARK_MASTER \
    --deploy-mode client \
    --name spark-tpcds-sgx \
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
    --properties-file /ppml/trusted-big-data-ml/work/bigdl-2.1.0-SNAPSHOT/conf/spark-bigdl.conf \
    --conf spark.kubernetes.authenticate.serviceAccountName=spark \
    --conf spark.kubernetes.container.image=$RUNTIME_K8S_SPARK_IMAGE \
    --conf spark.kubernetes.executor.podTemplateFile=/ppml/trusted-big-data-ml/spark-executor-template.yaml \
    --conf spark.kubernetes.executor.deleteOnTermination=false \
    --conf spark.kubernetes.executor.podNamePrefix=spark-tpcds-sgx \
    --conf spark.kubernetes.sgx.enabled=true \
    --conf spark.kubernetes.sgx.executor.mem=32g \
    --conf spark.kubernetes.sgx.executor.jvm.mem=6g \
    --conf spark.kubernetes.sgx.log.level=$SGX_LOG_LEVEL \
    --conf spark.authenticate=true \
    --conf spark.authenticate.secret=$secure_password \
    --conf spark.kubernetes.executor.secretKeyRef.SPARK_AUTHENTICATE_SECRET="spark-secret:secret" \
    --conf spark.kubernetes.driver.secretKeyRef.SPARK_AUTHENTICATE_SECRET="spark-secret:secret" \
    --conf spark.authenticate.enableSaslEncryption=true \
    --conf spark.network.crypto.enabled=true \
    --conf spark.network.crypto.keyLength=128 \
    --conf spark.network.crypto.keyFactoryAlgorithm=PBKDF2WithHmacSHA1 \
    --conf spark.io.encryption.enabled=true \
    --conf spark.io.encryption.keySizeBits=128 \
    --conf spark.io.encryption.keygen.algorithm=HmacSHA1 \
    --conf spark.ssl.enabled=true \
    --conf spark.ssl.port=8043 \
    --conf spark.ssl.keyPassword=$secure_password \
    --conf spark.ssl.keyStore=/ppml/trusted-big-data-ml/work/keys/keystore.jks \
    --conf spark.ssl.keyStorePassword=$secure_password \
    --conf spark.ssl.keyStoreType=JKS \
    --conf spark.ssl.trustStore=/ppml/trusted-big-data-ml/work/keys/keystore.jks \
    --conf spark.ssl.trustStorePassword=$secure_password \
    --conf spark.ssl.trustStoreType=JKS \
    --class "TPCDSBenchmark" \
    --verbose \
    $TPCDS_DIR/target/scala-2.12/tpcds-benchmark_2.12-0.1.jar \
    $OUTPUT_DIR $QUERY
```

After benchmark is finished, the performance result is saved as `part-*.csv` file under `<OUTPUT_DIR>/performance` directory.
