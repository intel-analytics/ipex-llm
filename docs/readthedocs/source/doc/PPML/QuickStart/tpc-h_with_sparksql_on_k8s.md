## TPC-H with Trusted SparkSQL on Kubernetes ##

### Prerequisites ###
- Hardware that supports SGX
- A fully configured Kubernetes cluster
- Intel SGX Device Plugin to use SGX in K8S cluster (install following instructions [here](https://bigdl.readthedocs.io/en/latest/doc/PPML/QuickStart/deploy_intel_sgx_device_plugin_for_kubernetes.html "here"))

### Prepare TPC-H kit and data ###
1. Generate data

Go to [TPC Download](https://www.tpc.org/tpc_documents_current_versions/current_specifications5.asp) site, choose `TPC-H` source code, then download the TPC-H toolkits. **Follow the download instructions carefully.**
After you download the tpc-h tools zip and uncompressed the zip file. Go to `dbgen` directory, and create `makefile` based on `makefile.suite`, and modify `makefile` according to the prompts inside, and run `make`.

This should generate an executable called `dbgen`
```
./dbgen -h
```

gives you the various options for generating the tables. The simplest case is running:
```
./dbgen
```
which generates tables with extension `.tbl` with scale 1 (default) for a total of rougly 1GB size across all tables. For different size tables you can use the `-s` option:
```
./dbgen -s 10
```
will generate roughly 10GB of input data.

You need to move all .tbl files to a new directory as raw data.

You can then either upload your data to remote file system or read them locally.

2. Encrypt Data

Encrypt data with specified Key Management Service (`SimpleKeyManagementService`, or `EHSMKeyManagementService` , or `AzureKeyManagementService`). Details can be found here: https://github.com/intel-analytics/BigDL/tree/main/ppml/services/kms-utils/docker

The example code of encrypt data with `SimpleKeyManagementService` is like below:
```
java -cp "$BIGDL_HOME/jars/bigdl-ppml-spark_3.1.2-2.1.0-SNAPSHOT.jar:$SPARK_HOME/conf/:$SPARK_HOME/jars/*:$BIGDL_HOME/jars/*"  \
   -Xmx10g \
   com.intel.analytics.bigdl.ppml.examples.tpch.EncryptFiles \
   --inputPath xxx/dbgen-input \
   --outputPath xxx/dbgen-encrypted
   --kmsType SimpleKeyManagementService
   --simpleAPPID xxxxxxxxxxxx \
   --simpleAPPKEY xxxxxxxxxxxx \
   --primaryKeyPath /path/to/simple_encrypted_primary_key \
   --dataKeyPath /path/to/simple_encrypted_data_key
```

### Deploy PPML TPC-H on Kubernetes ###
1.  Pull docker image
```
sudo docker pull intelanalytics/bigdl-ppml-trusted-big-data-ml-python-graphene:2.1.0-SNAPSHOT
```
2. Prepare SGX keys (following instructions [here](https://github.com/intel-analytics/BigDL/tree/main/ppml/trusted-big-data-ml/python/docker-graphene#11-prepare-the-keyspassworddataenclave-keypem "here")), make sure keys and tpch-spark can be accessed on each K8S node
3. Start a bigdl-ppml enabled Spark K8S client container with configured local IP, key, tpch and kuberconfig path
```
export ENCLAVE_KEY=/path/to/enclave-key.pem
export SECURE_PASSWORD_PATH=/path/to/password
export DATA_PATH=/path/to/data
export KEYS_PATH=/path/to/keys
export KUBERCONFIG_PATH=/path/to/kuberconfig
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
        -v $SECURE_PASSWORD_PATH:/ppml/trusted-big-data-ml/work/password \
        -v $ENCLAVE_KEY:/graphene/Pal/src/host/Linux-SGX/signer/enclave-key.pem \
        -v $DATA_PATH:/ppml/trusted-big-data-ml/work/data \
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
```bash
secure_password=`openssl rsautl -inkey /ppml/trusted-big-data-ml/work/password/key.txt -decrypt </ppml/trusted-big-data-ml/work/password/output.bin` && \
export TF_MKL_ALLOC_MAX_BYTES=10737418240 && \
export SPARK_LOCAL_IP=$LOCAL_IP && \
export INPUT_DIR=xxx/dbgen-encrypted && \
export OUTPUT_DIR=xxx/dbgen-output && \
  /opt/jdk8/bin/java \
    -cp '/ppml/trusted-big-data-ml/work/bigdl-2.1.0-SNAPSHOT/lib/bigdl-ppml-spark_3.1.2-2.1.0-SNAPSHOT-jar-with-dependencies.jar:/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*' \
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
    --properties-file /ppml/trusted-big-data-ml/work/bigdl-2.1.0-SNAPSHOT/conf/spark-bigdl.conf \
    --conf spark.kubernetes.authenticate.serviceAccountName=spark \
    --conf spark.kubernetes.container.image=$RUNTIME_K8S_SPARK_IMAGE \
    --conf spark.kubernetes.executor.podTemplateFile=/ppml/trusted-big-data-ml/spark-executor-template.yaml \
    --conf spark.kubernetes.executor.deleteOnTermination=false \
    --conf spark.kubernetes.executor.podNamePrefix=spark-tpch-sgx \
    --conf spark.kubernetes.sgx.enabled=true \
    --conf spark.kubernetes.sgx.executor.mem=32g \
    --conf spark.kubernetes.sgx.executor.jvm.mem=10g \
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
    --conf spark.bigdl.kms.type=SimpleKeyManagementService \
    --conf spark.bigdl.kms.simple.id=simpleAPPID \
    --conf spark.bigdl.kms.simple.key=simpleAPIKEY \
    --conf spark.bigdl.kms.key.primary=xxxx/primaryKey \
    --conf spark.bigdl.kms.key.data=xxxx/dataKey \
    --class com.intel.analytics.bigdl.ppml.examples.tpch.TpchQuery \
    --verbose \
    /ppml/trusted-big-data-ml/work/bigdl-2.1.0-SNAPSHOT/lib/bigdl-ppml-spark_3.1.2-2.1.0-SNAPSHOT-jar-with-dependencies.jar \
    $INPUT_DIR $OUTPUT_DIR aes/cbc/pkcs5padding plain_text [QUERY]
```
The optional parameter [QUERY] is the number of the query to run e.g 1, 2, ..., 22.

The result is in OUTPUT_DIR. There should be a file called TIMES.TXT with content formatted like:
>Q01     39.80204010