# Privacy Preserving Machine Learning (PPML) TDX-VM User Guide

TDX-based Trusted Big Data ML allows the user to run end-to-end big data analytics application and BigDL model training with distributed cluster on Intel Trust Domain Extensions (Intel TDX). Currently, we provide TDX-VM and TDX-CC as Trusted Execution Environments, this user guide will focus on TDX-VM.

### Overview Architecture
![image](https://user-images.githubusercontent.com/30695225/190288851-fd852a51-f193-444c-bdea-1edad8375dd1.png)


## Prepare TDX-VM Environment

To deploy an actual workload with TDX-VM, you need to prepare the environment first.

1. Configure Hardware
    CPU and firmware need to be upgraded to the latest release version. Some jumpers must be set to enable TDX work on Archer City or Vulcan City board. 
2. Configure BIOS
    TDX should be enabled in BIOS. This step is required to be performed every time BIOS is flashed.
3. Build and install packages
    Packages of host kernel, guest kernel, qemu, libvirt should be built first. 
4. Setup TDX Guest Image
    A proper guest image utilizing the guest kernel, grub, and shim should be built. 
5. Launch TD Guests
    It is time to launch TD guests. Section Launch TD Guest leads you step by step to create and launch TD guests.
6. Verify statuses
    The Verify TDX Status section provides guidance on how to verify whether TDX is initializing on both the host and guest.
7. Test TDX
    TDX tests are used to validate basic functionality of TDX software stack. The tests focus on TDVM lifecycle management and environment validation.



## Run Sparkpi as Spark Local Mode
Start the client container
```bash
sudo docker run -itd --net=host \
    --name bigdl-ppml-client \
    intelanalytics/bigdl-ppml-trusted-bigd-data:2.3.0-SNAPSHOT bash
```
Run `docker exec -it bigdl-ppml-client bash` to entry the client container.

The example for run Spark Pi:
```bash
${SPARK_HOME}/bin/spark-submit \
  --class org.apache.spark.examples.SparkPi \
  --master local[2] \
  --executor-memory 2G \
  --num-executors 2 \
  local://${SPARK_HOME}/examples/jars/spark-examples_2.12-3.1.3.jar \
  1000
```

## Run Simple Query as Spark on Kubernetes Mode

### 1. Start the client container to run applications in spark K8s mode
#### 1.1 Prepare the keys and password
Please refer to the previous section about [prepare keys](#Prepare the key) and [prepare password](#Prepare the password).
```bash
bash ../../../scripts/generate-keys.sh
bash ../../../scripts/generate-password.sh YOUR_PASSWORD
kubectl apply -f keys/keys.yaml
kubectl apply -f password/password.yaml
```
#### 1.2 Prepare the k8s configurations
##### 1.2.1 Create the RBAC
```bash
kubectl create serviceaccount spark
kubectl create clusterrolebinding spark-role --clusterrole=edit --serviceaccount=default:spark --namespace=default
```
##### 1.2.2 Generate k8s config file
```bash
kubectl config view --flatten --minify > /YOUR_DIR/kubeconfig
```
##### 1.2.3 Create k8s secret
```bash
kubectl create secret generic spark-secret --from-literal secret=YOUR_PASSWORD
```
The secret created (YOUR_PASSWORD) should be the same as the password you specified in section 1.1 for generating the key.

#### 1.3 Start the client container
```bash
export K8S_MASTER=k8s://$(kubectl cluster-info | grep 'https.*6443' -o -m 1)
echo The k8s master is $K8S_MASTER .
export SPARK_IMAGE=intelanalytics/bigdl-ppml-trusted-bigdata-gramine-reference-16g:2.3.0-SNAPSHOT
sudo docker run -itd --net=host \
-v /etc/kubernetes:/etc/kubernetes \
-v /root/.kube/config:/root/.kube/config \
-v /mnt/data:/mnt/data \
-e RUNTIME_SPARK_MASTER=$K8S_MASTER \
-e RUNTIME_K8S_SPARK_IMAGE=$SPARK_IMAGE \
-e RUNTIME_PERSISTENT_VOLUME_CLAIM=task-pv-claim \
--name bigdl-ppml-client \
$SPARK_IMAGE bash
```
Run `docker exec -it bigdl-ppml-client bash` to entry the client container.


### 2. Encrypt Data
#### 2.1 prepare the training data people.csv

Use [generate_people_csv.py](https://github.com/intel-analytics/BigDL/blob/main/ppml/scripts/generate_people_csv.py) to generate the training data people.csv.

Execute `python generate_people_csv.py </save/path/of/people.csv> <num_lines>` to generate the training data people.csv, and upload people.csv to the /mnt/data/simplekms directory.

The people.csv data is shown in the figure below:

![image](https://user-images.githubusercontent.com/61072813/211201026-4cdaab09-e6c0-4d1d-95b0-450e40fa4c37.png)



#### 2.2 In client container, use simple kms to generate appid, apikey, primarykey to encrypt the training data people.csv.

Randomly generate appid and apikey with a length of 1 to 12 and store them in a safe place.


such as：  APPID： 984638161854
           APIKEY： 157809360993


Generate primarykey and datakey with appid and apikey. --primaryKeyPath specify where primarykey is stored.

```
java -cp '/ppml/spark-3.1.3/conf/:/ppml/spark-3.1.3/jars/*:/ppml/bigdl-2.3.0-SNAPSHOT/jars/*' \
com.intel.analytics.bigdl.ppml.examples.GeneratePrimaryKey \
        --primaryKeyPath /mnt/data/simplekms/primaryKey \
        --kmsType SimpleKeyManagementService \
        --simpleAPPID 984638161854 \
        --simpleAPIKEY 157809360993
```

#### 2.3 create encrypt.py

Switch to the directory /mnt/data/simplekms, create encrypt.py file, the content is as follows:
```
# encrypt.py
from bigdl.ppml.ppml_context import *
args = {"kms_type": "SimpleKeyManagementService",
        "app_id": "984638161854",
        "api_key": "157809360993",
        "primary_key_material": "/mnt/data/simplekms/primaryKey"
        }
sc = PPMLContext("PPMLTest", args)
csv_plain_path = "/mnt/data/simplekms/people.csv"
csv_plain_df = sc.read(CryptoMode.PLAIN_TEXT) \
            .option("header", "true") \
            .csv(csv_plain_path)
csv_plain_df.show()
output_path = "/mnt/data/simplekms/encrypted-input"
sc.write(csv_plain_df, CryptoMode.AES_CBC_PKCS5PADDING) \
    .mode('overwrite') \
    .option("header", True) \
    .csv(output_path)
```

#### 2.4 Use appid, apikey, primarykey to encrypt people.csv, and the encrypted data is stored under output_path.
```
java \
-cp '/ppml/spark-3.1.3/conf/:/ppml/spark-3.1.3/jars/*:/ppml/bigdl-2.3.0-SNAPSHOT/jars/*' \
-Xmx1g org.apache.spark.deploy.SparkSubmit \
--master 'local[4]' \
--conf spark.network.timeout=10000000 \
--conf spark.executor.heartbeatInterval=10000000 \
--conf spark.python.use.daemon=false \
--conf spark.python.worker.reuse=false \
--py-files /ppml/bigdl-2.3.0-SNAPSHOT/python/bigdl-ppml-spark_3.1.3-2.3.0-SNAPSHOT-python-api.zip,/ppml/bigdl-2.3.0-SNAPSHOT/python/bigdl-spark_3.1.3-2.3.0-SNAPSHOT-python-api.zip,/ppml/bigdl-2.3.0-SNAPSHOT/python/bigdl-dllib-spark_3.1.3-2.3.0-SNAPSHOT-python-api.zip \
/mnt/data/simplekms/encrypt.py
```



### 3. Run application in spark K8S mode
#### 3.1 Run application in K8S client mode
Sample submit command for Simple Query example.
```bash
${SPARK_HOME}/bin/spark-submit \
--master $RUNTIME_SPARK_MASTER \
--deploy-mode client \
--name spark-simplequery-tdx \
--conf spark.driver.memory=4g \
--conf spark.executor.cores=4 \
--conf spark.executor.memory=4g \
--conf spark.executor.instances=2 \
--conf spark.kubernetes.authenticate.driver.serviceAccountName=spark \
--conf spark.cores.max=8 \
--conf spark.kubernetes.container.image=$RUNTIME_K8S_SPARK_IMAGE \
--class com.intel.analytics.bigdl.ppml.examples.SimpleQuerySparkExample \
--conf spark.network.timeout=10000000 \
--conf spark.executor.heartbeatInterval=10000000 \
--conf spark.kubernetes.executor.deleteOnTermination=false \
--conf spark.driver.extraClassPath=local://${BIGDL_HOME}/jars/* \
--conf spark.executor.extraClassPath=local://${BIGDL_HOME}/jars/* \
--conf spark.kubernetes.file.upload.path=/mnt/data \
--conf spark.kubernetes.driver.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.options.claimName=${RUNTIME_PERSISTENT_VOLUME_CLAIM} \
--conf spark.kubernetes.driver.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.mount.path=/mnt/data \
--conf spark.kubernetes.executor.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.options.claimName=${RUNTIME_PERSISTENT_VOLUME_CLAIM} \
--conf spark.kubernetes.executor.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.mount.path=/mnt/data \
--jars local:///ppml/bigdl-2.3.0-SNAPSHOT/jars/bigdl-ppml-spark_3.1.3-2.3.0-SNAPSHOT.jar \
local:///ppml/bigdl-2.3.0-SNAPSHOT/jars/bigdl-ppml-spark_3.1.3-2.3.0-SNAPSHOT.jar \
--inputPartitionNum 8 \
--outputPartitionNum 8 \
--inputEncryptModeValue AES/CBC/PKCS5Padding \
--outputEncryptModeValue AES/CBC/PKCS5Padding \
--inputPath /mnt/data/simplekms/encrypted-input \
--outputPath /mnt/data/simplekms/encrypted-output \
--primaryKeyPath /mnt/data/simplekms/primaryKey \
--kmsType SimpleKeyManagementService \
--simpleAPPID 984638161854 \
--simpleAPIKEY 157809360993
```
#### 3.2 Run application in K8s cluster mode

```bash
${SPARK_HOME}/bin/spark-submit \
--master $RUNTIME_SPARK_MASTER \
--deploy-mode cluster \
--name spark-simplequery-tdx \
--conf spark.driver.memory=4g \
--conf spark.executor.cores=4 \
--conf spark.executor.memory=4g \
--conf spark.executor.instances=2 \
--conf spark.kubernetes.authenticate.driver.serviceAccountName=spark \
--conf spark.cores.max=8 \
--conf spark.kubernetes.container.image=$RUNTIME_K8S_SPARK_IMAGE \
--class com.intel.analytics.bigdl.ppml.examples.SimpleQuerySparkExample \
--conf spark.network.timeout=10000000 \
--conf spark.executor.heartbeatInterval=10000000 \
--conf spark.kubernetes.executor.deleteOnTermination=false \
--conf spark.driver.extraClassPath=local://${BIGDL_HOME}/jars/* \
--conf spark.executor.extraClassPath=local://${BIGDL_HOME}/jars/* \
--conf spark.kubernetes.file.upload.path=/mnt/data \
--conf spark.kubernetes.driver.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.options.claimName=${RUNTIME_PERSISTENT_VOLUME_CLAIM} \
--conf spark.kubernetes.driver.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.mount.path=/mnt/data \
--conf spark.kubernetes.executor.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.options.claimName=${RUNTIME_PERSISTENT_VOLUME_CLAIM} \
--conf spark.kubernetes.executor.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.mount.path=/mnt/data \
--jars local:///ppml/bigdl-2.3.0-SNAPSHOT/jars/bigdl-ppml-spark_3.1.3-2.3.0-SNAPSHOT.jar \
local:///ppml/bigdl-2.3.0-SNAPSHOT/jars/bigdl-ppml-spark_3.1.3-2.3.0-SNAPSHOT.jar \
--inputPartitionNum 8 \
--outputPartitionNum 8 \
--inputEncryptModeValue AES/CBC/PKCS5Padding \
--outputEncryptModeValue AES/CBC/PKCS5Padding \
--inputPath /mnt/data/simplekms/encrypted-input \
--outputPath /mnt/data/simplekms/encrypted-output \
--primaryKeyPath /mnt/data/simplekms/primaryKey \
--kmsType SimpleKeyManagementService \
--simpleAPPID 984638161854 \
--simpleAPIKEY 157809360993
```


### 4. Decrypt Data
#### 4.1 create decrypt.py

Switch to the directory /mnt/data/simplekms and create a decrypt.py file, the content of which is as follows:
```
from bigdl.ppml.ppml_context import *
args = {"kms_type": "SimpleKeyManagementService",
        "app_id": "984638161854",
        "api_key": "157809360993",
        "primary_key_material": "/mnt/data/simplekms/primaryKey"
        }
sc = PPMLContext("PPMLTest", args)
encrypted_csv_path = "/mnt/data/simplekms/encrypted-output"
csv_plain_df = sc.read(CryptoMode.AES_CBC_PKCS5PADDING) \
    .option("header", "true") \
    .csv(encrypted_csv_path)
csv_plain_df.show()
output_path = "/mnt/data/simplekms/decrypted-output"
sc.write(csv_plain_df, CryptoMode.PLAIN_TEXT) \
    .mode('overwrite') \
    .option("header", True)\
    .csv(output_path)
```


#### 4.2 Use appid, apikey, primarykey to decrypt the data in the encrypted_csv_path directory, and the decrypted data is stored in output_path.
```
java \
-cp '/ppml/spark-3.1.3/conf/:/ppml/spark-3.1.3/jars/*:/ppml/bigdl-2.3.0-SNAPSHOT/jars/*' \
-Xmx1g org.apache.spark.deploy.SparkSubmit \
--master 'local[4]' \
--conf spark.network.timeout=10000000 \
--conf spark.executor.heartbeatInterval=10000000 \
--conf spark.python.use.daemon=false \
--conf spark.python.worker.reuse=false \
--py-files /ppml/bigdl-2.3.0-SNAPSHOT/python/bigdl-ppml-spark_3.1.3-2.3.0-SNAPSHOT-python-api.zip,/ppml/bigdl-2.3.0-SNAPSHOT/python/bigdl-spark_3.1.3-2.3.0-SNAPSHOT-python-api.zip,/ppml/bigdl-2.3.0-SNAPSHOT/python/bigdl-dllib-spark_3.1.3-2.3.0-SNAPSHOT-python-api.zip \
/mnt/data/simplekms/decrypt.py
```

The decrypted result is as follows:：

![image](https://user-images.githubusercontent.com/61072813/211201115-dd15aeaa-14e1-478c-8252-3afaca27e896.png)
