# Privacy Preserving Machine Learning (PPML) TDX User Guide

TDX-based Trusted Big Data ML allows the user to run end-to-end big data analytics application and BigDL model training with distributed cluster on Intel Trust Domain Extensions (Intel TDX).

- [Before running the code](#before-running-the-code)
- [Run as Spark on Kubernetes Mode](#run-as-spark-on-kubernetes-mode)

## Before running the code
#### 1. Prepare the key
BigDL PPML needs secured keys to enable spark security such as Authentication, RPC Encryption, Local Storage Encryption and TLS, you need to prepare the secure keys and keystores. In this tutorial, you can generate keys and keystores with root permission (test only, need input security password for keys).

```bash
git clone https://github.com/intel-analytics/BigDL.git
bash ppml/scripts/generate-keys.sh
```
It will generate the keystores in `./keys`.
#### 2. Prepare the password
Next, you need to store the password you used for key generation in a secured file.

```bash
bash ppml/scripts/generate-password.sh used_password_when_generate_keys
```
It will generate that in `./password`.

## Run as Spark on Kubernetes Mode
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
export K8S_MASTER=k8s://$(sudo kubectl cluster-info | grep 'https.*6443' -o -m 1)
export KEYS_PATH=YOUR_LOCAL_KEYS_PATH
export SECURE_PASSWORD_PATH=YOUR_LOCAL_PASSWORD_PATH
export KUBECONFIG_PATH=KUBECONFIG_PATH
export LOCAL_IP=YOUR_LOCAL_IP
export DOCKER_IMAGE=intelanalytics/bigdl-k8s:latest

kubectl apply -f tdx-client.yaml
```
Run `kubectl exec spark-local-client -- /bin/bash` to entry the client pod.
### 2. Run application in spark K8S mode
#### 2.1 Run application in K8S client mode

```bash
export secure_password=.. && \
bash spark-submit-with-ppml-tdx-k8s.sh \
--master k8s://https://x.x.x.x:6443 \
--deploy-mode client \
--name spark-tdx \
--conf spark.driver.host=x.x.x.x \
--conf spark.driver.port=54321 \
--conf spark.driver.memory=8g \
--conf spark.executor.cores=8 \
--conf spark.executor.memory=8g \
--conf spark.executor.instances=1 \
--conf spark.cores.max=8 \
--class com.intel.analytics.bigdl.ppml.examples.SimpleQuerySparkExample \
--jars /bigdl2.0/data/bigdl-ppml-spark_3.1.2-2.1.0-SNAPSHOT-jar-with-dependencies.jar \
/bigdl2.0/data/bigdl-ppml-spark_3.1.2-2.1.0-SNAPSHOT-jar-with-dependencies.jar \
--inputPath /people/encrypted \
--outputPath /people/people_encrypted_output \
--inputPartitionNum 8 \
--outputPartitionNum 8 \
--inputEncryptModeValue AES/CBC/PKCS5Padding \
--outputEncryptModeValue AES/CBC/PKCS5Padding \
--primaryKeyPath /keys/primaryKey \
--dataKeyPath /keys/dataKey \
--kmsType SimpleKeyManagementService \
--simpleAPPID xx \
--simpleAPPKEY xx
```
#### 2.2 Run application in K8s cluster mode

```bash
export secure_password=.. && \
bash spark-submit-with-ppml-tdx-k8s.sh \
--master k8s://https://x.x.x.x:6443 \
--deploy-mode cluster \
--name spark-tdx \
--conf spark.driver.memory=8g \
--conf spark.executor.cores=8 \
--conf spark.executor.memory=8g \
--conf spark.executor.instances=1 \
--conf spark.cores.max=8 \
--class com.intel.analytics.bigdl.ppml.examples.SimpleQuerySparkExample \
--jars /bigdl2.0/data/bigdl-ppml-spark_3.1.2-2.1.0-SNAPSHOT-jar-with-dependencies.jar \
/bigdl2.0/data/bigdl-ppml-spark_3.1.2-2.1.0-SNAPSHOT-jar-with-dependencies.jar \
--inputPath /people/encrypted \
--outputPath /people/people_encrypted_output \
--inputPartitionNum 8 \
--outputPartitionNum 8 \
--inputEncryptModeValue AES/CBC/PKCS5Padding \
--outputEncryptModeValue AES/CBC/PKCS5Padding \
--primaryKeyPath /keys/primaryKey \
--dataKeyPath /keys/dataKey \
--kmsType SimpleKeyManagementService \
--simpleAPPID $simpleAPPID \
--simpleAPPKEY $simpleAPPKEY
```
