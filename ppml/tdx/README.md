# Privacy Preserving Machine Learning (PPML) TDX User Guide

TDX-based Trusted Big Data ML allows the user to run end-to-end big data analytics application and BigDL model training with distributed cluster on Intel Trust Domain Extensions (Intel TDX).

### Overview Architecture
![image](https://user-images.githubusercontent.com/30695225/190288851-fd852a51-f193-444c-bdea-1edad8375dd1.png)
### BigDL PPML on TDX-CC
![image](https://user-images.githubusercontent.com/30695225/190289025-dfcb3d01-9eed-4676-9df5-8412bd845894.png)

## Prepare TDX CC Environment
[`Confidential Containers`](https://github.com/confidential-containers/documentation/blob/main/Overview.md) (CC) is an open source community working to enable cloud native confidential computing by leveraging [`Trusted Execution Environments`](https://en.wikipedia.org/wiki/Trusted_execution_environment) (TEE) to protect containers and data.

The TEE seeks to protect the application and data from outside threats, with the application owner having complete control of all communication across the TEE boundary. The application is considered a single complete entity and once supplied with the resources it requires, the TEE protects those resources (memory and CPU) from the infrastructure and all communication across the TEE boundary is under the control of the application owner. 

Confidential Containers supports multiple TEE Technologies, such as Intel SGX and Intel TDX. [`Intel Trust Domain Extensions`](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-trust-domain-extensions.html) (Intel TDX) is introducing new, architectural elements to help deploy hardware-isolated, virtual machines (VMs) called trust domains (TDs). Intel TDX is designed to isolate VMs from the virtual-machine manager (VMM)/hypervisor and any other non-TD software on the platform to protect TDs from a broad range of software.

Combining the advantages of Intel TDX and Confidential Container, TDX-CC provides transparent deployment of unmodified containers and allows cloud-native application owners to enforce application security requirements.

To deploy an actual workload with TDX-CC, you need to prepare the environment in two parts, including **hardware environment** and **Kata CCv0**.

### Hardware Environment
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
###  **Kata CCv0**
Refer to the [`ccv0.sh`](https://github.com/kata-containers/kata-containers/blob/CCv0/docs/how-to/ccv0.sh) to install kata ccv0.
To ensure the successful creation of Kata confidential containers, please follow the [`how-to-build-and-test-ccv0`](https://github.com/kata-containers/kata-containers/blob/CCv0/docs/how-to/how-to-build-and-test-ccv0.md#using-kubernetes-for-end-to-end-provisioning-of-a-kata-confidential-containers-pod-with-an-unencrypted-image) to verify.

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

## Run as Spark Local Mode
Start the client pod
```bash
export KEYS_PATH=YOUR_LOCAL_KEYS_PATH
export DOCKER_IMAGE=intelanalytics/bigdl-tdx-client:latest

# modift tdx-client.yaml
kubectl apply -f tdx-client.yaml
```
Run `kubectl exec -it YOUR_CLIENT_POD -- /bin/bash` to entry the client pod.

The example for run Spark Pi:
```bash
bash spark-submit-with-ppml-tdx-local.sh \
    --master local[4] \
    --name spark-pi \
    --class org.apache.spark.examples.SparkPi \
    --conf spark.executor.instances=1 \
    local:///opt/spark/examples/jars/spark-examples_2.12-3.1.2.jar
```

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
export KUBECONFIG_PATH=KUBECONFIG_PATH
export LOCAL_IP=YOUR_LOCAL_IP
export DOCKER_IMAGE=intelanalytics/bigdl-tdx-client:latest

# modift tdx-client.yaml
kubectl apply -f tdx-client.yaml
```
Run `kubectl exec -it YOUR_CLIENT_POD -- /bin/bash` to entry the client pod.

### 2. Run application in spark K8S mode
#### 2.1 Run application in K8S client mode
Sample submit command for Simple Query example.
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
--jars ${BIGDL_HOME}/jars/bigdl-ppml-spark_3.1.2-*-jar-with-dependencies.jar \
${BIGDL_HOME}/jars/bigdl-ppml-spark_3.1.2-*-jar-with-dependencies.jar \
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
--jars ${BIGDL_HOME}/jars/bigdl-ppml-spark_3.1.2-*-jar-with-dependencies.jar \
${BIGDL_HOME}/jars/bigdl-ppml-spark_3.1.2-*-jar-with-dependencies.jar \
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
