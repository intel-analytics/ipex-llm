# Gramine Bigdata Toolkit

This image contains Gramine and some popular Big Data frameworks including Hadoop, Spark and Hive.

## Before Running Code
### 1. Build Docker Images

**Tip:** if you want to skip building the custom image, you can use our public image `intelanalytics/bigdl-ppml-trusted-bigdata-gramine-reference:2.2.0-SNAPSHOT` for a quick start, which is provided for a demo purpose. Do not use it in production.

#### 1.1 Build Bigdata Base Image

The bigdata base image is a public one that does not contain any secrets. You will use the base image to get your own custom image in the following.

You can use out pulic bigdata base image `intelanalytics/bigdl-ppml-trusted-bigdata-gramine-base:2.2.0-SNAPSHOT`, which is recommended. Or you can build your own base image, which is expected to be exactly same with our one.

Before building your own base image, please modify the paths in `ppml/trusted-bigdata/build-base-image.sh`. Then build the docker image with the following command.

```bash
./build-bigdata-base-image.sh
```

#### 1.2 Build Customer Image

First, You need to generate your enclave key using the command below, and keep it safely for future remote attestations and to start SGX enclaves more securely.

It will generate a file `enclave-key.pem` in `./custom-image`  directory, which will be your enclave key. To store the key elsewhere, modify the outputted file path.

```bash
cd custom-image
openssl genrsa -3 -out enclave-key.pem 3072
```

Then, use the `enclave-key.pem` and the bigdata base image to build your own custom image. In the process, SGX MREnclave will be made and signed without saving the sensitive encalve key inside the final image, which is safer.

```bash
# under custom-image dir
# modify custom parameters in build-custom-image.sh
./build-custom-image.sh
```

The docker build console will also output `mr_enclave` and `mr_signer` like below, which are hash values and used to  register your MREnclave in the following.

````bash
......
[INFO] Use the below hash values of mr_enclave and mr_signer to register enclave:
mr_enclave       : c7a8a42af......
mr_signer        : 6f0627955......
````
### 2. Prepare SSL key

#### 2.1 Prepare the Key

  The ppml in bigdl needs secured keys to enable spark security such as Authentication, RPC Encryption, Local Storage Encryption and TLS, you need to prepare the secure keys and keystores. In this tutorial, you can generate keys and keystores with root permission (test only, need input security password for keys).

```bash
  sudo bash BigDL/ppml/scripts/generate-keys.sh
```

#### 2.2 Prepare the Password

  Next, you need to store the password you used for key generation, i.e., `generate-keys.sh`, in a secured file.

```bash
  sudo bash BigDL/ppml/scripts/generate-password.sh <used_password_when_generate_keys>
```

### 3. Register MREnclave

#### 3.1 Deploy EHSM KMS&AS

KMS (Key Management Service) and AS (Attestation Service) make sure applications of the customer actually run in the SGX MREnclave signed above by customer-self, rather than a fake one fake by an attacker.

Bigdl ppml use EHSM as reference KMS&AS, you can deploy EHSM following a guide [here](https://github.com/intel-analytics/BigDL/tree/main/ppml/services/pccs-ehsm/kubernetes#deploy-bigdl-pccs-ehsm-kms-on-kubernetes-with-helm-charts).

#### 3.2 Enroll yourself on EHSM

Enroll yourself as below, The `<kms_ip>` is your configured-ip of EHSM service in the deployment section:

```bash
curl -v -k -G "https://<kms_ip>:9000/ehsm?Action=Enroll"
......
{"code":200,"message":"successful","result":{"apikey":"E8QKpBB******","appid":"8d5dd3b*******"}}
```

 You will get a `appid` and `apikey` pair and save it.

#### 3.3 Attest EHSM Server

##### 3.3.1 Start a BigDL client container

First, start a bigdl container, which uses the custom image build before.

```bash
export KEYS_PATH=YOUR_LOCAL_SPARK_SSL_KEYS_FOLDER_PATH
export LOCAL_IP=YOUR_LOCAL_IP
export CUSTOM_IMAGE=YOUR_CUSTOM_IMAGE_BUILT_BEFORE
export PCCS_URL=YOUR_PCCS_URL # format like https://1.2.3.4:xxxx, obtained from KMS services or a self-deployed one

sudo docker run -itd \
    --privileged \
    --net=host \
    --cpuset-cpus="0-5" \
    --oom-kill-disable \
    -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
    -v $KEYS_PATH:/ppml/keys \
    --name=gramine-verify-worker \
    -e LOCAL_IP=$LOCAL_IP \
    -e PCCS_URL=$PCCS_URL \
    $CUSTOM_IMAGE bash
```

Enter the work environment:

```bash
sudo docker exec -it gramine-verify-worker bash
```

##### 3.3.2 Verify  EHSM Quote

You need to first attest the EHSM server and verify the service as trusted before running workloads, to avoid sending your secrets to a fake EHSM service.

In the container, you can use `verify-attestation-service.sh` to verify the attestation service quote. Please set the variables in the script and then run it:

**Parameters in verify-attestation-service.sh:**

**ATTESTATION_URL**: URL of attestation service. Should match the format `<ip_address>:<port>`.

**APP_ID**, **API_KEY**: The appID and apiKey pair generated by your attestation service.

**ATTESTATION_TYPE**: Type of attestation service. Currently support `EHSMAttestationService`.

**CHALLENGE**: Challenge to get quote of attestation service which will be verified by local SGX SDK. Should be a BASE64 string. It can be a casual BASE64 string, for example, it can be generated by the command `echo anystring|base64`.

```bash
bash verify-attestation-service.sh
```

#### 3.4 Register your MREnclave to EHSM

Upload the metadata of your MREnclave obtained above to EHSM, and then only registerd MREnclave can pass the runtime verification in the following. You can register the MREnclave through running a python script:

```bash
# At /ppml inside the container now
python register-mrenclave.py --appid <your_appid> \
                             --apikey <your_apikey> \
                             --url https://<kms_ip>:9000 \
                             --mr_enclave <your_mrenclave_hash_value> \
                             --mr_signer <your_mrensigner_hash_value>
```

## Run Spark

Follow the guide below to run Spark on Kubernetes manually. Alternatively, you can also use Helm to set everything up automatically. See [kubernetes/README.md][helmGuide].

### 1. Start the spark client as Docker container
### 1.1 Prepare the keys/password/data/enclave-key.pem
Please refer to the previous section about [preparing keys and passwords](#2-prepare-spark-ssl-key).

``` bash
bash BigDL/ppml/scripts/generate-keys.sh
bash BigDL/ppml/scripts/generate-password.sh YOUR_PASSWORD
kubectl apply -f keys/keys.yaml
kubectl apply -f password/password.yaml
```

### 1.2 Prepare the k8s configurations
#### 1.2.1 Create the RBAC
```bash
kubectl create serviceaccount spark
kubectl create clusterrolebinding spark-role --clusterrole=edit --serviceaccount=default:spark --namespace=default
kubectl get secret|grep service-account-token # you will find a spark service account secret, format like spark-token-12345

# bind service account and user
kubectl config set-credentials spark-user \
--token=$(kubectl get secret <spark_service_account_secret> -o jsonpath={.data.token} | base64 -d)

# bind user and context
kubectl config set-context spark-context --user=spark-user

# bind context and cluster
kubectl config get-clusters
kubectl config set-context spark-context --cluster=<cluster_name> --user=spark-user
```
#### 1.2.2 Generate k8s config file
```bash
kubectl config use-context spark-context
kubectl config view --flatten --minify > /YOUR_DIR/kubeconfig
```
#### 1.2.3 Create k8s secret
```bash
kubectl create secret generic spark-secret --from-literal secret=YOUR_SECRET
kubectl create secret generic kms-secret \
                      --from-literal=app_id=YOUR_KMS_APP_ID \
                      --from-literal=api_key=YOUR_KMS_API_KEY \
                      --from-literal=policy_id=YOUR_POLICY_ID
kubectl create secret generic kubeconfig-secret --from-file=/YOUR_DIR/kubeconfig
```
**The secret created (`YOUR_SECRET`) should be the same as the password you specified in section 1.1**

### 1.3 Start the client container
Configure the environment variables in the following script before running it. Check [Bigdl ppml SGX related configurations](#1-bigdl-ppml-sgx-related-configurations) for detailed memory configurations.
```bash
export K8S_MASTER=k8s://$(sudo kubectl cluster-info | grep 'https.*6443' -o -m 1)
export NFS_INPUT_PATH=/YOUR_DIR/data
export KEYS_PATH=/YOUR_DIR/keys
export SECURE_PASSWORD_PATH=/YOUR_DIR/password
export KUBECONFIG_PATH=/YOUR_DIR/kubeconfig
export LOCAL_IP=$LOCAL_IP
export DOCKER_IMAGE=YOUR_DOCKER_IMAGE
sudo docker run -itd \
    --privileged \
    --net=host \
    --name=gramine-bigdata \
    --cpuset-cpus="20-24" \
    --oom-kill-disable \
    --device=/dev/sgx/enclave \
    --device=/dev/sgx/provision \
    -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
    -v $KEYS_PATH:/ppml/keys \
    -v $SECURE_PASSWORD_PATH:/ppml/password \
    -v $KUBECONFIG_PATH:/root/.kube/config \
    -v $NFS_INPUT_PATH:/ppml/data \
    -e RUNTIME_SPARK_MASTER=$K8S_MASTERK8S_MASTER \
    -e RUNTIME_K8S_SPARK_IMAGE=$DOCKER_IMAGE \
    -e RUNTIME_DRIVER_PORT=54321 \
    -e RUNTIME_DRIVER_MEMORY=10g \
    -e LOCAL_IP=$LOCAL_IP \
    $DOCKER_IMAGE bash
```
run `docker exec -it spark-local-k8s-client bash` to entry the container.

### 1.4 Init the client and run Spark applications on k8s (1.4 can be skipped if you are using 1.5 to submit jobs)

#### 1.4.1 Configure `spark-executor-template.yaml` in the container

We assume you have a working Network File System (NFS) configured for your Kubernetes cluster. Configure the `nfsvolumeclaim` on the last line to the name of the Persistent Volume Claim (PVC) of your NFS.

Please prepare the following and put them in your NFS directory:

- The data (in a directory called `data`),
- The kubeconfig file.

#### 1.4.2 Submit spark command

```bash
./init.sh
export secure_password=`openssl rsautl -inkey /ppml/password/key.txt -decrypt </ppml/password/output.bin`
export TF_MKL_ALLOC_MAX_BYTES=10737418240
export SPARK_LOCAL_IP=$LOCAL_IP
export sgx_command="${JAVA_HOME}/bin/java \
        -cp '${SPARK_HOME}/conf/:${SPARK_HOME}/jars/*:${SPARK_HOME}/examples/jars/*' \
        -Xmx8g \
        org.apache.spark.deploy.SparkSubmit \
        --master $RUNTIME_SPARK_MASTER \
        --deploy-mode $SPARK_MODE \
        --name spark-pi-sgx \
        --conf spark.driver.host=$SPARK_LOCAL_IP \
        --conf spark.driver.port=$RUNTIME_DRIVER_PORT \
        --conf spark.driver.memory=$RUNTIME_DRIVER_MEMORY \
        --conf spark.driver.cores=$RUNTIME_DRIVER_CORES \
        --conf spark.executor.cores=$RUNTIME_EXECUTOR_CORES \
        --conf spark.executor.memory=$RUNTIME_EXECUTOR_MEMORY \
        --conf spark.executor.instances=$RUNTIME_EXECUTOR_INSTANCES \
        --conf spark.kubernetes.authenticate.driver.serviceAccountName=spark \
        --conf spark.kubernetes.container.image=$RUNTIME_K8S_SPARK_IMAGE \
        --conf spark.kubernetes.driver.podTemplateFile=/ppml/spark-driver-template.yaml \
        --conf spark.kubernetes.executor.podTemplateFile=/ppml/spark-executor-template.yaml \
        --conf spark.kubernetes.executor.deleteOnTermination=false \
        --conf spark.network.timeout=10000000 \
        --conf spark.executor.heartbeatInterval=10000000 \
        --conf spark.python.use.daemon=false \
        --conf spark.python.worker.reuse=false \
        --conf spark.kubernetes.sgx.enabled=$SGX_ENABLED \
        --conf spark.kubernetes.sgx.driver.mem=$SGX_DRIVER_MEM \
        --conf spark.kubernetes.sgx.driver.jvm.mem=$SGX_DRIVER_JVM_MEM \
        --conf spark.kubernetes.sgx.executor.mem=$SGX_EXECUTOR_MEM \
        --conf spark.kubernetes.sgx.executor.jvm.mem=$SGX_EXECUTOR_JVM_MEM \
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
        --conf spark.ssl.keyStore=/ppml/keys/keystore.jks \
        --conf spark.ssl.keyStorePassword=$secure_password \
        --conf spark.ssl.keyStoreType=JKS \
        --conf spark.ssl.trustStore=/ppml/keys/keystore.jks \
        --conf spark.ssl.trustStorePassword=$secure_password \
        --conf spark.ssl.trustStoreType=JKS \
        --class org.apache.spark.examples.SparkPi \
        --verbose \
        --jars local://${SPARK_HOME}/examples/jars/spark-examples_2.12-${SPARK_VERSION}.jar \
        local://${SPARK_HOME}/examples/jars/spark-examples_2.12-${SPARK_VERSION}.jar 3000"
```

Note that: you can run your own Spark Appliction after changing `--class` and jar path.

1. `local://${SPARK_HOME}/examples/jars/spark-examples_2.12-${SPARK_VERSION}.jar` => `your_jar_path`
2. `--class org.apache.spark.examples.SparkPi` => `--class your_class_path`

#### 1.4.3 Spark-Pi example

```bash
gramine-sgx bash 2>&1 | tee spark-pi-sgx-$SPARK_MODE.log
```
### 1.5 Use bigdl-ppml-submit.sh to submit ppml jobs

Here, we assume you have started the client container and executed `init.sh`.

#### 1.5.1 Spark-Pi on local mode
![image2022-6-6_16-18-10](https://user-images.githubusercontent.com/61072813/174703141-63209559-05e1-4c4d-b096-6b862a9bed8a.png)
```
#!/bin/bash
bash bigdl-ppml-submit.sh \
        --master local[2] \
        --driver-memory 32g \
        --driver-cores 8 \
        --executor-memory 32g \
        --executor-cores 8 \
        --num-executors 2 \
        --class org.apache.spark.examples.SparkPi \
        --name spark-pi \
        --verbose \
        --log-file spark-pi-local.log \
        --jars local://${SPARK_HOME}/examples/jars/spark-examples_2.12-${SPARK_VERSION}.jar \
        local://${SPARK_HOME}/examples/jars/spark-examples_2.12-${SPARK_VERSION}.jar 3000
```
#### 1.5.2 Spark-Pi on local sgx mode
![image2022-6-6_16-18-57](https://user-images.githubusercontent.com/61072813/174703165-2afc280d-6a3d-431d-9856-dd5b3659214a.png)
```
#!/bin/bash
bash bigdl-ppml-submit.sh \
        --master local[2] \
        --sgx-enabled true \
        --sgx-log-level error \
        --sgx-driver-jvm-memory 12g\
        --sgx-executor-jvm-memory 12g\
        --driver-memory 32g \
        --driver-cores 8 \
        --executor-memory 32g \
        --executor-cores 8 \
        --num-executors 2 \
        --class org.apache.spark.examples.SparkPi \
        --name spark-pi \
        --log-file spark-pi-local-sgx.log
        --verbose \
        --jars local://${SPARK_HOME}/examples/jars/spark-examples_2.12-${SPARK_VERSION}.jar \
        local://${SPARK_HOME}/examples/jars/spark-examples_2.12-${SPARK_VERSION}.jar 3000

```
#### 1.5.3 Spark-Pi on client mode
![image2022-6-6_16-19-43](https://user-images.githubusercontent.com/61072813/174703216-70588315-7479-4b6c-9133-095104efc07d.png)

```
#!/bin/bash
export secure_password=`openssl rsautl -inkey /ppml/password/key.txt -decrypt </ppml/password/output.bin`
bash bigdl-ppml-submit.sh \
        --master $RUNTIME_SPARK_MASTER \
        --deploy-mode client \
        --sgx-enabled true \
        --sgx-log-level error \
        --sgx-driver-jvm-memory 12g\
        --sgx-executor-jvm-memory 12g\
        --driver-memory 32g \
        --driver-cores 8 \
        --executor-memory 32g \
        --executor-cores 8 \
        --num-executors 2 \
        --conf spark.kubernetes.container.image=$RUNTIME_K8S_SPARK_IMAGE \
        --class org.apache.spark.examples.SparkPi \
        --name spark-pi \
        --log-file spark-pi-client-sgx.log
        --verbose \
        --jars local://${SPARK_HOME}/examples/jars/spark-examples_2.12-${SPARK_VERSION}.jar \
        local://${SPARK_HOME}/examples/jars/spark-examples_2.12-${SPARK_VERSION}.jar 3000
```

#### 1.5.4 Spark-Pi on cluster mode
![image2022-6-6_16-20-0](https://user-images.githubusercontent.com/61072813/174703234-e45b8fe5-9c61-4d17-93ef-6b0c961a2f95.png)

```
#!/bin/bash
export secure_password=`openssl rsautl -inkey /ppml/password/key.txt -decrypt </ppml/password/output.bin`
bash bigdl-ppml-submit.sh \
        --master $RUNTIME_SPARK_MASTER \
        --deploy-mode cluster \
        --sgx-enabled true \
        --sgx-log-level error \
        --sgx-driver-jvm-memory 12g\
        --sgx-executor-jvm-memory 12g\
        --driver-memory 32g \
        --driver-cores 8 \
        --executor-memory 32g \
        --executor-cores 8 \
        --conf spark.kubernetes.container.image=$RUNTIME_K8S_SPARK_IMAGE \
        --num-executors 2 \
        --class org.apache.spark.examples.SparkPi \
        --name spark-pi \
        --log-file spark-pi-cluster-sgx.log
        --verbose \
        --jars local://${SPARK_HOME}/examples/jars/spark-examples_2.12-${SPARK_VERSION}.jar \
        local://${SPARK_HOME}/examples/jars/spark-examples_2.12-${SPARK_VERSION}.jar 3000
```
#### 1.5.5 bigdl-ppml-submit.sh explanations

bigdl-ppml-submit.sh is used to simplify the steps in 1.4

1. To use bigdl-ppml-submit.sh, first set the following required arguments:
```
--master $RUNTIME_SPARK_MASTER \
--deploy-mode cluster \
--driver-memory 32g \
--driver-cores 8 \
--executor-memory 32g \
--executor-cores 8 \
--sgx-enabled true \
--sgx-driver-jvm-memory 12g \
--sgx-executor-jvm-memory 12g \
--conf spark.kubernetes.container.image=$RUNTIME_K8S_SPARK_IMAGE \
--num-executors 2 \
--name spark-pi \
--verbose \
--class org.apache.spark.examples.SparkPi \
--log-file spark-pi-cluster-sgx.log
local://${SPARK_HOME}/examples/jars/spark-examples_2.12-${SPARK_VERSION}.jar 3000
```
if you are want to enable sgx, don't forget to set the sgx-related arguments
```
--sgx-enabled true \
--sgx-driver-memory 64g \
--sgx-driver-jvm-memory 12g \
--sgx-executor-memory 64g \
--sgx-executor-jvm-memory 12g \
```
you can update the application arguments to anything you want to run
```
--class org.apache.spark.examples.SparkPi \
local://${SPARK_HOME}/examples/jars/spark-examples_2.12-${SPARK_VERSION}.jar 3000
```

2. If you want to enable the spark security configurations as in 2.Spark security configurations, export secure_password to enable it.
```
export secure_password=`openssl rsautl -inkey /ppml/password/key.txt -decrypt </ppml/password/output.bin`
```

3. The following spark properties are set by default in bigdl-ppml-submit.sh. If you want to overwrite them or add new spark properties, just append the spark properties to bigdl-ppml-submit.sh as arguments.
```
--conf spark.driver.host=$LOCAL_IP \
--conf spark.driver.port=$RUNTIME_DRIVER_PORT \
--conf spark.network.timeout=10000000 \
--conf spark.executor.heartbeatInterval=10000000 \
--conf spark.python.use.daemon=false \
--conf spark.python.worker.reuse=false \
--conf spark.kubernetes.authenticate.driver.serviceAccountName=spark \
--conf spark.kubernetes.driver.podTemplateFile=/ppml/spark-driver-template.yaml \
--conf spark.kubernetes.executor.podTemplateFile=/ppml/spark-executor-template.yaml \
--conf spark.kubernetes.executor.deleteOnTermination=false \
```


### Configuration Explainations

#### 1. Bigdl ppml SGX related configurations

<img title="" src="../../docs/readthedocs/image/ppml_memory_config.png" alt="ppml_memory_config.png" data-align="center">

The following parameters enable spark executor running on SGX.
`spark.kubernetes.sgx.enabled`: true -> enable spark executor running on sgx, false -> native on k8s without SGX.
`spark.kubernetes.sgx.driver.mem`: Spark driver SGX epc memeory.
`spark.kubernetes.sgx.driver.jvm.mem`: Spark driver JVM memory, Recommended setting is less than half of epc memory.
`spark.kubernetes.sgx.executor.mem`: Spark executor SGX epc memeory.
`spark.kubernetes.sgx.executor.jvm.mem`: Spark executor JVM memory, Recommended setting is less than half of epc memory.
`spark.kubernetes.sgx.log.level`: Spark executor on SGX log level, Supported values are error,all and debug.
The following is a recommended configuration in client mode.
```bash
    --conf spark.kubernetes.sgx.enabled=true
    --conf spark.kubernetes.sgx.driver.jvm.mem=10g
    --conf spark.kubernetes.sgx.executor.jvm.mem=12g
    --conf spark.driver.memory=10g
    --conf spark.executor.memory=1g
```
The following is a recommended configuration in cluster mode.
```bash
    --conf spark.kubernetes.sgx.enabled=true
    --conf spark.kubernetes.sgx.driver.jvm.mem=10g
    --conf spark.kubernetes.sgx.executor.jvm.mem=12g
    --conf spark.driver.memory=1g
    --conf spark.executor.memory=1g
```
When SGX is not used, the configuration is the same as spark native.
```bash
    --conf spark.driver.memory=10g
    --conf spark.executor.memory=12g
```
#### 2. Spark security configurations
Below is an explanation of these security configurations, Please refer to [Spark Security](https://spark.apache.org/docs/3.1.2/security.html) for detail.
##### 2.1 Spark RPC
###### 2.1.1 Authentication
`spark.authenticate`: true -> Spark authenticates its internal connections, default is false.
`spark.authenticate.secret`: The secret key used authentication.
`spark.kubernetes.executor.secretKeyRef.SPARK_AUTHENTICATE_SECRET` and `spark.kubernetes.driver.secretKeyRef.SPARK_AUTHENTICATE_SECRET`: mount `SPARK_AUTHENTICATE_SECRET` environment variable from a secret for both the Driver and Executors.
`spark.authenticate.enableSaslEncryption`: true -> enable SASL-based encrypted communication, default is false.
```bash
    --conf spark.authenticate=true
    --conf spark.authenticate.secret=$secure_password
    --conf spark.kubernetes.executor.secretKeyRef.SPARK_AUTHENTICATE_SECRET="spark-secret:secret"
    --conf spark.kubernetes.driver.secretKeyRef.SPARK_AUTHENTICATE_SECRET="spark-secret:secret"
    --conf spark.authenticate.enableSaslEncryption=true
```

###### 2.1.2 Encryption
`spark.network.crypto.enabled`: true -> enable AES-based RPC encryption, default is false.
`spark.network.crypto.keyLength`: The length in bits of the encryption key to generate.
`spark.network.crypto.keyFactoryAlgorithm`: The key factory algorithm to use when generating encryption keys.
```bash
    --conf spark.network.crypto.enabled=true
    --conf spark.network.crypto.keyLength=128
    --conf spark.network.crypto.keyFactoryAlgorithm=PBKDF2WithHmacSHA1
```
###### 2.1.3. Local Storage Encryption
`spark.io.encryption.enabled`: true -> enable local disk I/O encryption, default is false.
`spark.io.encryption.keySizeBits`: IO encryption key size in bits.
`spark.io.encryption.keygen.algorithm`: The algorithm to use when generating the IO encryption key.
```bash
    --conf spark.io.encryption.enabled=true
    --conf spark.io.encryption.keySizeBits=128
    --conf spark.io.encryption.keygen.algorithm=HmacSHA1
```
###### 2.1.4 SSL Configuration
`spark.ssl.enabled`: true -> enable SSL.
`spark.ssl.port`: the port where the SSL service will listen on.
`spark.ssl.keyPassword`: the password to the private key in the key store.
`spark.ssl.keyStore`: path to the key store file.
`spark.ssl.keyStorePassword`: password to the key store.
`spark.ssl.keyStoreType`: the type of the key store.
`spark.ssl.trustStore`: path to the trust store file.
`spark.ssl.trustStorePassword`: password for the trust store.
`spark.ssl.trustStoreType`: the type of the trust store.
```bash
      --conf spark.ssl.enabled=true
      --conf spark.ssl.port=8043
      --conf spark.ssl.keyPassword=$secure_password
      --conf spark.ssl.keyStore=/ppml/keys/keystore.jks
      --conf spark.ssl.keyStorePassword=$secure_password
      --conf spark.ssl.keyStoreType=JKS
      --conf spark.ssl.trustStore=/ppml/keys/keystore.jks
      --conf spark.ssl.trustStorePassword=$secure_password
      --conf spark.ssl.trustStoreType=JKS
```
[helmGuide]: https://github.com/intel-analytics/BigDL/blob/main/ppml/python/docker-gramine/kubernetes/README.md