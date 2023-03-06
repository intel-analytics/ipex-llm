# Gramine Machine Learning Toolkit

This image contains Gramine and some popular Machine Learning frameworks including Spark and LightGBM. 

## Before Running Code
### 1. Build Docker Images
#### 1.1 Build Machine Learning Base Image

The machine learning base image is a public one that does not contain any secrets. You will use the base image to get your own custom image in the following.

Before building your own base image, please modify the paths in `ppml/trusted-machine-learning/build-base-image.sh`. Then build the docker image with the following command.

```bash
./build-machine-learning-base-image.sh
```
#### 1.2 Build Customer Image

First, You need to generate your enclave key using the command below, and keep it safely for future remote attestations and to start SGX enclaves more securely.

It will generate a file `enclave-key.pem` in `./custom-image`  directory, which will be your enclave key. To store the key elsewhere, modify the outputted file path.

```bash
cd custom-image
openssl genrsa -3 -out enclave-key.pem 3072
```

Then, use the `enclave-key.pem` and the machine learning base image to build your own custom image. In the process, SGX MREnclave will be made and signed without saving the sensitive encalve key inside the final image, which is safer.

```bash
# under custom-image dir
# modify custom parameters in build-custom-image.sh
./build-custom-image.sh
```

The docker build console will also output `mr_enclave` and `mr_signer` like below, which are hash values and used to register your MREnclave in the following.

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
  sudo bash BigDL/ppml/scripts/generate-password.sh          <used_password_when_generate_keys>
```

## Run machine learning example

### 1. Start the Docker container
#### 1.1 Prepare the keys/password/data/enclave-key.pem
Please refer to the previous section about [preparing keys and passwords](#2-prepare-spark-ssl-key).

``` bash
bash BigDL/ppml/scripts/generate-keys.sh
bash BigDL/ppml/scripts/generate-password.sh YOUR_PASSWORD
kubectl apply -f keys/keys.yaml
kubectl apply -f password/password.yaml
```
#### 1.2 Prepare the k8s configurations
##### 1.2.1 Create the RBAC
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
##### 1.2.2 Generate k8s config file
```bash
kubectl config use-context spark-context
kubectl config view --flatten --minify > /YOUR_DIR/kubeconfig
```
##### 1.2.3 Create k8s secret
```bash
kubectl create secret generic spark-secret --from-literal secret=YOUR_SECRET
kubectl create secret generic kms-secret \
                      --from-literal=app_id=YOUR_KMS_APP_ID \
                      --from-literal=api_key=YOUR_KMS_API_KEY \
                      --from-literal=policy_id=YOUR_POLICY_ID
kubectl create secret generic kubeconfig-secret --from-file=/YOUR_DIR/kubeconfig
```
**The secret created (`YOUR_SECRET`) should be the same as the password you specified in section 1.1**

#### 1.3 Start the client container
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
    --name=machine-learning-gramine \
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
run `docker exec -it machine-learning-graming bash` to entry the container.

#### 1.4 Run Machine Learning applications
Execute `init.sh` to check the SGX and make some necessary settings.
```bash 
bash init.sh
```

The trusted machine learning image porvides some classic examples of machine learning, including but not limited to random forest and linear regression. You can check the scripts in `/ppml/scripts` and execute one of them like this:
```bash 
cd scripts
bash start-random-forest-classifier-on-local-sgx.sh
```
You can also modify the jar path and class name, and run your own machine learning program like below:
```bash 
/opt/jdk8/bin/java \
    -cp "/ppml/spark-${SPARK_VERSION}/conf/:your_jar_path" -Xmx1g \
    org.apache.spark.deploy.SparkSubmit \
    --master local[2] \
    --driver-memory 32g \
    --driver-cores 8 \
    --executor-memory 32g \
    --executor-cores 8 \
    --num-executors 2 \
    --class your_class_path \
    --name your_program_name \
    --verbose \
    --jars local://${SPARK_HOME}/examples/jars/spark-examples_2.12-${SPARK_VERSION}.jar \
    local://${SPARK_HOME}/examples/jars/spark-examples_2.12-${SPARK_VERSION}.jar 3000
```
