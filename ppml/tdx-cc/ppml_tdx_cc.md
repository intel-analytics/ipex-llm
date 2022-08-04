# Privacy Preserving Machine Learning (PPML) on Azure User Guide

TDX-based Trusted Big Data ML allows the user to run end-to-end big data analytics application and BigDL model training with spark local and distributed cluster on Intel Trust Domain Extensions (Intel TDX).

## Befor running the  code
#### 1. Prepare the key
The ppml in bigdl needs secured keys to enable spark security such as Authentication, RPC Encryption, Local Storage Encryption and TLS, you need to prepare the secure keys and keystores. In this tutorial, you can generate keys and keystores with root permission (test only, need input security password for keys).

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

## Run as Spsrk Local Mode
### 1. Start the client container to run applications in spark local mode
```bash
```bash
export KEYS_PATH=YOUR_LOCAL_KEYS_PATH
export LOCAL_IP=YOUR_LOCAL_IP
export DOCKER_IMAGE=intelanalytics/bigdl-k8s:latest

sudo docker run -itd \
    --privileged \
    --net=host \
    -v $KEYS_PATH:/opt/spark/work-dir/keys \
    --name=spark-local-client \
    -e LOCAL_IP=$LOCAL_IP \
    $DOCKER_IMAGE bash
```
Run `docker exec -it spark-local-client bash` to entry the container.
### 2. Run applications in spark local mode
The example for run Spark Pi:
```bash
spark-submit-with-ppml.sh
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
export SECURE_PASSWORD_PATH=YOUR_LOCAL_PASSWORD_PATH
export KUBECONFIG_PATH=KUBECONFIG_PATH
export LOCAL_IP=YOUR_LOCAL_IP
export DOCKER_IMAGE=intelanalytics/bigdl-k8s:latest

sudo docker run -itd \
    --privileged \
    --net=host \
    -v $KUBECONFIG_PATH:/root/.kube/config \
    -v $KEYS_PATH:/opt/spark/work-dir/keys \
    -v $SECURE_PASSWORD_PATH:/opt/spark/work-dir/password \
    --name=spark-k8s-client \
    -e LOCAL_IP=$LOCAL_IP \
    -e RUNTIME_SPARK_MASTER=$K8S_MASTER \
    -e RUNTIME_K8S_SERVICE_ACCOUNT=spark \
    -e RUNTIME_K8S_SPARK_IMAGE=$DOCKER_IMAGE \
    $DOCKER_IMAGE bash
```
Run `docker exec -it spark-local-client bash` to entry the container.

### 2. Run application in spark K8S mode
#### 2.1 Run application in K8S client mode
The example for run Spark Pi:
```bash
spark-submit-with-ppml.sh 
```
#### 2.2 Run application in K8s cluster mode
The example for run Spark Pi:
```bash
spark-submit-with-ppml.sh
```
