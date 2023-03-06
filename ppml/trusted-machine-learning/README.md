# Gramine Machine Learning Toolkit

This image contains Gramine and some popular Machine Learning frameworks including Spark and LightGBM. 

## Before Running Code
### 1. Build Docker Images
#### 1.1 Build Machine Learning Base Image

The machine learning base image is a public one that does not contain any secrets. You will use the base image to get your own custom image in the following.

Before building your own base image, please modify the paths in `build-base-image.sh`. Then build the docker image with the following command.

```bash
./build-machine-learning-base-image.sh
```
#### 1.2 Build Custom Image

Follow [here](https://github.com/intel-analytics/BigDL/tree/main/ppml/trusted-bigdata#12-build-customer-image) to build a custom image with enclave signed by your private key.

### 2. Prepare SSL key and password

Follow [here](https://github.com/intel-analytics/BigDL/blob/main/ppml/docs/prepare_environment.md#prepare-key-and-password) to prepare SSL key and password for secure container communication.

## Run machine learning example

### 1. Configure K8S Environment

Follow [here](https://github.com/intel-analytics/BigDL/blob/main/ppml/docs/prepare_environment.md#configure-the-environment) to create and configure K8S RBAC and secrets.

### 2. Start the client container

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
run `docker exec -it machine-learning-graming bash` to enter the container.

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

