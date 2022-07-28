Prior to run your Big Data & AI applications with BigDL PPML, please make sure the following is setup

* Hardware that supports SGX [(3rd Gen Intel Xeon Scalable Processors)](https://www.intel.com/content/www/us/en/products/docs/processors/xeon/3rd-gen-xeon-scalable-processors-brief.html)
* A fully configured Kubernetes cluster [(Production Cluster Setup)](https://kubernetes.io/docs/setup/production-environment/#production-cluster-setup)
* [Intel SGX Device Plugin](https://bigdl.readthedocs.io/en/latest/doc/PPML/QuickStart/deploy_intel_sgx_device_plugin_for_kubernetes.html) to use SGX in K8S cluster
* [Prepare Key and Password](#prepare-key-and-password)
* [Configure the Environment](#configure-the-environment)
* [Key Management Service (KMS) Setup](#key-management-service-kms-setup)
* [Attestation Service (AS) Setup](#attestation-service-as-setup)
* [BigDL PPML Client Container](#start-bigdl-ppml-client-container)
* (Optional) [K8s Monitioring](#optional-k8s-monitioring-setup)

### Prepare Key and Password
Download scripts from [here](https://github.com/intel-analytics/BigDL).

```
cd BigDL/ppml/
```

* **Prepare the ssl_key and ssl_password**
  
  Note: Make sure to add `${JAVA_HOME}/bin` to `$PATH` to avoid `keytool: command not found error`.

  Run the script to generate **keys/keys.yaml** and **password/password.yaml**
  ```bash
  sudo bash scripts/generate-keys.sh
  sudo bash scripts/generate-password.sh YOUR_PASSWORD
  ```

  Deploy **keys/keys.yaml** and **password/password.yaml** as secrets for Kubernetes
  ```
  kubectl apply -f keys/keys.yaml
  kubectl apply -f password/password.yaml
  ```
  Then two secrets **ssl_keys** and **ssl_password** should be listed in `kubectl get secret`



* **Prepare the enclave key**

  **enclave key** is the signing key for SGX Enclaves.
  
  Run the script to generate your enclave key and add it to your Kubernetes cluster as a secret.
  ```
  cd kubernetes
  bash enclave-key-to-secret.sh
  ```
  Then the secret **enclave_key** should be listed in `kubectl get secret`
  
* **Prepare k8s secret**

  The secret created `YOUR_PASSWORD` should be the same as the password you specified in step 1:

   ```bash
   kubectl create secret generic spark-secret --from-literal secret=YOUR_PASSWORD
   ```
   Then the secret **spark-secret** should be listed in `kubectl get secret`
   

>**Caution**: 
>
>It is important to protect your keys and passwords. The above steps for preparing keys and passwords are only for demo purposes, it is non-production. You are recommended to generate your keys and passwords according to your requirements and protect them safely.
>
>Besides, Kubernetes Secrets are, by default, stored unencrypted in the API server's underlying data store (etcd). Anyone with API access can retrieve or modify a Secret, and so can anyone with access to etcd. Additionally, anyone who is authorized to create a Pod in a namespace can use that access to read any Secret in that namespace; this includes indirect access such as the ability to create a Deployment. In order to safely use Secrets, take the steps in [safely use secrets](https://kubernetes.io/docs/concepts/configuration/secret/)

### Configure the Environment

1. Create the [RBAC(Role-based access control)](https://spark.apache.org/docs/latest/running-on-kubernetes.html#rbac) :

    ```bash
    kubectl create serviceaccount spark
    kubectl create clusterrolebinding spark-role --clusterrole=edit --serviceaccount=default:spark --namespace=default
    ```

2. Generate k8s config file, modify `YOUR_DIR` to the location you want to store the config:

    ```bash
    kubectl config view --flatten --minify > /YOUR_DIR/kubeconfig
    ```


### Key Management Service (KMS) Setup
Key Management Service (KMS) helps you manage cryptographic keys for your services. In BigDL PPML end-to-end workflow, KMS is used to generate keys, encrypt the input data and decrypt the result of Big Data & AI applications. You can choose to use the KMS service which PPML provides or your own one.

To use the KMS service in PPML, follow the document: https://github.com/intel-analytics/BigDL/blob/main/ppml/services/pccs-ehsm/kubernetes/README.md

### Attestation Service (AS) Setup
placeholder

### Start BigDL PPML Client Container
1. Prepare BigDL PPML Docker Image

    Pull Docker image from Dockerhub
    ```
    docker pull intelanalytics/bigdl-ppml-trusted-big-data-ml-scala-graphene:devel
    ```

    Alternatively, you can build Docker image from Dockerfile (this will take some time):
    ```
    cd trusted-big-data-ml/python/docker-graphene
    ./build-docker-image.sh
    ```
    
    **Note:** The above docker image `intelanalytics/bigdl-ppml-trusted-big-data-ml-scala-graphene:devel` is only used for demo purposes. You are recommended to refer to the [BigDL PPML Dockerfile](https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/python/docker-graphene/Dockerfile) or use BigDL PPML image as a base image to build your own image and sign your image with your own enclave_key.
    
2. Start BigDL PPML client container
    
    Configure the environment variables in the following script before running it.
    ```
    export K8S_MASTER=k8s://$(sudo kubectl cluster-info | grep 'https.*6443' -o -m 1)
    echo The k8s master is $K8S_MASTER .
    export ENCLAVE_KEY=/YOUR_DIR/enclave-key.pem
    export DATA_PATH=/YOUR_DIR/data
    export KEYS_PATH=/YOUR_DIR/keys
    export SECURE_PASSWORD_PATH=/YOUR_DIR/password
    export KUBECONFIG_PATH=/YOUR_DIR/kubeconfig
    export LOCAL_IP=$LOCAL_IP
    export DOCKER_IMAGE=intelanalytics/bigdl-ppml-trusted-big-data-ml-python-graphene:devel
    sudo docker run -itd \
        --privileged \
        --net=host \
        --name=bigdl-ppml-client-k8s \
        --cpuset-cpus="0-4" \
        --oom-kill-disable \
        --device=/dev/sgx/enclave \
        --device=/dev/sgx/provision \
        -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
        -v $ENCLAVE_KEY:/graphene/Pal/src/host/Linux-SGX/signer/enclave-key.pem \
        -v $DATA_PATH:/ppml/trusted-big-data-ml/work/data \
        -v $KEYS_PATH:/ppml/trusted-big-data-ml/work/keys \
        -v $SECURE_PASSWORD_PATH:/ppml/trusted-big-data-ml/work/password \
        -v $KUBECONFIG_PATH:/root/.kube/config \
        -e RUNTIME_SPARK_MASTER=$K8S_MASTER \
        -e RUNTIME_K8S_SERVICE_ACCOUNT=spark \
        -e RUNTIME_K8S_SPARK_IMAGE=$DOCKER_IMAGE \
        -e RUNTIME_DRIVER_HOST=$LOCAL_IP \
        -e RUNTIME_DRIVER_PORT=54321 \
        -e RUNTIME_DRIVER_CORES=1 \
        -e RUNTIME_EXECUTOR_INSTANCES=1 \
        -e RUNTIME_EXECUTOR_CORES=8 \
        -e RUNTIME_EXECUTOR_MEMORY=1g \
        -e RUNTIME_DRIVER_CORES=4 \
        -e RUNTIME_DRIVER_MEMORY=1g \
        -e SGX_DRIVER_MEM=32g \
        -e SGX_DRIVER_JVM_MEM=8g \
        -e SGX_EXECUTOR_MEM=32g \
        -e SGX_EXECUTOR_JVM_MEM=12g \
        -e SGX_ENABLED=true \
        -e SGX_LOG_LEVEL=error \
        -e SPARK_MODE=client \
        -e LOCAL_IP=$LOCAL_IP \
        $DOCKER_IMAGE bash
    ```

### (Optional) K8s Monitioring Setup
https://github.com/analytics-zoo/ppml-e2e-examples/blob/main/bigdl-ppml-sgx-k8s-prometheus/README.md
