Prior to run your Big Data & AI applications with BigDL PPML, please make sure the following is setup

* Hardware that supports SGX
* A fully configured Kubernetes cluster
* [Intel SGX Device Plugin](https://bigdl.readthedocs.io/en/latest/doc/PPML/QuickStart/deploy_intel_sgx_device_plugin_for_kubernetes.html) to use SGX in K8S cluster
* [Data, Key and Password Preparation](#prepare-data-key-and-password)
* [Configure the Environment](#configure-the-environment)
* [KMS Service Setup](kms-key-management-service-setup)
* [Attestation Service Setup](#attestation-service-setup)
* [BigDL PPML Docker Image](#prepare-bigdl-ppml-docker-image)
* (Optional) [K8s Monitioring](#optional-k8s-monitioring-setup)

### Prepare data, key and password
##### Prepare the Data
To run Big Data & AI applications with ppml in BigDL, you need to prepare the data first. 

##### Prepare the Key

  * The ppml in bigdl needs secured keys to enable spark security such as Authentication, RPC Encryption, Local Storage Encryption and TLS, you need to prepare the **secure keys and keystores**. In this tutorial, you can generate keys and keystores with root permission (test only, need input security password for keys).

      ```bash
      sudo bash ../../../scripts/generate-keys.sh
      ```

  * You also need to generate your **enclave key** using the command below, and keep it safely for future remote attestations and to start SGX enclaves more securely.

      It will generate a file `enclave-key.pem` in your present working directory, which will be your enclave key. To store the key elsewhere, modify the outputted file path.

      ```bash
      openssl genrsa -3 -out enclave-key.pem 3072
      ```

##### Prepare the Password

  Next, you need to store the password you used for key generation, i.e., `generate-keys.sh`, in a secured file.

  ```bash
  sudo bash ../../../scripts/generate-password.sh used_password_when_generate_keys
  ```

### Configure the Environment

1. Enter `BigDL/ppml/trusted-big-data-ml/python/docker-graphene` dir. Refer to the previous section about [preparing data, key and password](#prepare-data-key-and-password). Then run the following commands to generate your enclave key and add it to your Kubernetes cluster as a secret. 

    ```bash
    kubectl apply -f keys/keys.yaml
    kubectl apply -f password/password.yaml
    cd kubernetes
    bash enclave-key-to-secret.sh
    ```
2. Create the [RBAC(Role-based access control)](https://spark.apache.org/docs/latest/running-on-kubernetes.html#rbac) :

    ```bash
    kubectl create serviceaccount spark
    kubectl create clusterrolebinding spark-role --clusterrole=edit --serviceaccount=default:spark --namespace=default
    ```

3. Generate k8s config file, modify `YOUR_DIR` to the location you want to store the config:

    ```bash
    kubectl config view --flatten --minify > /YOUR_DIR/kubeconfig
    ```
4. Create k8s secret, the secret created `YOUR_SECRET` should be the same as the password you specified in step 1:

    ```bash
    kubectl create secret generic spark-secret --from-literal secret=YOUR_SECRET
    ```

### KMS (key management service) Setup
You can choose to use the KMS service which PPML provides or you own one.
To use the KMS service in PPML, deploy kms first: https://github.com/intel-analytics/BigDL/blob/main/ppml/services/pccs-ehsm/kubernetes/README.md

### Attestation Service Setup
placeholder





### Prepare BigDL PPML Docker Image

Pull Docker image from Dockerhub
```
docker pull intelanalytics/bigdl-ppml-trusted-big-data-ml-scala-graphene:2.1.0-SNAPSHOT
```
Alternatively, you can build Docker image from Dockerfile (this will take some time):
```
cd trusted-big-data-ml/python/docker-graphene
./build-docker-image.sh
```

### (Optional) K8s Monitioring Setup
https://github.com/analytics-zoo/ppml-e2e-examples/blob/main/bigdl-ppml-sgx-k8s-prometheus/README.md
