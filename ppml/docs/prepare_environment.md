Prior to run your with BigDL PPML, please make sure the following is setup

* Hardware that supports SGX
* A fully configured Kubernetes cluster
* [Intel SGX Device Plugin](https://bigdl.readthedocs.io/en/latest/doc/PPML/QuickStart/deploy_intel_sgx_device_plugin_for_kubernetes.html) to use SGX in K8S cluster
* [Data, Key and Password Preparation](#prepare-data-key-and-password)
* [KMS Service Setup](kms-key-management-service-setup)
* [Attestation Service Setup](#attestation-service-setup)
* [BigDL PPML Docker Image](#prepare-bigdl-ppml-docker-image)
* (Optional) [K8s Monitioring](#optional-k8s-monitioring-setup)

### Prepare data, key and password
##### Prepare the Data
To train a model with ppml in BigDL, you need to prepare the data first. 

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
