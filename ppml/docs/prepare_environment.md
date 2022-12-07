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

2. Generate k8s config file, modify `YOUR_DIR` to the location you want to store the config:

    ```bash
    kubectl config use-context spark-context
    kubectl config view --flatten --minify > /YOUR_DIR/kubeconfig
    ```


### Key Management Service (KMS) and Attestation Service (AS) Setup
Key Management Service (KMS) helps you manage cryptographic keys for your services. In BigDL PPML end-to-end workflow, KMS is used to generate keys, encrypt the input data and decrypt the result of Big Data & AI applications. Attestation Service (AS) makes sure applications of the customer actually run in the SGX MREnclave signed above by customer-self, rather than a fake one fake by an attacker. You can choose to use the KMS and AS which PPML provides or your own one.

BigDL PPML use EHSM as the reference type of KMS & AS, follow the document to deploy EHSM: https://github.com/intel-analytics/BigDL/blob/main/ppml/services/ehsm/kubernetes/README.md

### Prepare BigDL PPML Docker Image

Pull Docker image from Dockerhub
    ```
    docker pull intelanalytics/bigdl-ppml-trusted-big-data-ml-python-gramine-reference:2.2.0-SNAPSHOT
    ```

**Note:** The above docker image `intelanalytics/bigdl-ppml-trusted-big-data-ml-python-gramine-reference:2.2.0-SNAPSHOT` is only used for demo purposes. You are recommended to refer to the [Prepare your PPML image for production environment](./../README.md#step-1-prepare-your-ppml-image-for-production-environment) to use BigDL PPML image as a base image to build your own image and sign your image with your own enclave_key.

### (Optional) K8s Monitioring Setup
https://github.com/analytics-zoo/ppml-e2e-examples/blob/main/bigdl-ppml-sgx-k8s-prometheus/README.md
