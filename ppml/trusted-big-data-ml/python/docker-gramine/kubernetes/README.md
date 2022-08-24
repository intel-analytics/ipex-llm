# Trusted big data ML for Kubernetes with Helm Charts

## 1 Deploy the Intel SGX Device Plugin for Kubenetes

Please refer to the document [here][devicePluginK8sQuickStart].

## 2 Deploy Trusted Realtime ML for Kubernetes

### 2.1 Configurables

In `bigdl-ppml-helm/values.yaml`, configure the full values for: 
- `image`: The PPML image you want to use.
- `k8sMaster`: Run `kubectl cluster-info`. The output should be like `Kubernetes control plane is running at https://master_ip:master_port`. Fill in the master ip and port.
- `pvc`: The name of the Persistent Volume Claim (PVC) of your Network File System (NFS). We assume you have a working NFS configured for your Kubernetes cluster. 
- `jar`: The `jar` file you would like Spark to run, defaulted to `spark-examples_2.12-3.1.2.jar`. The path should be the path in the container defined in `bigdl-ppml-helm/templates/spark-job.yaml`
- `class`: The `class` you would like Spark to run, defaulted to `org.apache.spark.examples.SparkPi`.

Please prepare the following and put them in your NFS directory:
- The data (in a directory called `data`), 
- A kubeconfig file. Generate your Kubernetes config file with `kubectl config view --flatten --minify > kubeconfig`, then put it in your NFS.

The other values have self-explanatory names and can be left alone.

### 2.2 Secure keys, password, and the enclave key

You need to [generate secure keys and password][keysNpassword]. Run
``` bash
bash ../../../../scripts/generate-keys.sh
bash ../../../../scripts/generate-password.sh YOUR_PASSWORD
kubectl apply -f keys/keys.yaml
kubectl apply -f password/password.yaml
```

Run `bash enclave-key-to-secret.sh` to generate your enclave key and add it to your Kubernetes cluster as a secret.

### 2.3 Create the RBAC
```bash
sudo kubectl create serviceaccount spark
sudo kubectl create clusterrolebinding spark-role --clusterrole=edit --serviceaccount=default:spark --namespace=default
```

### 2.4 Create k8s secret

``` bash
sudo kubectl create secret generic spark-secret --from-literal secret=YOUR_SECRET
```

**The secret created (`YOUR_SECRET`) should be the same as `YOUR_PASSWORD` in section 2.2**.

### 2.5 Using [Helm][helmsite] to run your Spark job

You can use Helm to deploy your Spark job. Simply run 
``` bash
helm install <name> ./bigdl-ppml-helm
```
where `<name>` is a name you give for this installation. 

### 2.6 Debugging

To check the logs of the Kubernetes job, run
``` bash
sudo kubectl logs $( sudo kubectl get pod | grep spark-pi-job | cut -d " " -f1 )
```

To check the logs of the Spark driver, run
``` bash
sudo kubectl logs $( sudo kubectl get pod | grep "spark-pi-sgx.*-driver" -m 1 | cut -d " " -f1 )
```

To check the logs of an Spark executor, run
``` bash 
sudo kubectl logs $( sudo kubectl get pod | grep "spark-pi-.*-exec" -m 1 | cut -d " " -f1 )
```

### 2.7 Deleting the Job

To uninstall the helm chart, run
``` bash
helm uninstall <name>
```

Note that the `<name>` must be the same as the one you set in section 2.5. Helm does not delete the driver and executors that are run by the Kubernetes Job, so for now we can only delete them manually: 
``` bash
sudo kubectl get pod | grep -o "spark-pi-.*-exec-[0-9]*" | xargs sudo kubectl delete pod
sudo kubectl get pod | grep -o "spark-pi-sgx.*-driver" | xargs sudo kubectl delete pod
```

## 3 Attestation

With attestation, we can verify if any service is replaced or hacked by malicious nodes. This helps us ensure integrity of the our distributed applications.

### 3.1 Prerequisites

To enable attestation in BigDL PPML, you need to ensure you have correct access to attestation services (eHSM attestation service, amber or Azure attestation service etc). In this example, we will sue eHSM as attestation service. Please ensure eHSM is correctly configured.

### 3.2 Attestation Configurations

1. Set APP_ID and APP_KEY in [kms-secret.yaml](https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/python/docker-gramine/kubernetes/kms-secret.yaml). Apply this secret.
2. Mount APP_ID and APP_KEY in [spark-driver-template.yaml](https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/python/docker-gramine/spark-driver-template.yaml#L13) and [spark-executor-template.yaml](https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/python/docker-gramine/spark-executor-template.yaml#L13).
3. Change ATTESTATION to `true` in [spark-driver-template.yaml](https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/python/docker-gramine/spark-driver-template.yaml#L10) and [spark-executor-template.yaml](https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/python/docker-gramine/spark-executor-template.yaml#L10), and set ATTESTATION_URL, e.g., `http://192.168.0.8:9000`.

### 3.2 Test with examples

After updating `spark-driver-template.yaml` and `spark-executor-template.yaml`, attestation will by automatically added to BigDL PPML pipe line. That means PPML applications will be automatically attested by attestation service when they start in Kubernetes Pod. They will prevent malicious Pod from getting sensitive information in applications.

You can test attestation with [Spark Pi](https://github.com/intel-analytics/BigDL/tree/main/ppml/trusted-big-data-ml/python/docker-gramine#143-spark-pi-example) or other Kubernetes examples.


[devicePluginK8sQuickStart]: https://bigdl.readthedocs.io/en/latest/doc/PPML/QuickStart/deploy_intel_sgx_device_plugin_for_kubernetes.html
[keysNpassword]: https://github.com/intel-analytics/BigDL/tree/main/ppml/trusted-big-data-ml/python/docker-gramine#2-prepare-data-key-and-password
[helmsite]: https://helm.sh/
