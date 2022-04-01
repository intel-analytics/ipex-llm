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
- The script used to submit your Spark job (defaulted to `./submit-spark-k8s.sh`) 
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


[devicePluginK8sQuickStart]: https://bigdl.readthedocs.io/en/latest/doc/PPML/QuickStart/deploy_intel_sgx_device_plugin_for_kubernetes.html
[keysNpassword]: https://github.com/intel-analytics/BigDL/tree/main/ppml/trusted-big-data-ml/python/docker-graphene#2-prepare-data-key-and-password
[helmsite]: https://helm.sh/
