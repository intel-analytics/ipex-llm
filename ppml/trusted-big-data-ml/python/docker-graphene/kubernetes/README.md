# Trusted big data ML for Kubernetes with Helm Charts

## 1 Deploy the Intel SGX Device Plugin for Kubenetes

Please refer to the document [here][devicePluginK8sQuickStart].

## 2 Deploy Trusted Realtime ML for Kubernetes

### 2.1 Configurables (This part is going under changes)

In `bigdl-ppml-helm/values.yaml`, configure the full values for all items listed: 
- `enclaveKeysPath`: Generate your enclave key with `openssl genrsa -3 -out enclave-key.pem 3072`, and provide the full path to the enclave key here. 
- `dataPath`: Provide the full path to your data on your host machine.
- `kubeconfigPath`: Generate your Kubernetes config file with `kubectl config view --flatten --minify > kubeconfig`, and provide the full path to the kubeconfig file here. 
- `image`: The PPML image you want to use.
- `k8sMaster`: Run `kubectl cluster-info`. The output should be like `Kubernetes control plane is running at https://master_ip:master_port`. Fill in the master ip and port.
- `pvc`: The name of the Persistent Volume Claim (PVC) of your Network File System (NFS). We assume you have a working NFS configured for your Kubernetes cluster. Please also put the `enclave-key.pem` and `kubeconfig` files as well as the script used to submit your Spark job (defaulted to `submit-spark-k8s.sh`) in the NFS so they can be discovered by all the nodes.

### 2.2 Secure keys and password 

You need to [generate secure keys and password][keysNpassword]. Run
``` bash
bash ../../../../scripts/generate-keys.sh
bash ../../../../scripts/generate-password.sh YOUR_PASSWORD
kubectl apply -f keys/keys.yaml
kubectl apply -f password/password.yaml
```

### 2.3 Create the RBAC
```bash
kubectl create serviceaccount spark
kubectl create clusterrolebinding spark-role --clusterrole=edit --serviceaccount=default:spark --namespace=default
```

### 2.4 Create k8s secret 

``` bash
kubectl create secret generic spark-secret --from-literal secret=YOUR_SECRET
```
**The secret created should be the same as `YOUR_PASSWORD` in section 2.2**. 

### 2.5 Using [Helm][helmsite] to run your Spark job

You can use Helm to deploy your Spark job. Simply run 
``` bash
helm install <name> ./bigdl-ppml-helm
```
where `<name>` is a name you give for this installation. 

### 2.6 Debugging

To check the logs of the driver, run
``` bash
sudo kubectl logs $( sudo kubectl get pod | grep spark-pi-job | cut -d " " -f1 )
```

To check the logs of an executor, run
``` bash 
sudo kubectl logs $( sudo kubectl get pod | grep "spark-pi-.*-exec" -m 1 | cut -d " " -f1 )
```

### 2.7 Deleting the Job

To uninstall the helm chart, run
``` bash
helm uninstall <name>
```

Note that the `<name>` must be the same as the one you set in section 2.5. Helm does not delete the executors that are run by the driver, so for now we can only delete them manually. 

[devicePluginK8sQuickStart]: https://bigdl.readthedocs.io/en/latest/doc/PPML/QuickStart/deploy_intel_sgx_device_plugin_for_kubernetes.html
[keysNpassword]: https://github.com/intel-analytics/BigDL/tree/main/ppml/trusted-big-data-ml/python/docker-graphene#2-prepare-data-key-and-password
[helmsite]: https://helm.sh/
