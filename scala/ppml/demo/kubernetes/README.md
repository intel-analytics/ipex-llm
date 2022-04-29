# BigDL PPML FL Demo for Kubernetes with Helm Charts

##1 Deploy the Intel SGX Device Plugin for Kubenetes

Please refer to the document [here](https://bigdl.readthedocs.io/en/latest/doc/PPML/QuickStart/deploy_intel_sgx_device_plugin_for_kubernetes.html).

## 2 Deploy FL Demo for Kubernetes

### 2.1 Secure keys and the enclave key
You need to [generate secure keys](https://github.com/intel-analytics/BigDL/tree/main/scala/ppml/demo#prepare-the-key). Run:
```commandline
sudo bash ../../../../ppml/scripts/generate-keys.sh
kubectl apply -f keys/keys.yaml
```
Run 
```
bash enclave-key-to-secret.sh
```
to generate your enclave key and add it to your Kubernetes cluster as a secret.

### 2.2 Helm Install
Using helm to deploy fl-demo. Run:
```commandline
helm install <name> ./bigdl-ppml-fl-helm
```

### 2.3 Debugging
To check the logs of the Fl server, run
```commandline
sudo kubectl logs $( sudo kubectl get pod | grep fl-server | cut -d " " -f1 ) -f
```

To check the logs of the Hfl client1, run
```commandline
sudo kubectl logs $( sudo kubectl get pod | grep hfl-client1 | cut -d " " -f1 ) -f
```

To check the logs of the Hfl client2, run
```commandline
sudo kubectl logs $( sudo kubectl get pod | grep hfl-client2 | cut -d " " -f1 ) -f
```

### 2.4 Helm Uninstall
To uninstall the helm chart, run
```commandline
helm uninstall <name>
```

Note that the <name> must be the same as the one you set in section 2.2.