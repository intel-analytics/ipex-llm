# BigDL PPML FL Demo for Kubernetes with Helm Charts

## 1 Deploy the Intel SGX Device Plugin for Kubenetes

Please refer to the document [here](https://bigdl.readthedocs.io/en/latest/doc/PPML/QuickStart/deploy_intel_sgx_device_plugin_for_kubernetes.html).

## 2 Prepare the keys
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


## 3 Deploy FL Demo for Kubernetes

### 3.1 Helm install
Using helm to deploy fl-demo. Run:
```commandline
helm install <name> ./bigdl-ppml-fl-helm
```

### 3.2 Debugging
To check the logs of the Hfl server, run
```commandline
sudo kubectl logs $( sudo kubectl get pod | grep fl-server | cut -d " " -f1 ) hflserver -f
```

To check the logs of the Hfl client1, run
```commandline
sudo kubectl logs $( sudo kubectl get pod | grep fl-client | cut -d " " -f1 ) hflclient1 -f
```

To check the logs of the Hfl client2, run
```commandline
sudo kubectl logs $( sudo kubectl get pod | grep fl-client | cut -d " " -f1 ) hflclient2 -f
```

To check the logs of the Vfl server, run
```commandline
sudo kubectl logs $( sudo kubectl get pod | grep fl-server | cut -d " " -f1 ) vflserver -f
```

To check the logs of the Vfl client1, run
```commandline
sudo kubectl logs $( sudo kubectl get pod | grep fl-client | cut -d " " -f1 ) vflclient1 -f
```

To check the logs of the Vfl client2, run
```commandline
sudo kubectl logs $( sudo kubectl get pod | grep fl-client | cut -d " " -f1 ) vflclient2 -f

### 3.3 Helm Uninstall
To uninstall the helm chart, run
```commandline
helm uninstall <name>
```

Note that the <name> must be the same as the one you set in section 3.1.
