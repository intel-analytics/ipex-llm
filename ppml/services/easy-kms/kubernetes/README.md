# Deploy Easy KMS (Key Management Service) on Kubernetes

## Easy KMS Architecture
![EasyKMS](https://user-images.githubusercontent.com/60865256/229735029-b93f221a-7973-49fa-9474-a216121caf18.png)

Easy KMS applies multiple security techniques e.g. trusted execution environment (SGX/TDX), AES and TLS/SSL to ensure end-to-end secure key management.

## Prerequests

- Make sure you have a workable **Kubernetes cluster/machine**
- Prepare [easy-kms docker image](https://github.com/intel-analytics/BigDL/tree/main/ppml/services/easy-kms/docker#pullbuild-container-image). **Note** that if enable SGX, please use a custom-signed image.

## Start BigDL KMS on Kubernetes
### 1. Prepare SSL Key and Password
Follow [here](https://github.com/intel-analytics/BigDL/blob/main/ppml/docs/prepare_environment.md#prepare-key-and-password) to generate keys and passwords for TLS encryption, and upload them to k8s as secrets.

### 2. Run Install Script
Modify parameters in script `install-easy-kms.sh`:

```
......
imageName       ---> if enable SGX, replace with custom image
dataStoragePath ---> a host path for persistent stoarge
serviceIP       ---> your key management service ip to expose
rootKey         ---> your 256bit base64 AES key string
sgxEnabled      ---> true for SGX user, and false for TDX user
......
```

The `rootKey` can be created in this way:
```bash
openssl enc -aes-256-cbc -k secret -P -md sha1
# you will get a key, and copy it to below field
echo <key_generated_above> | base64
```

Then, deploy easy-kms on kubernetes by one command:

```bash
bash install-easy-kms.sh
```

### 3. Check Delopyment

Check the service whether it has successfully been running (it may take seconds).
```bash
kubectl get all | grep easy
```

It is expected to get outputs similar to the following:
```bash
pod/easy-key-management-server-f4985d65f-2qctz    1/1            Running
service/easy-key-management-service               LoadBalancer   10.99.15.77     <serviceIP>    9875:31157/TCP
deployment.apps/easy-key-management-server        1/1            1               1
replicaset.apps/easy-key-management-server-f4985d65f
```

## REST APIs of Key Management

Easy KMS has the [same APIs](https://github.com/intel-analytics/BigDL/tree/main/ppml/services/bigdl-kms/kubernetes#validate-status-of-bigdl-kms) to `bigdl-kms`, while `kmsIP` is the above `serviceIP` and default port number of Easy KMS is `9875`.


## Stop Easy KMS
```
kubectl delete -f easy-kms.yaml
```
