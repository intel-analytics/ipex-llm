# Trusted Realtime ML for Kubernetes

## Deploy the Intel SGX Device Plugin for Kubenetes

The instructions in this section are modified from the [Intel SGX Device Plugin homepage][intelSGX], to which please refer should questions arise.

### Prerequisites
Prerequisites for building and running these device plugins include:
- Appropriate hardware
- A fully configured Kubernetes cluster
- A working Go environment, of at least version v1.16

Here we would want to deploy the plugin as a DaemonSet, so pull the [source code][pluginCode]. In the working directory, compile with 
``` bash
make intel-sgx-plugin
make intel-sgx-initcontainer
```
Deploy the DaemonSet with
```bash
kubectl apply -k /usr/local/go-1.16.6/src/intel/sgx-device-plugin/deployments/sgx_plugin/overlays/epc-register/
```
Verify with (replace the `<node name>` with your own node name)
```
kubectl describe node <node name> | grep sgx.intel.com
```
## Deploying Trusted Realtime ML for Kubernetes

### Configurables

The file `templates/flink-configuration-configmap.yaml` contains the configurable parameters of the deployments. Most of the parameters have self-explanatory names. 
You can configure these at will, but it is adviced to keep `flink.jobmanager.ip` as it is.
It is worth mentioning that you can run the components without using sgx by setting the value of `sgx.mode` to `no_sgx`. 

### Secure keys and password 

You need to [generate secure keys and password][keysNpassword]. Modify the `OUTPUT` in both `../../../../scripts/generate-keys.sh` and `../../../../scripts/generate-password.sh` to your present working directory, and run both scripts. Then, run
``` bash
kubectl apply -f keys.yaml
kubectl apply -f password.yaml
```

### Using [Helm][helmsite] to deploy all components

If you have installed Helm, you can use Helm to deploy all the `yaml` files at once. In `values.yaml`, configure the full paths for `start-all-but-flink.sh` and `enclave-key.pem`. 
Then, simply run 
``` bash
helm install <name> ./
```
where `<name>` is a name you give for this installation. 

### Deploying everything by hand

Alternatively, you can also deploy everything one by one. All of the following `yaml` files are in `templates`.

#### Deploy the Flink job manager and task manager
In `jobmanager-session-deployment.yaml` and `taskmanager-session-deployment.yaml`, look for `{{ .Values.enclaveKeysPath }}`, and configure the paths accordingly. 
Then, run
```bash
# Configuration and service definition
kubectl create -f flink-configuration-configmap.yaml
kubectl create -f jobmanager-service.yaml
# Create the deployments for the cluster
kubectl create -f jobmanager-session-deployment.yaml
kubectl create -f taskmanager-session-deployment.yaml
```

Both the job manager and the task manager will start automatically in SGX on top of Graphene libos when the deployments are created.


#### Deploy cluster serving
In `master-deployment.yaml`, look for `{{ .Values.startAllButFlinkPath }}` and `{{ .Values.enclaveKeysPath }}`, and configure these two paths accordingly. 

Finally, run 
```bash
kubectl apply -f master-deployment.yaml
```
The components (Redis, http-frontend, and cluster serving) should start on their own, if jobmanager and taskmanager are running.

### Port forwarding

You can set up port forwarding to access the containers' ports on the host.
Taking jobmanager’s web ui port `8081` as an example:
1. Run `kubectl port-forward ${flink-jobmanager-pod} --address 0.0.0.0 8081:8081` to forward your jobmanager’s web ui port to local 8081.
2. Navigate to http://localhost:8081 in your browser.

The same goes for the master deployment's ports 6379, 10020, and 10023. Remember to change the name of the pod to that of the master pod.

[intelSGX]: https://intel.github.io/intel-device-plugins-for-kubernetes/cmd/sgx_plugin/README.html
[pluginCode]: https://github.com/intel/intel-device-plugins-for-kubernetes
[keysNpassword]: https://github.com/intel-analytics/analytics-zoo/tree/master/ppml/trusted-realtime-ml/scala/docker-graphene#prepare-the-keys
[helmsite]: https://helm.sh/
