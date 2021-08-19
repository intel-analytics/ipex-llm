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

## Deploy the Flink job manager and task manager
You need to [generate secure keys and password][keysNpassword]. Modify the `OUTPUT` in both `../../../../scripts/generate-keys.sh` and `../../../../scripts/generate-password.sh` to your present working directory, and run both scripts. Then, run
``` bash
kubectl apply -f keys.yaml
kubectl apply -f password.yaml
```
In `jobmanager-session-deployment.yaml` and `taskmanager-session-deployment.yaml`, look for `path_to_enclave-key.pem`, and configure the paths accordingly. 

```bash
# Configuration and service definition
kubectl create -f flink-configuration-configmap.yaml
kubectl create -f jobmanager-service.yaml
# Create the deployments for the cluster
kubectl create -f jobmanager-session-deployment.yaml
kubectl create -f taskmanager-session-deployment.yaml
```

Both the job manager and the task manager will start automatically in SGX on top of Graphene libos when the deployments are created.

Next, we set up a port forward to access the Flink UI:
1. Run `kubectl port-forward ${flink-jobmanager-pod} 8081:8081` to forward your jobmanager’s web ui port to local 8081.
2. Navigate to http://localhost:8081 in your browser.

## Deploy cluster serving
In `master-deployment.yaml`, look for `path_to_start-all-but-flink.sh` and `path_to_enclave-key.pem`, and configure these two paths accordingly. 

Finally, run 
```bash
kubectl apply -f master-deployment.yaml
```
The components (Redis, http-frontend, and cluster serving) should start on their own, if jobmanager and taskmanager are running.

Next, we set up a port forward to access the ports 6379, 10020, and 10023 on the host:
1. Run `kubectl port-forward ${master-pod} 10020:10020`.
2. Navigate to http://localhost:10020 in your browser.
3. Do the same to the other ports.

[intelSGX]: https://intel.github.io/intel-device-plugins-for-kubernetes/cmd/sgx_plugin/README.html
[pluginCode]: https://github.com/intel/intel-device-plugins-for-kubernetes
[keysNpassword]: https://github.com/intel-analytics/analytics-zoo/tree/master/ppml/trusted-realtime-ml/scala/docker-graphene#prepare-the-keys
