# Deploy the Intel SGX Device Plugin for Kubenetes

The instructions in this section are modified from the [Intel SGX Device Plugin homepage][intelSGX], to which please refer should questions arise.

## Prerequisites
Prerequisites for building and running these device plugins include:
- Appropriate hardware. ([3rd Gen Intel Xeon Scalable Processors][GIXSP])
- A fully configured Kubernetes cluster
- A working Go environment, of at least version v1.16

Here we would want to deploy the plugin as a DaemonSet, so pull the [source code][pluginCode]. In the working directory, compile with 
``` bash
make intel-sgx-plugin
make intel-sgx-initcontainer
```
Deploy the DaemonSet with
```bash
kubectl apply -k deployments/sgx_plugin/overlays/epc-register/
```
Verify with (replace the `<node name>` with your own node name)
```
kubectl describe node <node name> | grep sgx.intel.com
```

[intelSGX]: https://intel.github.io/intel-device-plugins-for-kubernetes/cmd/sgx_plugin/README.html
[GIXSP]: https://www.intel.com/content/www/us/en/products/docs/processors/xeon/3rd-gen-xeon-scalable-processors-brief.html
[pluginCode]: https://github.com/intel/intel-device-plugins-for-kubernetes
