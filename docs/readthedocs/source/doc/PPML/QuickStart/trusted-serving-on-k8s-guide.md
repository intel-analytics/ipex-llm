# Trusted Cluster Serving with Graphene on Kubernetes #

## Prerequisites ##
Prior to deploying PPML Cluster Serving, please make sure the following is setup
- Hardware that supports SGX
- A fully configured Kubernetes cluster 
- Intel SGX Device Plugin to use SGX in K8S cluster (install following instructions [here](https://github.com/intel-analytics/BigDL/tree/branch-2.0/ppml/trusted-realtime-ml/scala/docker-graphene/kubernetes#deploy-the-intel-sgx-device-plugin-for-kubernetes "here"))
- Java

## Deploy Trusted Realtime ML for Kubernetes ##
1. Pull docker image from dockerhub
	```
	$ docker pull intelanalytics/bigdl-ppml-trusted-realtime-ml-scala-graphene:0.14.0-SNAPSHOT
	```
2. Pull the source code of BigDL and enter PPML graphene k8s directory
	```
	$ git clone https://github.com/intel-analytics/BigDL.git
	$ cd BigDL/ppml/trusted-realtime-ml/scala/docker-graphene/kubernetes
	```
3. Generate secure keys and passwords, and deploy as secrets (Refer [here](https://github.com/intel-analytics/BigDL/tree/branch-2.0/ppml/trusted-realtime-ml/scala/docker-graphene/kubernetes#secure-keys-and-password) for details)
	1. Generate keys and passwords
		
		Note: Make sure to add `${JAVA_HOME}/bin` to `$PATH` to avoid `keytool: command not found` error.
		```
		$ sudo ../../../../scripts/generate-keys.sh
		$ openssl genrsa -3 -out enclave-key.pem 3072
		$ ../../../../scripts/generate-password.sh <used_password_when_generate_keys>
		```
	2. Deploy as secrets for Kubernetes
		```
		$ kubectl apply -f keys/keys.yaml
		$ kubectl apply -f password/password.yaml
		```

4. In `values.yaml`, configure pulled image name, path of `enclave-key.pem` generated in step 3 and path of script `start-all-but-flink.sh`.
5. If kernel version is 5.11+ with built-in SGX support, create soft links for SGX device
	```
	$ sudo ln -s /dev/sgx_enclave /dev/sgx/enclave
	$ sudo ln -s /dev/sgx_provision /dev/sgx/provision
	```

### Configure SGX mode ###
In `templates/flink-configuration-configmap.yaml`, configure `sgx.mode` to `sgx` or `nonsgx` to determine whether to run the workload with SGX.

### Configure Resource for Components ###
1.  Configure jobmanager resource allocation in `templates/jobmanager-deployment.yaml`
	```
	...
	env:
      - name: SGX_MEM_SIZE
        value: "16G"
	...
    resources:
      requests:
        cpu: 2
        memory: 16Gi
        sgx.intel.com/enclave: "1"
        sgx.intel.com/epc: 16Gi
      limits:
        cpu: 2
        memory: 16Gi
        sgx.intel.com/enclave: "1"
        sgx.intel.com/epc: 16Gi
	...
	```
	
2.  Configure Taskmanager resource allocation
	- Memory allocation in `templates/flink-configuration-configmap.yaml`
		```
		taskmanager.memory.managed.size: 4gb
	    taskmanager.memory.task.heap.size: 5gb
	    xmx.size: 5g
	 	```
	- Pod resource allocation
		
		Use `taskmanager-deployment.yaml` instead of `taskmanager-statefulset.yaml` for functionality test
		```
		$ mv templates/taskmanager-statefulset.yaml ./
		$ mv taskmanager-deployment.yaml.back templates/taskmanager-deployment.yaml
		``` 
		Configure resource in `templates/taskmanager-deployment.yaml` (allocate 16 cores in this example, please configure according to scenario)
		```
		...
		env:
	      - name: CORE_NUM
	        value: "16"
	      - name: SGX_MEM_SIZE
	        value: "32G"
		...
	    resources:
	      requests:
	        cpu: 16
	        memory: 32Gi
	        sgx.intel.com/enclave: "1"
	        sgx.intel.com/epc: 32Gi
	      limits:
	        cpu: 16
	        memory: 32Gi
	        sgx.intel.com/enclave: "1"
	        sgx.intel.com/epc: 32Gi
		...
		```
3. Configure Redis and client resource allocation
   - SGX memory allocation in `start-all-but-flink.sh`
	   ```
		...
		cd /ppml/trusted-realtime-ml/java
		export SGX_MEM_SIZE=16G
		test "$SGX_MODE" = sgx && ./init.sh
		echo "java initiated"
		...
		```
   - Pod resource allocation in `templates/master-deployment.yaml`
		```
		...
		env:
	      - name: CORE_NUM  #batchsize per instance
	        value: "16"
		...
	    resources:
	      requests:
	        cpu: 12
	        memory: 32Gi
	        sgx.intel.com/enclave: "1"
	        sgx.intel.com/epc: 32Gi
	      limits:
	        cpu: 12
	        memory: 32Gi
	        sgx.intel.com/enclave: "1"
	        sgx.intel.com/epc: 32Gi
		...
		```

### Deploy Cluster Serving ###
1. Deploy all components and start job
	1. Download helm from [release page](https://github.com/helm/helm/releases) and install
	2. Deploy cluster serving
		```
		$ helm install ppml ./
		```   
2. Port forwarding

   Set up port forwarding of jobmanager Rest port for access to Flink WebUI on host
   1. Run `kubectl port-forward <flink-jobmanager-pod> --address 0.0.0.0 8081:8081` to forward jobmanager’s web UI port to 8081 on host.
   2. Navigate to `http://<host-IP>:8081` in web browser to check status of Flink cluster and job.
3. Performance benchmark
	```
	$ kubectl exec <master-deployment-pod> -it -- bash
	$ cd /ppml/trusted-realtime-ml/java/work/benchmark/
	$ bash init-benchmark.sh
	$ python3 e2e_throughput.py -n <image_num> -i ../data/ILSVRC2012_val_00000001.JPEG
	```
	The `e2e_throughput.py` script pushes test image for `-n` times (default 1000 if not manually set), and time the process from push images (enqueue) to retrieve all inference results (dequeue), to calculate cluster serving end-to-end throughput. The output should look like `Served xxx images in xxx sec, e2e throughput is xxx images/sec`
