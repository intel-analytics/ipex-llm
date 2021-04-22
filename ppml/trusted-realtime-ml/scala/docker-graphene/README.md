# Trusted Realtime ML
SGX-based Trusted Big Data ML allows user to run end-to-end Intel Analytics Zoo cluster serving with Flink local and distributed cluster on Graphene-SGX.

*Please mind the IP and file path settings. They should be changed to the IP/path of your own SGX server on which you are running the programs.*

## How To Build
Before running the following command, please modify the paths in `build-docker-image.sh` first. <br>
Then, build the docker image by running: <br>
```bash
./build-docker-image.sh
```

## How To Run
### Prerequisite
To launch Trusted Realtime ML on Graphene-SGX, you need to install graphene-sgx-driver:
```bash
../../../scripts/install-graphene-driver.sh
```

### Prepare the keys
The PPML in Analytics Zoo needs secured keys to enable Flink TLS, https and TLS enabled Redis. You need to prepare the secure keys and keystores. <br>
This script is in /analytics-zoo/ppml/scripts: <br>
```bash
../../../scripts/generate-keys.sh
```
You also need to generate your enclave key using the command below, and safekeep it for future remote attestations and to start SGX enclaves more securely.
It will generate a file `enclave-key.pem` in your present working directory, which will be your enclave key. To store the key elsewhere, modify the outputted file path.
```bash
openssl genrsa -3 -out enclave-key.pem 3072
```
### Prepare the password
Next, you need to store the password you used in the previous step in a secured file: <br>
This script is also in /analytics-zoo/ppml/scripts: <br>
```bash
../../../scripts/generate-password.sh used_password_when_generate_keys
```

### Run the PPML as Docker containers
#### In local mode
##### Start the container to run analytics zoo cluster serving in ppml.
Before running the following command, please modify the paths in `start-local-cluster-serving.sh` first. <br>
Then run the example with docker: <br>
```bash
./start-local-cluster-serving.sh
```
##### Troubleshooting
You can run the script `/ppml/trusted-realtime-ml/check-status.sh` in the docker container to check whether the components have been correctly started.

To test a specific component, pass one or more argument to it among the following:
"redis", "flinkjm", "flinktm", "frontend", and "serving". For example, run the following command to check the status of the Flink job manager.

```bash
docker exec -it flink-local bash /ppml/trusted-realtime-ml/check-status.sh flinkjm
```

To test all components, you can either pass no argument or pass the "all" argument.

```bash
docker exec -it flink-local bash /ppml/trusted-realtime-ml/check-status.sh
```
If all is well, the following results should be displayed:

```
Detecting redis status...
Redis initilization successful.
Detecting Flink job manager status...
Flink job manager initilization successful.
Detecting Flink task manager status...
Flink task manager initilization successful.
Detecting http frontend status. This may take a while.
Http frontend initilization successful.
Detecting cluster-serving-job status...
cluster-serving-job initilization successful.
```

It is suggested to run this script once after starting local cluster serving to verify that all components are up and running.


#### In distributed mode
*Please setup passwordless ssh login to all the nodes first.*
##### Specify the environments for master, workers, docker image and security keys/password files in environments.sh.
```bash
nano environments.sh
```
##### Start the distributed cluster serving
To start all the services of distributed cluster serving, run
```bash
./start-distributed-cluster-serving.sh
```
You can also run the following command to start the flink jobmanager and taskmanager containers only:
```bash
./deploy-standalone-flink.sh
```
##### Stop the distributed cluster serving
To stop all the services of distributed cluster serving, run
```bash
./stop-distributed-cluster-serving.sh
```
You can also run the following command to stop the flink jobmanager and taskmanager containers only:
```bash
./undeploy-distributed-flink.sh
```

##### Troubleshooting
You can run the script `./distributed-check-status.sh` after starting distributed cluster serving to check whether the components have been correctly started.

To test a specific component, pass one or more argument to it among the following:
"redis", "flinkjm", "flinktm", "frontend", and "serving". For example, run the following command to check the status of the Flink job master.

```bash
bash ./distributed-check-status.sh flinkjm
```

To test all components, you can either pass no argument or pass the "all" argument.

```bash
bash ./distributed-check-status.sh
```
If all is well, the following results should be displayed:

```
Detecting redis status...
Redis initilization successful.
Detecting Flink job manager status...
Flink job manager initilization successful.
Detecting Flink task manager status...
Flink task manager initilization successful.
Detecting http frontend status. This may take a while.
Http frontend initilization successful.
Detecting cluster-serving-job status...
cluster-serving-job initilization successful.
```

It is suggested to run this script once after starting distributed cluster serving to verify that all components are up and running.
