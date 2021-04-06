# Trusted Cluster Serving
SGX-based Trusted Big Data ML allows user to run end to end Intel Analytics Zoo cluster serving with flink local and distributed cluster on Graphene-SGX.

*Please mind the ip and file path settings, they should be changed to the ip/path of your own sgx server on which you are running the programs.*

## How To Build
Before run the following command, please modify the pathes in the build-docker-image.sh file at first. <br>
Then build docker image by running this command: <br>
```bash
./build-docker-image.sh
```

## How To Run
### Prerequisite
To launch Trusted Cluster Serving on Graphene-SGX, you need to install graphene-sgx-driver:
```bash
../../../scripts/install-graphene-driver.sh
```

### Prepare the keys
The ppml in analytics zoo needs secured keys to enable flink TLS, https and TLS enabled Redis, you need to prepare the secure keys and keystores. <br>
This script is in /analytics-zoo/ppml/scripts: <br>
```bash
../../../scripts/generate-keys.sh
```

### Prepare the password
You also need to store the password you used in previous step in a secured file: <br>
This script is also in /analytics-zoo/ppml/scripts: <br>
```bash
../../../scripts/generate-password.sh used_password_when_generate_keys
```

### Run the PPML as Docker containers
#### In local mode
##### Start the container to run analytics zoo cluster serving in ppml.
Before run the following command, please modify the pathes in the start-local-cluster-serving.sh file at first. <br>
Then run the example with docker: <br>
```bash
./start-local-cluster-serving.sh
```
##### Troubleshooting
You can run the script `/ppml/trusted-cluster-serving/check-status.sh` in the docker container to check whether the components have been correctly started. 
Note that this only works for local cluster serving (for now).

To test a specific component, pass one or more argument to it among the following:
"redis", "flinkjm", "flinktm", "frontend", and "serving". For example, run the following command to check the status of the Flink job master.

```bash
docker exec -it flink-local bash /ppml/trusted-cluster-serving/check-status.sh flinkjm
```

To test all components, you can either pass no argument or pass the "all" argument.

```bash
docker exec -it flink-local bash /ppml/trusted-cluster-serving/check-status.sh
```
If all is well, the following results should be displayed:

```
Detecting redis state...
Redis initilization successful.
Detecting Flink job manager state...
Flink job manager initilization successful.
Detecting Flink task manager state...
Flink task manager initilization successful.
Detecting http frontend state. This may take a while.
Http frontend initilization successful.
Detecting cluster-serving-job state...
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
```bash
./start-distributed-cluster-serving.sh
```
##### Stop the distributed cluster serving 
```bash
./stop-distributed-cluster-serving.sh
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
Detecting redis state...
Redis initilization successful.
Detecting Flink job manager state...
Flink job manager initilization successful.
Detecting Flink task manager state...
Flink task manager initilization successful.
Detecting http frontend state. This may take a while.
Http frontend initilization successful.
Detecting cluster-serving-job state...
cluster-serving-job initilization successful.
```

It is suggested to run this script once after starting distributed cluster serving to verify that all components are up and running.
