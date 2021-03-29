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

#### In distributed mode
*Please setup passwordless ssh login to all the nodes first.*
##### Specify the environments for master, workers, docker image and security keys/passowrd files in environments.sh.
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
