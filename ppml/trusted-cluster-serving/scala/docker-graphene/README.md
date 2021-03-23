# trusted-cluster-serving
Please mind the ip and file path settings, they should be changed to the ip/path of your own sgx server on which you are running the programs.

## How To Build
Before run the following command, please modify the pathes in the build-docker-image.sh file at first. <br>
Then build docker image by running this command: <br>
```bash
./build-docker-image.sh
```

## How To Run
### Prepare the keys
The ppml in analytics zoo need secured keys to enable flink TLS, https and tlse enabled Redis, you need to prepare the secure keys and keystores. <br>
This script is under /analytics-zoo/ppml: <br>
```bash
./generate-keys.sh
```
You also need to store the password you used in previous step in a secured file: <br>
This script is also under /analytics-zoo/ppml: <br>
```bash
./generate-password.sh used_password_in_generate-keys.sh
```
For example: <br>
`./generate-password.sh abcd1234`

### Run the PPML Docker image
#### In local mode
##### Start the container to run analytics zoo cluster serving in ppml.
Before run the following command, please modify the pathes in the run-docker-local-example.sh file at first. <br>
Then run the example with docker: <br>
```bash
./start-local-cluster-serving.sh
```

#### In distributed mode
##### setup passwordless ssh login to all the nodes.
##### config the environments for master, workers, docker image and security keys/passowrd files.
```bash
nano environments.sh
```
##### start the distributed cluster serving
```bash
./start-distributed-cluster-serving.sh
```
##### stop the distributed cluster serving 
```bash
./stop-distributed-cluster-serving.sh
```
