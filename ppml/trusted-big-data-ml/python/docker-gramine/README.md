# Gramine
## Before Running code
### 1. Build Docker Image

Before running the following command, please modify the paths in `build-docker-image.sh`. Then build the docker image with the following command.

```bash
./build-docker-image.sh
```
### 2. Prepare key
- Generate SSL Keys
```
cd BigDL/ppml/
sudo bash scripts/generate-keys.sh
```
- Generate enclave-key.pem
```
openssl genrsa -3 -out enclave-key.pem 3072
```
### 3. Run container
```
#!/bin/bash

# KEYS_PATH means the absolute path to the keys folder
# ENCLAVE_KEY_PATH means the absolute path to the "enclave-key.pem" file
# LOCAL_IP means your local IP address.
export SSL_KEYS_PATH=YOUR_LOCAL_SSL_KEYS_FOLDER_PATH
export ENCLAVE_KEY_PATH=YOUR_LOCAL_ENCLAVE_KEY_PATH
export LOCAL_IP=YOUR_LOCAL_IP
export DOCKER_IMAGE=YOUR_DOCKER_IMAGE

sudo docker run -itd \
    --privileged \
    --net=host \
    --cpuset-cpus="0-5" \
    --oom-kill-disable \
    --device=/dev/gsgx \
    --device=/dev/sgx/enclave \
    --device=/dev/sgx/provision \
    -v $ENCLAVE_KEY_PATH:/graphene/Pal/src/host/Linux-SGX/signer/enclave-key.pem \
    -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
    -v $SSL_KEYS_PATH:/ppml/trusted-big-data-ml/work/keys \
    --name=gramine-test \
    -e LOCAL_IP=$LOCAL_IP \
    -e SGX_MEM_SIZE=64G \
    $DOCKER_IMAGE bash
```
## Test Examples
### 1. Python Examples
#### Example 1:helloworld
```
sudo docker exec -it gramine-test bash work/start-scripts/start-python-helloworld-sgx.sh
```
The result should be:
> Hello World
#### Example 2:numpy
```
sudo docker exec -it gramine-test bash work/start-scripts/start-python-numpy-sgx.sh
```
The result should be like:
> numpy.dot: 0.04753961563110352 sec
### 2. Spark Examples
#### Example 1: pyspark pi
```
sudo docker exec -it gramine-test bash work/start-scripts/start-spark-local-pi-sgx.sh
```
The result should be like:
> pi is roughly 3.135360