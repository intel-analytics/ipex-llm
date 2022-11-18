# A Hello World Example


In this section, you can get started with running a simple native python HelloWorld program and a simple native Spark Pi program locally in a BigDL PPML client container to get an initial understanding of the usage of ppml.



## a. Prepare Keys

* generate ssl_key

  Download scripts from [here](https://github.com/intel-analytics/BigDL).

  ```
  cd BigDL/ppml/
  sudo bash scripts/generate-keys.sh
  ```
  This script will generate keys under keys/ folder

* generate enclave-key.pem

  ```
  openssl genrsa -3 -out enclave-key.pem 3072
  ```
  This script generates a file enclave-key.pem which is used to sign image.


## b. Start the BigDL PPML client container

```
#!/bin/bash

# ENCLAVE_KEY_PATH means the absolute path to the "enclave-key.pem" in step a
# KEYS_PATH means the absolute path to the keys folder in step a
# LOCAL_IP means your local IP address.
export ENCLAVE_KEY_PATH=YOUR_LOCAL_ENCLAVE_KEY_PATH
export KEYS_PATH=YOUR_LOCAL_KEYS_PATH
export LOCAL_IP=YOUR_LOCAL_IP
export DOCKER_IMAGE=intelanalytics/bigdl-ppml-trusted-big-data-ml-python-graphene:devel

sudo docker pull $DOCKER_IMAGE

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
    -v $KEYS_PATH:/ppml/trusted-big-data-ml/work/keys \
    --name=bigdl-ppml-client-local \
    -e LOCAL_IP=$LOCAL_IP \
    -e SGX_MEM_SIZE=64G \
    $DOCKER_IMAGE bash
```

## c. Run Python HelloWorld in BigDL PPML Client Container

Run the [script](https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/python/docker-graphene/start-scripts/start-python-helloworld-sgx.sh) to run trusted [Python HelloWorld](https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/python/docker-graphene/examples/helloworld.py) in BigDL PPML client container:
```
sudo docker exec -it bigdl-ppml-client-local bash work/start-scripts/start-python-helloworld-sgx.sh
```
Check the log:
```
sudo docker exec -it bigdl-ppml-client-local cat /ppml/trusted-big-data-ml/test-helloworld-sgx.log | egrep "Hello World"
```
The result should look something like this:
> Hello World


## d. Run Spark Pi in BigDL PPML Client Container

Run the [script](https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/python/docker-graphene/start-scripts/start-spark-local-pi-sgx.sh) to run trusted [Spark Pi](https://github.com/apache/spark/blob/v3.1.2/examples/src/main/python/pi.py) in BigDL PPML client container:

```bash
sudo docker exec -it bigdl-ppml-client-local bash work/start-scripts/start-spark-local-pi-sgx.sh
```

Check the log:

```bash
sudo docker exec -it bigdl-ppml-client-local cat /ppml/trusted-big-data-ml/test-pi-sgx.log | egrep "roughly"
```

The result should look something like this:

> Pi is roughly 3.146760

<br />
