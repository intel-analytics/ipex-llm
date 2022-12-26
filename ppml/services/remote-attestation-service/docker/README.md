## 1. Build image
After configure variables in `build-docker-image.sh`, build the container with command:
```bash
bash build-docker-image.sh
```

## 2. Start container

### SGX
If you want to verify SGX quote, you need to mount SGX device to the docker container.
```bash
export DOCKER_IMAGE=your_docker_image
sudo docker run -itd \
--privileged \
--net=host \
--name=$DOCKER_NAME \
--oom-kill-disable \
--device=/dev/sgx_enclave \
--device=/dev/sgx_provision \
-v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
$DOCKER_IMAGE 
```

### TDX
Mount TDX device `/dev/tdx-attest` when start container.
```bash
export DOCKER_IMAGE=your_docker_image
export DOCKER_NAME=your_docker_name
sudo docker run -itd \
--privileged \
--net=host \
--name=$DOCKER_NAME \
--oom-kill-disable \
--device=/dev/tdx-attest \
-v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
$DOCKER_IMAGE 
```

3. Start Attestation Service
```bash
docker exec -it $DOCKER_NAME bash

# For SGX
bash ./init_sgx.sh

java -cp $BIGDL_HOME/jars/*:$SPARK_HOME/jars/*:$SPARK_HOME/examples/jars/*: com.intel.analytics.bigdl.ppml.service.BigDLRemoteAttestationService -u <serviceURL> -p <servicePort> -s <httpsKeyStoreToken> -t <httpsKeyStorePath> -h <httpsEnabled>
```
Detailed usages can refer to [this]()