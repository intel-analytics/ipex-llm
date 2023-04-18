## 1. Build image
You can pull the BigDL Remote Attestation Service image from dockerhub.
``` bash
docker pull intelanalytics/bigdl-attestation-service:2.3.0-SNAPSHOT
```
Or you can clone BigDL repository and build the image with `build-docker-image.sh`.
First you need to generate your enclave key using the command below, and keep it safely for future remote attestations and to start SGX enclaves more securely.
```bash
openssl genrsa -3 -out enclave-key.pem 3072
```
After configure variables in `build-docker-image.sh`, build the container with command:
```bash
bash build-docker-image.sh
```

## 2. Start container

```bash
export DATA_PATH=
export KEYS_PATH=
export NFS_INPUT_PATH=
export LOCAL_IP=
export DOCKER_IMAGE=intelanalytics/bigdl-attestation-service:2.3.0-SNAPSHOT

export PCCS_URL=
export HTTPS_KEY_STORE_TOKEN=
export SECRET_KEY=bigdl
export SGX_ENABLED=false
export ATTESTATION_SERVICE_HOST=0.0.0.0
export ATTESTATION_SERVICE_PORT=9875

sudo docker run -itd \
--privileged \
--net=host \
--name=bigdl-remote-attestation-service \
--cpuset-cpus="31-34" \
--oom-kill-disable \
--device=/dev/sgx_enclave \
--device=/dev/sgx_provision \
-v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
-v $DATA_PATH:/opt/bigdl-as/data \
-v $KEYS_PATH:/ppml/keys \
-v $NFS_INPUT_PATH:/ppml/data \
-e LOCAL_IP=$LOCAL_IP \
-e PCCS_URL=$PCCS_URL \
-e HTTPS_KEY_STORE_TOKEN=$HTTPS_KEY_STORE_TOKEN \
-e SECRET_KEY=$SECRET_KEY \
-e SGX_ENABLED=$SGX_ENABLED \
-e ATTESTATION_SERVICE_HOST=$ATTESTATION_SERVICE_HOST \
-e ATTESTATION_SERVICE_PORT=$ATTESTATION_SERVICE_PORT \
$DOCKER_IMAGE 
```

Detailed usages can refer to [this](https://github.com/intel-analytics/BigDL/tree/main/scala/ppml/src/main/scala/com/intel/analytics/bigdl/ppml/attestation)