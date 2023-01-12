## 1. Build image
You can pull the BigDL Remote Attestation Service image from dockerhub.
``` bash
docker pull intelanalytics/bigdl-attestation-service:2.2.0-SNAPSHOT
```
Or you can clone BigDL repository and build the image with `build-docker-image.sh`. After configure variables in `build-docker-image.sh`, build the container with command:
```bash
bash build-docker-image.sh
```

## 2. Start container

```bash
sudo docker run -itd \
--privileged \
--net=host \
--name=bigdl-remote-attestation-service \
--oom-kill-disable \
-v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
-e PCCS_URL=127.0.0.1 \
-e ATTESTATION_SERVICE_HOST=0.0.0.0 \
-e ATTESTATION_SERVICE_PORT=9875 \
intelanalytics/bigdl-attestation-service:2.2.0-SNAPSHOT
```

Detailed usages can refer to [this](https://github.com/intel-analytics/BigDL/tree/main/scala/ppml/src/main/scala/com/intel/analytics/bigdl/ppml/attestation)