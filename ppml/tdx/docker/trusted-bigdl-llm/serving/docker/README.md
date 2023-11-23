# How to run the trusted llm serving on TDX

## Build bigdl-ppml-trusted-bigdl-llm-serving-tdx image
```bash
docker build \
  --build-arg http_proxy=.. \
  --build-arg https_proxy=.. \
  --build-arg no_proxy=.. \
  --rm --no-cache -t intelanalytics/trusted-bigdl-llm-serving-tdx:2.5.0-SNAPSHOT .
```
## Start the controller and worker images of bigdl-ppml-trusted-bigdl-llm-serving-tdx in the tdx-vm.

First, you need to enter tdx-vm and make sure you have the necessary environment, such as `/dev/tdx-guest`.

### Generate APP_ID and API_KEY for ATTESTATION use.

You can use this script to enroll and generate app_id and api_key. Please replace `your service url`
```python
import requests

headers = {"Content-Type":"application/json"}

resp = requests.get(url="your service url", headers=headers, verify=False)

print(resp.text)
```

Then I can get result like:
```text
{
  "appID" : "03821788-163d-4291-8bc9-eab5304da9dc",
  "apiKey" : "tTQKYFOR9mZsbDMB2D0BKCf1bjDmPMAO"
}
```

### Start bigdl-ppml-trusted-bigdl-llm-serving-tdx controller
You can start the controller using the command, similar to the one below:
```bash
export DOCKER_IMAGE=intelanalytics/bigdl-ppml-trusted-bigdl-llm-serving-tdx:2.5.0-SNAPSHOT
sudo docker run -itd \
        --privileged \
        --net=host \
        --cpuset-cpus="0-47" \  # According to your machine's configuration
        --cpuset-mems="0" \  # According to your machine's configuration
        --memory="32G" \  # According to your machine's configuration
        --name=llm-serving-cpu-tdx-controller \
        --shm-size="16g" \
        -e ATTESTATION=true \
        -e ENABLE_ATTESTATION_API=true \
        -e ATTESTATION_URL=YOUR_ATTESTATION_URL \
        -e APP_ID=YOUR_APP_ID \
        -e API_KEY=YOUR_API_KEY \
        -v /dev/tdx-guest:/dev/tdx-guest \
        $DOCKER_IMAGE \
        -m controller
```
### Start bigdl-ppml-trusted-bigdl-llm-serving-tdx worker
You can start the worker using the command, similar to the one below:
```bash
sudo docker run -itd \
    --privileged \
    --net=host \
    --cpuset-cpus="0-47" \  # According to your machine's configuration
    --cpuset-mems="0" \  # According to your machine's configuration
    --memory="32G" \  # According to your machine's configuration
    --name=llm-serving-cpu-tdx-worker \
    --shm-size="16g" \
    -e ATTESTATION=true \
    -e ENABLE_ATTESTATION_API=true \
    -e ATTESTATION_URL=YOUR_ATTESTATION_URL \
    -e APP_ID=YOUR_APP_ID \
    -e API_KEY=YOUR_API_KEY \
    -v /dev/tdx-guest:/dev/tdx-guest \
    -e MODEL_PATH=/llm/model/lmsys-vicuna-7b-v1.5-bigdl \
    -v /bigdl/models/lmsys-vicuna-7b-v1.5-bigdl:/llm/model/lmsys-vicuna-7b-v1.5-bigdl \
    $DOCKER_IMAGE \
    -m worker
```