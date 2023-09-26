## Run BF16-Optimized Lora Finetuning on TDX

First, prepare the [docker image](https://github.com/intel-analytics/BigDL/tree/main/ppml/trusted-bigdl-llm/finetune/cpu/docker#prepare-bigdl-image-for-lora-finetuning). Then, similarly, follow [here](https://github.com/intel-analytics/BigDL/tree/main/docker/llm/finetune/lora/cpu#run-bf16-optimized-lora-finetuning-on-kubernetes-with-oneccl) to prepare data and base model, and start fine-tuning with `helm`, which will automatically turn to TDX mode.

### (Optional) Enable TLS
To enable TLS in Remote Attestation API Serving, you should provide a TLS certificate and setting `enableTLS` ( to `true` ), `base64ServerCrt` and `base64ServerKey` in `./kubernetes/values.yaml`.
```bash
# Generate a self-signed TLS certificate (DEBUG USE ONLY)
export COUNTRY_NAME=your_country_name
export CITY_NAME=your_city_name
export ORGANIZATION_NAME=your_organization_name
export COMMON_NAME=your_common_name
export EMAIL_ADDRESS=your_email_address

openssl req -x509 -newkey rsa:4096 -nodes -out server.crt -keyout server.key -days 365 -subj "/C=$COUNTRY_NAME/ST=$CITY_NAME/L=$CITY_NAME/O=$ORGANIZATION_NAME/OU=$ORGANIZATION_NAME/CN=$COMMON_NAME/emailAddress=$EMAIL_ADDRESS/"

# Calculate Base64 format string in values.yaml
cat server.crt | base64 -w 0 # Set in base64ServerCrt
cat server.key | base64 -w 0 # Set in base64ServerKey
```

To use RA Rest API, you need to get the IP of job-launcher:
``` bash
kubectl get all -n bigdl-lora-finetuning 
```
You will find a line like:
```bash
service/bigdl-lora-finetuning-launcher-attestation-api-service   ClusterIP   10.109.87.248   <none>        9870/TCP   17m
```
Here are IP and port of the Remote Attestation API service.

The RA Rest API are listed below:
### 1. Generate launcher's quote
```bash
curl -X POST -H "Content-Type: application/json" -d '{"user_report_data": "<your_user_report_data>"}' http://<your_ra_api_service_ip>:<your_ra_api_service_port>/gen_quote
```

Example responce:

```json
{"quote":"BAACAIEAAAAAAAA..."}
```
### 2. Collect all cluster components' quotes (launcher and workers)
```bash
curl -X POST -H "Content-Type: application/json" -d '{"user_report_data": "<your_user_report_data>"}' http://<your_ra_api_service_ip>:<your_ra_api_service_port>/attest
```

Example responce:

```json
{"quote_list":{"bigdl-lora-finetuning-job-worker-0":"BAACAIEAAAAAAA...","bigdl-lora-finetuning-job-worker-1":"BAACAIEAAAAAAA...","launcher":"BAACAIEAAAAAA..."}}
```
