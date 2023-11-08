# Trusted BigDL-LLM using fastchat with Gramine
For running fastchat using bigdl-llm transformers int4 in Gramine

## Prerequisites
1.Check SGX and Kubernetes env.

2.Pull image from dockerhub.
```bash
docker pull intelanalytics/bigdl-ppml-trusted-bigdl-llm-gramine-ref:2.5.0-SNAPSHOT
```

## Deploy fastchat with openAI restful API in K8S cluster

0. prepare model and models_path(host or nfs), change model_name with bigdl.
Refer to [bigdl-llm](https://github.com/intel-analytics/BigDL/tree/main/python/llm) for more  information.
```bash
mv vicuna-7b-hf vicuna-7b-bigdl
```
1. get `controller-service.yaml` and `controller.yaml` and `worker.yaml`, and update the `nodeSelector`.
2. deploy controller-service and controller.
```bash
kubectl apply -f controller-service.yaml
kubectl apply -f controller.yaml
```
3. modify `worker.yaml`, set models mount path and `MODEL_PATH`.
```bash
kubectl apply -f worker.yaml
```
4. using openAI api to access
```
curl http://$controller_ip:8000/v1/models
# choose a model
curl http://$controller_ip:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vicuna-7b-bigdl",
    "prompt": "Once upon a time",
    "max_tokens": 64,
    "temperature": 0.5
  }'
```
More api details refer to [here](https://github.com/lm-sys/FastChat/blob/main/docs/openai_api.md)

## Deploy fastchat in Docker
Please refer to [here](https://github.com/intel-analytics/BigDL/tree/main/python/llm/src/bigdl/llm/serving#start-the-service)

To run inside SGX, need to make corresponding changes like k8s deployment.