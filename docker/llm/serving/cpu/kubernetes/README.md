## Deployment ipex-llm serving service in K8S environment

## Image

To deploy IPEX-LLM-serving cpu in Kubernetes environment, please use this image: `intelanalytics/ipex-llm-serving-cpu:2.2.0-SNAPSHOT`

## Before deployment

### Models

In this document, we will use `vicuna-7b-v1.5` as the deployment model.

After downloading the model, please change name from `vicuna-7b-v1.5` to `vicuna-7b-v1.5-ipex-llm` to use `ipex-llm` as the backend. The `ipex-llm` backend will be used if model path contains `ipex-llm`. Otherwise, the original transformer-backend will be used.

You can download the model from [here](https://huggingface.co/lmsys/vicuna-7b-v1.5).

### Kubernetes config

We recommend to setup your kubernetes cluster before deployment.  Mostly importantly, please set `cpu-management-policy` to `static` by using this [tutorial](https://kubernetes.io/docs/tasks/administer-cluster/cpu-management-policies/).  Also, it would be great to also set the `topology management policy` to `single-numa-node`.

### Machine config

Set hyper-threading to off, ensure that only physical cores are used during deployment.

## Deployment

### Reminder on `OMP_NUM_THREADS`

The entrypoint of the image will try to set `OMP_NUM_THREADS` to the correct number by reading configs from the `runtime`.  However, this only happens correctly if the `core-binding` feature is enabled.  If not, please set environment variable `OMP_NUM_THREADS` manually in the yaml file.

### Vllm usage

If you want to use the vllm AsyncLLMEngine for serving, you should set the args -w vllm_worker in worker part of deployment.yaml.

### PersistentVolume

We use the following yaml file for PersistentVolume deployment:

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: models-pv
  labels:
    app: models
spec:
  capacity:
    storage: 10Gi #Modify according to model size
  accessModes:
    - ReadWriteMany
  storageClassName: models
  nfs:
    path: YOUR_NFS_PATH
    server: YOUR_NFS_SERVER

```

Then you should upload model to `YOUR_NFS_PATH`

### Controller

We use the following yaml file for controller deployment:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: ipex-llm-fschat-a1234bd-controller
  labels:
    fastchat-appid: a1234bd
    fastchat-app-type: controller
spec:
  dnsPolicy: "ClusterFirst"
  containers:
  - name: fastchat-controller # fixed
    image: intelanalytics/ipex-llm-serving-cpu:2.2.0-SNAPSHOT
    imagePullPolicy: IfNotPresent
    env:
    - name: CONTROLLER_HOST # fixed
      value: "0.0.0.0"
    - name: CONTROLLER_PORT # fixed
      value: "21005"
    - name: API_HOST # fixed
      value: "0.0.0.0"
    - name: API_PORT # fixed
      value: "8000"
    - name: "GRADIO_PORT" # You can change this port
      value: "8002"
    ports:
      - containerPort: 21005
        name: con-port
      - containerPort: 8000
        name: api-port
    resources:
      requests:
        memory: 16Gi
        cpu: 4
      limits:
        memory: 16Gi
        cpu: 4
    args: ["-m", "controller"]
  restartPolicy: "Never"
---
# Service for the controller
apiVersion: v1
kind: Service
metadata:
  name: ipex-llm-a1234bd-fschat-controller-service
spec:
  # You may also want to change this to use the cluster's feature
  type: NodePort
  selector:
    fastchat-appid: a1234bd
    fastchat-app-type: controller
  ports:
    - name: cont-port
      protocol: TCP
      port: 21005
      targetPort: 21005
    - name: api-port
      protocol: TCP
      port: 8000
      targetPort: 8000
```

### Worker

We use the following deployment for worker deployment:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ipex-llm-fschat-a1234bd-worker-deployment
spec:
  # Change this to the number you want
  replicas: 1
  selector:
    matchLabels:
      fastchat: worker
  template:
    metadata:
      labels:
        fastchat: worker
    spec:
      dnsPolicy: "ClusterFirst"
      containers:
      - name: fastchat-worker # fixed
        image: intelanalytics/ipex-llm-serving-cpu:2.2.0-SNAPSHOT
        imagePullPolicy: IfNotPresent
        env:
        - name: CONTROLLER_HOST # fixed
          value: ipex-llm-a1234bd-fschat-controller-service
        - name: CONTROLLER_PORT # fixed
          value: "21005"
        - name: WORKER_HOST # fixed
          valueFrom:
            fieldRef:
              fieldPath: status.podIP
        - name: WORKER_PORT # fixed
          value: "21841"
        - name: MODEL_PATH 
          value: "/llm/models/vicuna-7b-v1.5-ipex-llm/" # change this to your model
        - name: OMP_NUM_THREADS
          value: "16"
        resources:
          requests:
            memory: 32Gi
            cpu: 16
          limits:
            memory: 32Gi
            cpu: 16
        args: ["-m", "worker"] # add , "-w", "vllm_worker" if vllm_worker is expected
        volumeMounts:
          - name: llm-models
            mountPath: /llm/models/
      restartPolicy: "Always"
      volumes:
      - name: llm-models
        persistentVolumeClaim:
          claimName: models-pvc
```

You may want to change the `MODEL_PATH` variable in the yaml.  Also, please remember to change the volume path accordingly.

### Use gradio_web_ui

We have set port using `GRADIO_PORT` envrionment variable in `deployment.yaml`, you can use this command

```bash
k port-forward ipex-llm-fschat-a1234bd-controller --address 0.0.0.0 8002:8002
```

Then visit http://YOUR_HOST_IP:8002 to access ui.

### Testing

#### Check pod ip and port mappings

If you need to access the serving on host , you can use `kubectl get nodes -o wide` to get internal ip and `kubectl get service` to get port mappings.

#### Using openai-python

First, install openai-python:

```bash
pip install --upgrade openai
```

Then, interact with model vicuna-7b-v1.5-ipex-llm:

```python
import openai
openai.api_key = "EMPTY"
openai.api_base = "http://localhost:8000/v1"

model = "vicuna-7b-v1.5-ipex-llm"
prompt = "Once upon a time"

# create a completion
completion = openai.Completion.create(model=model, prompt=prompt, max_tokens=64)
# print the completion
print(prompt + completion.choices[0].text)

# create a chat completion
completion = openai.ChatCompletion.create(
  model=model,
  messages=[{"role": "user", "content": "Hello! What is your name?"}]
)
# print the completion
print(completion.choices[0].message.content)
```

#### cURL

cURL is another good tool for observing the output of the api.

Before using cURL, you should set your `http_proxy` and `https_proxy` to empty

```bash
export http_proxy=
export https_proxy=
```

For the following examples, you may also change the service deployment address.

List Models:

```bash
curl http://localhost:8000/v1/models
```

If you have `jq` installed, you can use it to format the output like this:

```bash
curl http://localhost:8000/v1/models | jq
```

Chat Completions:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "YOUR_MODEL",
    "messages": [{"role": "user", "content": "Hello! What is your name?"}]
  }'
```

Text Completions:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "YOUR_MODEL",
    "prompt": "Once upon a time",
    "max_tokens": 41,
    "temperature": 0.5
  }'
```

Embeddings:

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "YOUR_MODEL",
    "input": "Hello world!"
  }'
```
