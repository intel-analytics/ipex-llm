## Deployment trusted bigdl-llm serving tdx service in K8S environment


## Image

To deploy trusted bigdl-llm serving tdx service in Kubernetes environment, please use this image: `intelanalytics/bigdl-ppml-trusted-bigdl-llm-serving-tdx:2.5.0-SNAPSHOT`

## Before deployment

### Models

In this document, we will use `vicuna-7b-v1.5` as the deployment model.

After downloading the model, please change name from `vicuna-7b-v1.5` to `vicuna-7b-v1.5-bigdl` to use `bigdl-llm` as the backend. The `bigdl-llm` backend will be used if model path contains `bigdl`. Otherwise, the original transformer-backend will be used.

You can download the model from [here](https://huggingface.co/lmsys/vicuna-7b-v1.5).

### Kubernetes config

We recommend to setup your kubernetes cluster before deployment.  Mostly importantly, please set `cpu-management-policy` to `static` by using this [tutorial](https://kubernetes.io/docs/tasks/administer-cluster/cpu-management-policies/).  Also, it would be great to also set the `topology management policy` to `single-numa-node`.

### Machine config

Set hyper-threading to off, ensure that only physical cores are used during deployment.

## Deployment

### Reminder on `OMP_NUM_THREADS`

The entrypoint of the image will try to set `OMP_NUM_THREADS` to the correct number by reading configs from the `runtime`.  However, this only happens correctly if the `core-binding` feature is enabled.  If not, please set environment variable `OMP_NUM_THREADS` manually in the yaml file.


### Controller

We use the following yaml file for controller deployment:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: trusted-bigdl-llm-serving-tdx-a1234bd-controller
  labels:
    fastchat-appid: a1234bd
    fastchat-app-type: controller
spec:
  dnsPolicy: "ClusterFirst"
  runtimeClassName: kata-qemu-tdx
  nodeSelector:
    tdx-ac: "1"
  containers:
  - name: trusted-bigdl-llm-serving-tdx-controller # fixed
    image: intelanalytics/bigdl-ppml-trusted-bigdl-llm-serving-tdx:2.5.0-SNAPSHOT
    securityContext:
      privileged: true
      runAsUser: 0
    imagePullPolicy: Always
    env:
    - name: CONTROLLER_HOST # fixed
      value: "0.0.0.0"
    - name: CONTROLLER_PORT # fixed
      value: "21005"
    - name: API_HOST # fixed
      value: "0.0.0.0"
    - name: API_PORT # fixed
      value: "8000"
    - name: "ENABLE_ATTESTATION_API"
      value: "true"
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
    volumeMounts:
      - name: dev
        mountPath: /dev
  restartPolicy: "Never"
  volumes:
  - name: dev
    hostPath:
      path: /dev
---
# Service for the controller
apiVersion: v1
kind: Service
metadata:
  name: trusted-bigdl-llm-serving-tdx-a1234bd-controller-service
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
---
# Service for the controller
apiVersion: v1
kind: Service
metadata:
  name: trusted-bigdl-llm-serving-tdx-a1234bd-controller-service
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
  name: trusted-bigdl-llm-serving-tdx-a1234bd-worker-deployment
spec:
  # Change this to the number you want
  replicas: 2
  selector:
    matchLabels:
      fastchat: worker
  template:
    metadata:
      labels:
        fastchat: worker
    spec:
      dnsPolicy: "ClusterFirst"
      runtimeClassName: kata-qemu-tdx
      nodeSelector:
        tdx-ac: "1"
      containers:
      - name: trusted-bigdl-llm-serving-tdx-worker # fixed
        image: intelanalytics/bigdl-ppml-trusted-bigdl-llm-serving-tdx:2.5.0-SNAPSHOT
        securityContext:
          runAsUser: 0
          privileged: true
        imagePullPolicy: Always
        env:
        - name: CONTROLLER_HOST # fixed
          value: trusted-bigdl-llm-serving-tdx-a1234bd-controller-service
        - name: CONTROLLER_PORT # fixed
          value: "21005"
        - name: WORKER_HOST # fixed
          valueFrom:
            fieldRef:
              fieldPath: status.podIP
        - name: WORKER_PORT # fixed
          value: "21841"
        - name: MODEL_PATH # Change this
          value: "/ppml/models/vicuna-7b-bigdl/"
        - name: OMP_NUM_THREADS
          value: "16"
        - name: "ENABLE_ATTESTATION_API"
          value: "true"
        resources:
          requests:
            memory: 32Gi
            cpu: 16
          limits:
            memory: 32Gi
            cpu: 16
        args: ["-m", "worker"]
        volumeMounts:
          - name: dev
            mountPath: /dev
          - name: ppml-models
            mountPath: /ppml/models/
      restartPolicy: "Always"
      volumes:
      - name: dev
        hostPath:
          path: /dev
      - name: ppml-models
        hostPath:
          path: /chatllm/models # change this in other envs
```

You may want to change the `MODEL_PATH` variable in the yaml.  Also, please remember to change the volume path accordingly.

## Attestation
Please make sure your environment has been configured according to the `tdx-coco` specifications to enable attestation.


### Testing
#### Apply the deployment.yaml
```bash
kubectl apply -f deployment.yaml
```

#### Test using cURL
Find the cluster ip
Your should run this command:
```bash
kubectl get nodes -o wide
```
Then you can see output like this
```bash
NAME     STATUS   ROLES                  AGE   VERSION   INTERNAL-IP
tdx-ac   Ready    control-plane,worker   32d   v1.24.0   172.168.0.216
```
The INTERNAL-IP `172.168.0.216` is ip we need to use

Then find the api-port
```bash
kubectl get svc
```
You can see output like this
```bash
NAME                                                       TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)                          AGE
trusted-bigdl-llm-serving-tdx-a1234bd-controller-service   NodePort    10.107.252.231   <none>        21005:30811/TCP,8000:31717/TCP   44m
```
The controller's port 8000 is mapped to the local port 31717.

You can use this command
```bash
curl --location 'http://host_ip:port/api/v1/attest' \  # Replace the host_ip and port
--header 'Content-Type: application/json' \
--data '{"userdata": "ppml"}'
```
If you get output like this, it indicates that your attestation was successful.
```json
{
  "message": "Success",
  "quote_list": [
    {
      "role": "openai_api_server",
      "quote": "***"
    },
    {
      "role": "controller",
      "quote": "***"
    },
    {
      "role": "worker-http://ip1:port",
      "quote": "***"
    },
    {
      "role": "worker-http://ip2:port",
      "quote": "***"
    }
  ]
}
```