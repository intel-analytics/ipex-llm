# BigDL PPML Trusted Deep Learning Serving 

SGX-based Trusted Deep Learning Serving (hereinafter called DL-Serving) allows the user to run end-to-end dl-serving services in a secure environment.

The following sections will introduce three different components that are included in this toolkit, which are `TorchServe`, `Triton Inference Server`, and `TensorFlow Serving`.

Besides, some demos and performance benchmark results will also be included in this document.

*Please pay attention to the IP and file path settings. They should be changed to the IP/path of your own sgx server on which you are running the programs.*

## Before Running code
### 1. Build Docker Images

**Tip:** if you want to skip building the custom image, you can use our public image `intelanalytics/bigdl-ppml-trusted-dl-serving-gramine-ref:2.3.0-SNAPSHOT` for a quick start, which is provided for a demo purpose. Do not use it for production.

### 1.1 Build BigDL Base Image

The bigdl base image is a public one that does not contain any secrets. You will use the base image to get your own custom image in the following steps. 

Please be noted that the `intelanalytics/bigdl-ppml-trusted-dl-serving-gramine-base:2.3.0-SNAPSHOT` image relies on the `intelanalytics/bigdl-ppml-gramine-base:2.3.0-SNAPSHOT` image.  

For the instructions on how to build the `gramine-base` image, check `ppml/base/README.md` in our repository.  Another option is to use our public image `intelanalytics/bigdl-ppml-gramine-base:2.3.0-SNAPSHOT` for a quick start.

Before running the following command, please modify the paths in `../base/build-docker-image.sh`. Then build the docker image with the following command.

```bash
# Assuming you are in ppml/trusted-deep-learning/base directory 
# configure parameters in build-docker-image.sh please
./build-docker-image.sh
```
### 1.2 Build Customer Image

First, You need to generate your enclave key using the command below, and keep it safe for future remote attestations and to start SGX enclaves more securely.

It will generate a file `enclave-key.pem` in `ppml/trusted-deep-learning/ref` directory, which will be your enclave key. To store the key elsewhere, modify the outputted file path.

```bash
# Assuming you are in ppml/trusted-deep-learning/ref directory
openssl genrsa -3 -out enclave-key.pem 3072
```

Then, use the `enclave-key.pem` and the `intelanalytics/bigdl-ppml-trusted-deep-learning-gramine-base:2.3.0-SNAPSHOT` image to build your own custom image. In the process, SGX MREnclave will be made and signed without saving the sensitive enclave key inside the final image, which is safer.


Before running the following command, please modify the paths in `./build-custom-image.sh`. Then build the docker image with the following command.

```bash
# under ppml/trusted-deep-learning/ref dir
# modify custom parameters in build-custom-image.sh
./build-custom-image.sh
```

The docker build console will also output `mr_enclave` and `mr_signer` like below, which are hash values and used to register your MREnclave in the following.

```log
......
[INFO] Use the below hash values of mr_enclave and mr_signer to register enclave:
mr_enclave       : c7a8a42af......
mr_signer        : 6f0627955......
```

## Demo

*WARNING: We are currently actively developing our images, which indicate that the ENTRYPOINT of the docker image may be changed accordingly.  We will do our best to update our documentation in time.*

### TorchServe

In order to deploy TorchServe on k8s with helmchart, TorchServe in DL-Serving is slightly different with the original Torchserve. First, the frontend and backends of TorchServe are decoupled and run in different pods. Second, TorchServe does not need the config file because it can generate a config file from the parameters in `service/values.yaml` directly. Third,  we enable model encryption, SSL and sidecar to ensure the security of TorchServe. The following picture shows the architecture of TorchServe in DL-Serving.

To start TorchServe, you need to set the parameters' values in `service/values.yaml` and copy the MAR file which is needed by original TorchServe to `$nfsPath/model/torchserve`. Then, run the helm command to run it on k8s.
```
cd service
helm install $your_helm_name .
```
You can check the status of pods and services by `kubectl`.
```
kubectl get all -n bigdl-ppml-serving

NAME                                                       READY   STATUS    RESTARTS       AGE
pod/bigdl-torchserve-backend-deployment-6bbbdd64ff-6kk4q   1/1     Running   0              4m4s
pod/bigdl-torchserve-backend-deployment-6bbbdd64ff-hpcpx   1/1     Running   0              4m4s
pod/torchserve-frontend                                    1/1     Running   0              4m4s

NAME                                        TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)                      AGE
service/bigdl-torchserve-frontend-service   ClusterIP   ...             <none>        8085/TCP,8081/TCP,8082/TCP   4m4s

NAME                                                  READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/bigdl-torchserve-backend-deployment   2/2     2            2           4m4s

NAME                                                             DESIRED   CURRENT   READY   AGE
replicaset.apps/bigdl-torchserve-backend-deployment-6bbbdd64ff   2         2         2       4m4s

```
Send a prediction request to test if Torchserve is running correctly.
```
curl https://$clusterIP:$inferencePort/predictions/BERT_LARGE -T sample_input_bert.json  -k

# you should get the inference output as response.
[
  [
    0.1334414780139923,
    -0.2059822380542755
  ]
]
```



### Triton Server (WIP)



### TensorFlow Serving (WIP)

