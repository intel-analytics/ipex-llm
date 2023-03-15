# BigDL PPML Trusted Deep Learning Serving Toolkit
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

We have included bash scripts for start up `TorchServe` in both native/SGX environment in folder `/ppml/torchserve`. Basically, the user only needs to interact with script `/ppml/torchserve/start-torchserve.sh`. In the following subsections, we will show how to run a `TorchServe` service in SGX environment with two backend workers.

#### Prepare model/config files

Same as using normal TorchServe service, users need to prepare the `Model Archive` file using `torch-model-archiver` in advance.  Check [here](https://github.com/pytorch/serve/tree/master/model-archiver#torch-model-archiver-for-torchserve) for detail instructions on how to package the model files into a `mar` file.

TorchServe uses a `config.properties` file to store configurations. Examples can be found [here](https://pytorch.org/serve/configuration.html#config-model). An important configuration is `minWorkers`, the start script will try to boot up to `minWorkers` backends.

To ensure end-to-end security, the SSL should be enabled.  You can refer to the official [document](https://pytorch.org/serve/configuration.html#enable-ssl) on how to enable SSL.


#### Start TorchServe Service in Native mode

In this section, we will try to launch a TorchServe service in native mode.  The example `config.properties` is shown as follows:


```text
inference_address=http://127.0.0.1:8085
management_address=http://127.0.0.1:8081
metrics_address=http://127.0.0.1:8082
grpc_inference_port=7070
grpc_management_port=7071
model_store=/ppml/
#initial_worker_port=25712
load_models=NANO_FP32CL.mar
enable_metrics_api=false
models={\
  "NANO_FP32CL": {\
    "1.0": {\
        "defaultVersion": true,\
        "marName": "NANO_FP32CL.mar",\
        "minWorkers": 2,\
        "workers": 2,\
        "maxWorkers": 2,\
        "batchSize": 1,\
        "maxBatchDelay": 100,\
        "responseTimeout": 1200\
    }\
  }\
}
```

Assuming the above configuration file is stored at `/ppml/tsconfigfp32cl`, then to start the TorchServe Service:

 ```bash
 bash /ppml/torchserve/start-torchserve.sh -c /ppml/tsconfigfp32cl -f "0" -b "1,2"
 ```
As introduced in this performance tuning [guild](https://tutorials.pytorch.kr/intermediate/torchserve_with_ipex#efficient-cpu-usage-with-core-pinning-for-multi-worker-inference), we also pinned the cpu while booting up our frontend and backends. The `"-f 0"` indicates that the frontend will be pinned to core 0, while the `"-b 1,2"` indicates that the first backend will be pinned to core 1, and the second backend will be pinned to core 2.

After the service has booted up, we can use the [wrk](https://github.com/wg/wrk) tool to test its throughput, the result with 5 threads and 10 connections are:

```text
Running 5m test @ http://127.0.0.1:8085/predictions/NANO_FP32CL
  5 threads and 10 connections
  Thread Stats   Avg      Stdev     Max   +/- Stdev
    Latency   355.44ms   19.27ms 783.08ms   98.86%
    Req/Sec     6.42      2.69    10.00     49.52%
  8436 requests in 5.00m, 1.95MB read
Requests/sec:     28.11
Transfer/sec:      6.67KB
```

#### Start TorchServe Service in SGX mode

To start TorchServe service in SGX mode, only a minor modification needs to be made:

```bash
 bash /ppml/torchserve/start-torchserve.sh -c /ppml/tsconfigfp32cl -f "0" -b "1,2" -x
```

The `"-x"` here indicates that the frontend/backend will be run in SGX environment using Gramine.

The performance result:

```text
Running 5m test @ http://127.0.0.1:8085/predictions/NANO_FP32CL
  5 threads and 10 connections
  Thread Stats   Avg      Stdev     Max   +/- Stdev
    Latency     1.24s     2.98s   18.61s    92.24%
    Req/Sec     5.74      2.58    10.00     51.79%
  6379 requests in 5.00m, 1.48MB read
Requests/sec:     21.26
Transfer/sec:      5.04KB
```


### Triton Server (WIP)



### TensorFlow Serving (WIP)