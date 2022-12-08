# Gramine
This image contains Gramine and some popular python toolkits including numpy, pandas, flask and torchserve.

*Please mind the IP and file path settings. They should be changed to the IP/path of your own sgx server on which you are running.*

## 1. Build Docker Images

**Tip:** if you want to skip building the custom image, you can use our public image `intelanalytics/bigdl-ppml-trusted-python-toolkit-ref:2.2.0-SNAPSHOT` for a quick start, which is provided for a demo purpose. Do not use it in production.

### 1.1 Build Gramine Base Image
Gramine base image provides necessary tools including gramine, python, java, etc for the image in this directory. You can build your own gramine base image following the steps in [Gramine PPML Base Image](https://github.com/intel-analytics/BigDL/tree/main/ppml/base#gramine-ppml-base-image). You can also use our public image `intelanalytics/bigdl-ppml-gramine-base:2.2.0-SNAPSHOT` for a quick start.

### 1.2 Build Python Toolkit Base Image

The python toolkit base image is a public one that does not contain any secrets. You will use the base image to get your own custom image. 

You can use our public base image `intelanalytics/bigdl-ppml-trusted-python-toolkit-base:2.2.0-SNAPSHOT`, or, You can build your own base image based on `intelanalytics/bigdl-ppml-gramine-base:2.2.0-SNAPSHOT`  as follows. Remember to assign values to the variables in `build-toolkit-base-image.sh` before running the script.

```shell
# configure parameters in build-toolkit-base-image.sh please
bash build-toolkit-base-image.sh
```

### 1.3 Build Custom Image

Before build the final image, You need to generate your enclave key using the command below, and keep it safe for future remote attestations and to start SGX enclaves more securely.

It will generate a file `enclave-key.pem` in `./custom-image`. To store the key elsewhere, modify the outputted file path.

```bash
cd custom-image
openssl genrsa -3 -out enclave-key.pem 3072
```

Then, use the `enclave-key.pem` and the toolkit base image to build your own custom image. In the process, SGX MREnclave will be made and signed without saving the sensitive enclave key inside the final image, which is safer.

Remember to assign values to the parameters in `build-custom-image.sh` before running the script.

```bash
# configure parameters in build-custom-image.sh please
bash build-custom-image.sh
```

The docker build console will also output `mr_enclave` and `mr_signer` like below, which are hash values and used to  register your MREnclave in the following.

````bash
......
[INFO] Use the below hash values of mr_enclave and mr_signer to register enclave:
mr_enclave       : c7a8a42af......
mr_signer        : 6f0627955......
````

## 2. Examples

*WARNING: We are currently actively developing our images, which indicate that the ENTRYPOINT of the docker image may be changed accordingly.  We will do our best to update our documentation in time.*

### 2.1 Numpy Examples

Use the following code to build a container and run the numpy example based on the image built before.
```shell
export LOCAL_IP=your_local_ip
export DOCKER_IMAGE=your_docker_image
sudo docker run -itd \
	--privileged \
	--net=host \
	--name= \
	--cpus=10 \
	--oom-kill-disable \
	--device=/dev/sgx/enclave \
	--device=/dev/sgx/provision \
	-v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
	-e LOCAL_IP=$LOCAL_IP \
	-e SGX_ENABLED=true \
	-e ATTESTATION=false \
	$DOCKER_IMAGE python /ppml/examples/numpy/hello-numpy.py
docker logs -f your_docker_image
```

You will see the version of numpy and the time of numpy dot.
```shell
numpy version: 1.21.6
numpy.dot: 0.010580737050622702 sec
```

### 2.2 Pandas Examples

Use the following code to build a container and run the pandas example based on the image built before.
```shell
export LOCAL_IP=your_local_ip
export DOCKER_IMAGE=your_docker_image
sudo docker run -itd \
	--privileged \
	--net=host \
	--name= \
	--cpus=10 \
	--oom-kill-disable \
	--device=/dev/sgx/enclave \
	--device=/dev/sgx/provision \
	-v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
	-e LOCAL_IP=$LOCAL_IP \
	-e SGX_ENABLED=true \
	-e ATTESTATION=false \
	$DOCKER_IMAGE python /ppml/examples/pandas/hello-pandas.py
docker logs -f your_docker_image
```

You will see the version of pandas and a random dataframe.
```shell
pandas version: 1.3.5
Random Dataframe:
    A  ...   J
0  26  ...  52
1  56  ...  98
2  74  ...  28
3   9  ...  67
4  73  ...  73
5  41  ...  74
6  13  ...  37
7  70  ...  31
8  69  ...  47
9  74  ...  75

[10 rows x 10 columns]
```

#### 2.3 Flask Examples

Use the following code to build a container and run the flask example based on the image built before. The flask example will receive a GET/POST request from clients and return the feature of the url and the method of the request.
```shell
export LOCAL_IP=your_local_ip
export DOCKER_IMAGE=your_docker_image
sudo docker run -itd \
	--privileged \
	--net=host \
	--name= \
	--cpus=10 \
	--oom-kill-disable \
	--device=/dev/sgx/enclave \
	--device=/dev/sgx/provision \
	-v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
	-e LOCAL_IP=$LOCAL_IP \
	-e SGX_ENABLED=false \
	-e ATTESTATION=false \
	$DOCKER_IMAGE /ppml/work/start-scripts/start-python-flask-sgx.sh
docker logs -f your_docker_image
```

You can use python to send a request and assign the request method.
If you send a GET request, you can use the following script.
```python
import requests

flask_address = your_flask_address
url = flask_address + '/World!'
res = requests.get(url=url)
print(res.text)
```
Run it and you will get:
```shell
Hello World! GET
```

You can try POST similarly. 

### 2.4 Torchserve Example

Use the following code to build a container and run the torchserve example based on the image built before. 
```shell
export LOCAL_IP=your_local_ip
export DOCKER_IMAGE=your_docker_image
sudo docker run -itd \
	--privileged \
	--net=host \
	--name= \
	--cpus=10 \
	--oom-kill-disable \
	--device=/dev/sgx/enclave \
	--device=/dev/sgx/provision \
	-v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
	-v your_data_path:/ppml/work/data \
	-e LOCAL_IP=$LOCAL_IP \
	-e SGX_ENABLED=false \
	-e ATTESTATION=false \
	$DOCKER_IMAGE /ppml/work/start-scripts/start-torchserve-sgx.sh -c your_config_file_path
docker logs -f your_docker_image
```

Before run torchserve on SGX, prepare the config file in which you can assign ip address, model location, worker thread number and so on. Please make sure that your config file is under your_data_path. The following script shows a simple example of config file. Refer to [5.3. config.properties file](https://pytorch.org/serve/configuration.html#config-properties-file) for more information.
```shell
inference_address=your_inference_address
management_address=your_management_address
metrics_address=your_metrics_address
model_store=/ppml/work/data/your_model_store
load_models=all
models={\
  "densenet161": {\
    "1.0": {\
        "defaultVersion": true,\
        "marName": "densenet161.mar",\
        "minWorkers": 2,\
        "maxWorkers": 2,\
        "batchSize": 4,\
        "maxBatchDelay": 100,\
        "responseTimeout": 1200\
    }\
  }\
}
```

Note that store your model under your_data_path/your_model_store while set `model_store` to /ppml/work/data/your_model_store.
Refer to [Torch Model archiver for TorchServe](https://github.com/pytorch/serve/blob/master/model-archiver/README.md)to see how to archive your own model.
To test the model server, send a request to the server's predictions API through gRPC or HTTP/REST.

#### For gRPC
First, download the torchserve project and install grpc python dependencies:
```shell
git clone https://github.com/analytics-zoo/pytorch-serve.git
pip install -U grpcio protobuf grpcio-tools
```
Then, generate inference client using proto files:
```shell
python -m grpc_tools.protoc --proto_path=frontend/server/src/main/resources/proto/ --python_out=ts_scripts --grpc_python_out=ts_scripts frontend/server/src/main/resources/proto/inference.proto frontend/server/src/main/resources/proto/management.proto
```
Finally, run inference using a sample client:
```shell
python ts_scripts/torchserve_grpc_client.py infer densenet161 examples/image_classifier/kitten.jpg
```

#### For HTTP/REST
Download a picture:
```shell
curl -O https://raw.githubusercontent.com/pytorch/serve/master/docs/images/kitten_small.jpg
```
Send it to pytorch server:
```shell
curl http://your_local_ip:your_inference_address/predictions/densenet161 -T kitten_small.jpg
```

The format of results is similar to the followings.
```shell
{
  "tabby": ...,
  "lynx": ...,
  "tiger_cat": ...,
  "tiger": ...,
  "Egyptian_cat": ...
}
```
