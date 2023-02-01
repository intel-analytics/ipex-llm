# trusted-python-toolkit
This image contains Gramine and some popular python toolkits including numpy, pandas, flask and torchserve.

*Please mind the IP and file path settings. They should be changed to the IP/path of your own sgx server on which you are running.*

## 1. Build Docker Images

**Tip:** if you want to skip building the custom image, you can use our public image `intelanalytics/bigdl-ppml-trusted-python-toolkit-ref:2.3.0-SNAPSHOT` for a quick start, which is provided for a demo purpose. Do not use it in production.

### 1.1 Build Gramine Base Image
Gramine base image provides necessary tools including gramine, python, java, etc for the image in this directory. You can build your own gramine base image following the steps in [Gramine PPML Base Image](https://github.com/intel-analytics/BigDL/tree/main/ppml/base#gramine-ppml-base-image). You can also use our public image `intelanalytics/bigdl-ppml-gramine-base:2.3.0-SNAPSHOT` for a quick start.

### 1.2 Build Python Toolkit Base Image

The python toolkit base image is a public one that does not contain any secrets. You will use the base image to get your own custom image. 

You can use our public base image `intelanalytics/bigdl-ppml-trusted-python-toolkit-base:2.3.0-SNAPSHOT`, or, You can build your own base image based on `intelanalytics/bigdl-ppml-gramine-base:2.3.0-SNAPSHOT`  as follows. Remember to assign values to the variables in `build-toolkit-base-image.sh` before running the script.

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

## 2. Demo

*WARNING: We are currently actively developing our images, which indicate that the ENTRYPOINT of the docker image may be changed accordingly.  We will do our best to update our documentation in time.*

### 2.1 Start the container

Use the following code to start the container.
```shell
export LOCAL_IP=your_local_ip
export DOCKER_IMAGE=your_docker_image
sudo docker run -itd \
	--privileged \
	--net=host \
	--name= your_container_name\
	--cpus=10 \
	--oom-kill-disable \
	--device=/dev/sgx/enclave \
	--device=/dev/sgx/provision \
	-v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
	-e LOCAL_IP=$LOCAL_IP \
	-e SGX_ENABLED=false \
	-e ATTESTATION=false \
	$DOCKER_IMAGE bash
```

Get into your container and run examples.
```shell
	docker exec -it your_container_name bash
```

### 2.2 Examples

The native python toolkit examples are put under `/ppml/examples`. You can run them on SGX through shell scripts under `/ppml/work/start-scripts`.

#### 2.2.1 Numpy Examples

Change directory to `/ppml/work/start-scripts` and run `start-python-numpy-example-sgx.sh`.
```shell
cd /ppml/work/start-scripts
bash start-python-numpy-example-sgx.sh
```

You will see the version of numpy and the time of numpy dot.
```shell
numpy version: 1.21.6
numpy.dot: ... sec
```

#### 2.2.2 Pandas Examples

Change directory to `/ppml/work/start-scripts` and run `start-python-pandas-example-sgx.sh`.
```shell
cd /ppml/work/start-scripts
bash start-python-pandas-example-sgx.sh
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

#### 2.2.3 Flask Examples

Change directory to `/ppml/work/start-scripts` and run `start-python-flask-sgx.sh`. The flask example will receive a GET/POST request from clients and return the feature of the url and the method of the request.
```shell
cd /ppml/work/start-scripts
bash start-python-flask-sgx.sh
```

You can find the python code that send GET/POST request under `/ppml/examples/flask`.
Run `get.py`. (Remember to modify the ip address to your_local_ip)
```shell
cd /ppml/examples/flask
python get.py
```
You will get:
```shell
Hello World! GET
```

You can try `post.py` similarly. 

#### 2.2.4 Torchserve Example

Before run torchserve on SGX, prepare the config file in which you can assign ip address, model location, worker thread number and so on. The following script shows a simple example of config file. Refer to [5.3. config.properties file](https://pytorch.org/serve/configuration.html#config-properties-file) for more information.
```shell
inference_address=your_inference_address
management_address=your_management_address
metrics_address=your_metrics_address
model_store=your_model_store
load_models=all
enable_metrics_api=false
enable_grpc_ssl=false
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

Note that store your model under your_model_store while set `model_store` to /ppml/work/data/your_model_store.
Refer to [Torch Model archiver for TorchServe](https://github.com/pytorch/serve/blob/master/model-archiver/README.md)to see how to archive your own model.
Start pytorch server.
```shell
cd /ppml/work/start-scripts
bash start-torchserve-sgx.sh -c your_config_file_path
```

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
curl your_inference_address/predictions/densenet161 -T kitten_small.jpg
```

#### Enable SSL
Generate your self signed cert and key with OpenSSL.
```shell
#Remember to set CN which is important to gRPC with SSL.
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout mykey.key -out mycert.pem
```

##### For gRPC
Set `enable_grpc_ssl=true` in config file. Run inference using the sample client:
```shell
#Remember to modify the name of key and CN in torchserve_grpc_client_ssl.py
python ts_scripts/torchserve_grpc_client_ssl.py infer densenet161 examples/image_classifier/kitten.jpg
```

##### For HTTP/REST
Modify the IP addresses from `http` to `https` in config file.
Then, send requests:
```shell
curl --insecure your_inference_address/predictions/densenet161 -T kitten_small.jpg --cert mycert.pem --key mykey.key
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

### 2.3 Run Examples Through Entrypoint

You can also run examples with entrypoint.sh. Take numpy for example:
```shell
export LOCAL_IP=your_local_ip
export DOCKER_IMAGE=your_docker_image
sudo docker run -itd \
	--privileged \
	--net=host \
	--name=your_container_name \
	--cpus=10 \
	--oom-kill-disable \
	--device=/dev/sgx/enclave \
	--device=/dev/sgx/provision \
	-v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
	-e LOCAL_IP=$LOCAL_IP \
	-e ATTESTATION=false \
	$DOCKER_IMAGE /ppml/work/start-scripts/start-python-numpy-example-sgx.sh
docker logs -f your_container_name
```

You will get the same result.
```shell
numpy version: 1.21.6
numpy.dot: ... sec
```

