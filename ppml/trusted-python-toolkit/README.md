# Gramine
This image contains Gramine and some popular python toolkits including numpy, pandas and flask.

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

## 2. Demo

*WARNING: We are currently actively developing our images, which indicate that the ENTRYPOINT of the docker image may be changed accordingly.  We will do our best to update our documentation in time.*

### 2.1 Examples
#### 2.1.1 Numpy Examples

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
	-e ATTESTATION=false \
	$DOCKER_IMAGE python /ppml/examples/numpy/hello-numpy.py
docker logs -f your_docker_image
```

You will see the version of numpy and the time of numpy dot.
```shell
numpy version: 1.21.6
numpy.dot: 0.010580737050622702 sec
```

#### 2.1.2 Pandas Examples

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

### 2.2 Benchmark
#### 2.2.1 Numpy Benchmark

Use the following code to build a container and test the performance of numpy. Set `-n` to tune the size of array and set `-t` to select the type of data.
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
	-e ATTESTATION=false \
	$DOCKER_IMAGE /ppml/work/start-scripts/start-python-numpy-sgx.sh -n 4096 -t int
docker logs -f your_docker_image
```
You will see the result of the performance test.
```shell
Dotted two 100x100 matrices in ... s.
SVD of a 100x100 matrix in ... s.
Cholesky decomposition of a 500x500 matrix in ... s.
Eigendecomposition of a 100x100 matrix in ... s.
```

#### 2.2.2 Pandas Benchmark

Before testing the performance of pandas, download the [dataset](https://www.kaggle.com/datasets/rdwstats/open-data-rdw-gekentekende-voertuigen/download?datasetVersionNumber=1) and put it under `your_nfs_input_path` first.
Use the following code to build a container and test the performance of pandas.
```shell
export NFS_INPUT_PATH=your_nfs_input_path
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
	-v $NFS_INPUT_PATH:/ppml/work/data \
	-e LOCAL_IP=$LOCAL_IP \
	-e ATTESTATION=false \
	$DOCKER_IMAGE /ppml/work/start-scripts/start-python/pandas-sgx.sh -d your_data
docker logs -f your_docker_image
```

You will see the result of the performance test.
```shell
Complex select
Time elapsed:  ...s
Sorting the dataset
Time elapsed:  ...s
Joining the dataset
Time elapsed:  ...s
Self join
Time elapsed:  ...s
Grouping the data
Time elapsed:  ...s
```


