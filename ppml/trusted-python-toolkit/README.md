# trusted-python-toolkit
This image contains Gramine and some popular python toolkits including numpy, pandas, flask and torchserve.

*Please mind the IP and file path settings. They should be changed to the IP/path of your own sgx server on which you are running.*

## 1. Build Docker Images

**Tip:** if you want to skip building the custom image, you can use our public image `intelanalytics/bigdl-ppml-trusted-python-toolkit-ref:2.5.0-SNAPSHOT` for a quick start, which is provided for a demo purpose. Do not use it in production.

### 1.1 Build Gramine Base Image
Gramine base image provides necessary tools including gramine, python, java, etc for the image in this directory. You can build your own gramine base image following the steps in [Gramine PPML Base Image](https://github.com/intel-analytics/BigDL/tree/main/ppml/base#gramine-ppml-base-image). You can also use our public image `intelanalytics/bigdl-ppml-gramine-base:2.5.0-SNAPSHOT` for a quick start.

### 1.2 Build Python Toolkit Base Image

The python toolkit base image is a public one that does not contain any secrets. You will use the base image to get your own custom image. 

You can use our public base image `intelanalytics/bigdl-ppml-trusted-python-toolkit-base:2.5.0-SNAPSHOT`, or, You can build your own base image based on `intelanalytics/bigdl-ppml-gramine-base:2.5.0-SNAPSHOT`  as follows. Remember to assign values to the variables in `build-toolkit-base-image.sh` before running the script.

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
Attributes:
    mr_enclave:  56ba......
    mr_signer:   422c......
````

## 2. Run SGX Application

### 2.1 Start the container

Use the following code to start the container.
```shell
export DOCKER_NAME=
export DOCKER_IMAGE=

docker pull $DOCKER_IMAGE

docker run -itd \
        --privileged \
        --net=host \
        --name=$DOCKER_NAME \
        --oom-kill-disable \
        --device=/dev/sgx/enclave \
        --device=/dev/sgx/provision \
        -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
        $DOCKER_IMAGE bash
docker exec -it $DOCKER_NAME bash
```

Get into your container and run examples.
```shell
	docker exec -it your_container_name bash
```

### 2.2 Examples

The native python toolkit examples are put under `/ppml/examples`. You can run them on SGX through shell scripts under `/ppml/work/scripts`.


#### subprocess

Suppose there is `helloworld.py` in `/ppml`:
```python
print("hello world!")
```

And you want to run it with `subprocess`, so you write `subprocess_helloworld.py` in `/ppml`:
```python
import subprocess

print("Running the python task...")
subprocess.run(["python", "/ppml/helloworld.py"], check=True)
print("python task completed.")
```
Normally, you can execute this file with `python /ppml/subprocess_helloworld.py`.

But if you want to execute it in SGX, you should:
1. Deploy the sgx environment on your machine successfully.
2. Create Python Toolkit container.
3. Run `bash /ppml/init` before executing the SGX command for the first time.
4. `export sgx_command="python /ppml/subprocess_helloworld.py"`
5. `gramine-sgx bash`