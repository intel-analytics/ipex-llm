## 1. Pull/Build Base Image

Download base image as below:

```bash
docker pull intelanalytics/bigdl-kms-base:2.5.0-SNAPSHOT
```

Or you are allowed to build the image manually:
```
cd base
# Note: set the arguments inside the build script first
./build-docker-image.sh
```

## 2. Pull/Build Custom Image

In consider of security, SGX user needs to build his own custom-signed image, and thus the compiling and execution of in-enclave application are verifiable. This can be achived by the following:

```
cd custom
openssl genrsa -3 -out enclave-key.pem 3072
./build-custom-image.sh
```

Or you can download our executable reference image directly if do not want to build custom image by yourself, but note that it is not feasible in production as the reference image is signed by open key of BigDL:

```bash
docker pull intelanalytics/bigdl-kms-reference:2.5.0-SNAPSHOT
```

More details about custom-signed gramine-sgx image can be seen [here](https://github.com/intel-analytics/BigDL/tree/main/ppml#step-0-preparation-your-environment).

