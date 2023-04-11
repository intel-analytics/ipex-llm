## Pull/Build container image

Download image as below:

```bash
docker pull intelanalytics/easy-kms:2.3.0-SNAPSHOT
```

Or you are allowed to build the image manually:
```
# Note: set the arguments inside the build script first
bash build-docker-image.sh

In consider of security, SGX user needs to build his own custom-signed image, and thus the compiling and execution of in-enclave application are verifiable. This can be achived by the following:
```
cd sgx
openssl genrsa -3 -out enclave-key.pem 3072
./build-custom-image.sh
```

More details about custom-signed gramine-sgx image can be seen [here](https://github.com/intel-analytics/BigDL/tree/main/ppml#step-0-preparation-your-environment).

