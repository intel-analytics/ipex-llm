# PPML Demo Without SGX

## Get Prepared
### Get jar ready
#### Build from source
```bash
git clone https://github.com/intel-analytics/BigDL.git
cd BigDL/scala
./make-dist.sh
```
#### Download pre-build
```bash
wget
```
### Config
If deploying PPML on cluster, need to overwrite config `./ppml-conf.yaml`. Default config (localhost:8980) would be used if no `ppml-conf.yaml` exists in the directory.
### Start FL Server
```bash
java -cp com.intel.analytics.bigdl.ppml.FLServer
```
## HFL Logistic Regression
```bash
# client 1
java -cp com.intel.analytics.bigdl.ppml.example.HflLogisticRegression -d data/diabetes-hfl-1.csv

# client 2
java -cp com.intel.analytics.bigdl.ppml.example.HflLogisticRegression -d data/diabetes-hfl-2.csv
```
## VFL Logistic Regression
```bash
# client 1
java -cp com.intel.analytics.bigdl.ppml.example.VflLogisticRegression -d data/diabetes-vfl-1.csv

# client 2
java -cp com.intel.analytics.bigdl.ppml.example.VflLogisticRegression -d data/diabetes-vfl-2.csv
```


# PPML Demo With SGX

## Before running code

### Prepare Docker Image
#### Build jar from Source

```bash
cd .. && mvn clean package -DskipTests -Pspark_3.x
mv target/bigdl-ppml-spark_3.1.2-0.14.0-SNAPSHOT-jar-with-dependencies.jar demo
cd demo
```

#### Build Image
Modify your `http_proxy` in `build-image.sh` then run:

```bash
./build-image.sh
```

### Prepare the Key

The ppml in bigdl needs secured keys to enable spark security such as Authentication, RPC Encryption, Local Storage Encryption and TLS, you need to prepare the secure keys and keystores. In this tutorial, you can generate keys and keystores with root permission (test only, need input security password for keys).

```bash
bash ../../../ppml/scripts/generate-keys.sh
```

You also need to generate your enclave key using the command below, and keep it safely for future remote attestations and to start SGX enclaves more securely.

It will generate a file `enclave-key.pem` in your present working directory, which will be your enclave key. To store the key elsewhere, modify the outputted file path.

```bash
openssl genrsa -3 -out enclave-key.pem 3072
```

### Prepare the Password

Next, you need to store the password you used for key generation, i.e., `generate-keys.sh`, in a secured file.

```bash
bash ../../../ppml/scripts/generate-password.sh used_password_when_generate_keys
```

Then modify these path and your local ip in `deploy_fl_container.sh`.

## Start container
run:

```bash
bash deploy_fl_container.sh
sudo docker exec -it flDemo bash
./init.sh
```

## Start FLServer
In container run:

```bash
./runFlServer.sh
```

## Run Horizontal FL Demo
Open two new terminals, run:

```bash
sudo docker exec -it flDemo bash
```

to enter the container, then in a terminal run:

```bash
./runHflClient1.sh
```

in another terminal run:

```bash
./runHflClient2.sh
```

## Run Vertical FL Demo
Open two new windows, run:

```bash
sudo docker exec -it flDemo bash
```

to enter the container, then in a terminal run:

```bash
./runVflClient1.sh
```

in another terminal run:

```bash
./runVflClient2.sh
```
