# Privacy Preserving Machine Learning(PPML) Demo Without Software Guard Extensions(SGX)

## Get Prepared
### Spark
You need to download `spark-3.1.2-bin-hadoop2.7`. Then delete `$SPARK_HOME/jars/guava-14.0.1.jar`.
### Get jar ready
You can download nightly-build jar
```
NIGHTLY_VERSION=$(echo $(echo `wget -qO - https://oss.sonatype.org/content/repositories/snapshots/com/intel/analytics/bigdl/bigdl-ppml-spark_3.1.2/2.1.0-SNAPSHOT/maven-metadata.xml \
    | sed -n '/<value>[0-9]*\.[0-9]*\.[0-9]*-[0-9][0-9]*\.[0-9][0-9]*-[0-9][0-9]*.*value>/p' | head -n1 | awk -F'>' '{print $2}' | tr '</value' ' '`)) && \
    wget https://oss.sonatype.org/content/repositories/snapshots/com/intel/analytics/bigdl/bigdl-ppml-spark_3.1.2/2.1.0-SNAPSHOT/bigdl-ppml-spark_3.1.2-$NIGHTLY_VERSION-jar-with-dependencies.jar -O ./bigdl-ppml-spark_3.1.2-2.1.0-SNAPSHOT-jar-with-dependencies.jar
```

or build from source
```bash
git clone https://github.com/intel-analytics/BigDL.git
cd BigDL/scala
bash make-dist.sh -DskipTests -Pspark_3.x
```

### Config
If deploying PPML on the cluster, need to overwrite config `./ppml-conf.yaml`. Default config (localhost:8980) would be used if no `ppml-conf.yaml` exists in the directory.

### Tls certificate
If you want to build the Transport Layer Security(TLS) channel with the certificate, you need to prepare the secure keys. In this tutorial, you can generate keys with root permission (test only, need input security password for keys).

**Note: Must enter `localhost` in step `Common Name` for test purposes.**

```bash
sudo bash ../../../ppml/scripts/generate-keys.sh
```

Then modify the `privateKeyFilePath` to `keys/server.pem` and `certChainFilePath` to `keys/server.crt` in `ppml-conf.yaml` with your local path.

If you don't want to build the TLS channel with the certificate, just delete the `privateKeyFilePath` and `certChainFilePath` in `ppml-conf.yaml`.


### Start Federated Learning(FL) Server
```bash
cd ppml/demo
java -cp $SPARK_HOME/jars/*:../target/bigdl-ppml-spark_3.1.2-2.1.0-SNAPSHOT-jar-with-dependencies.jar com.intel.analytics.bigdl.ppml.fl.FLServer
```
## Horizontal Federated Learning (HFL) Logistic Regression
```bash
# client 1
java -cp $SPARK_HOME/jars/*:../target/bigdl-ppml-spark_3.1.2-2.1.0-SNAPSHOT-jar-with-dependencies.jar com.intel.analytics.bigdl.ppml.fl.example.HFLLogisticRegression -d data/diabetes-hfl-1.csv

# client 2
java -cp $SPARK_HOME/jars/*:../target/bigdl-ppml-spark_3.1.2-2.1.0-SNAPSHOT-jar-with-dependencies.jar com.intel.analytics.bigdl.ppml.fl.example.HFLLogisticRegression -d data/diabetes-hfl-2.csv
```
## Vertical Federated Learning (VFL) Logistic Regression
```bash
# client 1
java -cp $SPARK_HOME/jars/*:../target/bigdl-ppml-spark_3.1.2-2.1.0-SNAPSHOT-jar-with-dependencies.jar com.intel.analytics.bigdl.ppml.fl.example.VFLLogisticRegression -d data/diabetes-vfl-1.csv -c 1

# client 2
java -cp $SPARK_HOME/jars/*:../target/bigdl-ppml-spark_3.1.2-2.1.0-SNAPSHOT-jar-with-dependencies.jar com.intel.analytics.bigdl.ppml.fl.example.VFLLogisticRegression -d data/diabetes-vfl-2.csv -c 2
```


# PPML Demo With SGX

## Before running code

### Prepare Docker Image
Pull image from dockerhub.

```bash
docker pull intelanalytics/bigdl-ppml-trusted-fl-graphene:2.1.0-SNAPSHOT
```

Also, you can build image with `build-image.sh`. Configure environment variables in `build-image.sh`.

Build the docker image:

``` bash
bash build-image.sh
```

### Prepare the Key

#### Enclave key
You need to generate your enclave key using the command below, keep it safely for future remote attestations and to start SGX enclaves more securely.

It will generate a file `enclave-key.pem` in your present working directory, which will be your enclave key. To store the key elsewhere, modify the outputted file path.

```bash
openssl genrsa -3 -out enclave-key.pem 3072
```

Then modify `ENCLAVE_KEY_PATH` in `start-fl-server.sh` and `start-fl-client.sh` with your path to `enclave-key.pem`.

#### TLS certificate
If you want to build the TLS channel with the certificate, you need to prepare the secure keys. In this tutorial, you can generate keys with root permission (test only, need input security password for keys).

**Note: Must enter `localhost` in step `Common Name` for test purposes.**

```bash
sudo bash ../../../ppml/scripts/generate-keys.sh
```

If run in container, please modify `KEYS_PATH` to `keys/` you generated in last step in `start-fl-server.sh` and `start-fl-client.sh`. This dir will mount to container's `/ppml/trusted-big-data-ml/work/keys`, then modify the `privateKeyFilePath` to `keys/server.pem` and `certChainFilePath` to `keys/server.crt` in `ppml-conf.yaml` with container's absolute path.

If not in container, just modify the `privateKeyFilePath` to `keys/server.pem` and `certChainFilePath` to `keys/server.crt` in `ppml-conf.yaml` with your local path.

If you don't want to build the TLS channel with the certificate, just delete the `privateKeyFilePath` and `certChainFilePath` in `ppml-conf.yaml`.

Then modify `DATA_PATH` to `./data` with absolute path in your machine and your local IP in `start-fl-server.sh` and `start-fl-client.sh`. The `./data` path will mount to container's `/ppml/trusted-big-data-ml/work/data`, so if you don't run in the container, you need to modify the data path in `scripts/runH_Or_VflClient1_Or_2.sh`.


## Start FLServer

```bash
sudo bash start-fl-server.sh 
```

## Run Horizontal FL Demo
Open two new terminals, run respectively:

```bash
sudo bash start-fl-client.sh hfl 1
```

```bash
sudo bash start-fl-client.sh hfl 2
```

## Run Vertical FL Demo
Open two new terminals, run respectively:

```bash
sudo bash start-fl-client.sh vfl 1
```

```bash
sudo bash start-fl-client.sh vfl 2
```