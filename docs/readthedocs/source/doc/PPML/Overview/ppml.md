# PPML User Guide

## 1. Privacy Preserving Machine Learning
Protecting privacy and confidentiality is critical for large-scale data analysis and machine learning. Analytics Zoo ***PPML*** combines various low level hardware and software security technologies (e.g., Intel SGX, LibOS such as Graphene and Occlum, Federated Learning, etc.), so that users can continue to apply standard Big Data and AI technologies (such as Apache Spark, Apache Flink, Tensorflow, PyTorch, etc.) without sacrificing privacy.

## 1.1 PPML for Big Data AI
Analytics Zoo provides a distributed PPML platform for protecting the *end-to-end Big Data AI pipeline* (from data ingestion, data analysis, all the way to machine learning and deep learning). In particular, it extends the single-node [Trusted Execution Environment](https://en.wikipedia.org/wiki/Trusted_execution_environment) to provide a *Trusted Cluster Environment*, so as to run unmodified Big Data analysis and ML/DL programs in a secure fashion on (private or public) cloud:

 * Compute and memory protected by SGX Enclaves
 * Network communication protected by remote attestation and TLS
 * Storage (e.g., data and model) protected by encryption
 * Optional federated learning support

That is, even when the program runs in an untrusted cloud environment, all the data and models are protected (e.g., using encryption) on disk and network, and the compute and memory are also protected using SGX Enclaves, so as to preserve the confidentiality and privacy during data analysis and machine learning.

In the current release, two types of trusted Big Data AI applications are supported:

1. Big Data analytics and ML/DL (supporting [Apache Spark](https://spark.apache.org/) and [BigDL](https://github.com/intel-analytics/BigDL))
2. Realtime compute and ML/DL (supporting [Apache Flink](https://flink.apache.org/) and Analytics Zoo [Cluster Serving](https://www.usenix.org/conference/opml20/presentation/song))

## 2. Trusted Big Data Analytics and ML
With the trusted Big Data analytics and ML/DL support, users can run standard Spark data analysis (such as Spark SQL, Dataframe, MLlib, etc.) and distributed deep learning (using BigDL) in a secure and trusted fashion.

### 2.1 Prerequisite

Download scripts and dockerfiles from [this link](https://github.com/intel-analytics/analytics-zoo/tree/master/ppml).

1. Install SGX Driver

    Please check if the current HW processor supports [SGX](https://www.intel.com/content/www/us/en/support/articles/000028173/processors/intel-core-processors.html). Then, enable SGX feature in BIOS. Note that after SGX is enabled, a portion of memory will be assigned to SGX (this memory cannot be seen/used by OS and other applications).

    Check SGX driver with `ls /dev | grep sgx`. If SGX driver is not installed, please install [SGX DCAP driver](https://github.com/intel/SGXDataCenterAttestationPrimitives/tree/master/driver/linux):

    ```bash
    ./ppml/scripts/install-graphene-driver.sh
    ```

2. Generate key for SGX enclave

   Generate the enclave key using the command below, and keep it safely for future remote attestations and to start SGX enclaves more securely. It will generate a file `enclave-key.pem` in the current working directory, which will be the  enclave key. To store the key elsewhere, modify the output file path.

    ```bash
    openssl genrsa -3 -out enclave-key.pem 3072
    ```

3. Prepare keys for TLS with root permission (test only, need input security password for keys).

    ```bash
    sudo ./ppml/scripts/generate-keys.sh
    ```

    This scrips will generate 5 files in `keys` dir (you can replace them with your own TLS keys).

    ```bash
    keystore.pkcs12
    server.crt
    server.csr
    server.key
    server.pem
    ```

4. Generate `password` to avoid plain text security password (used for key generation in `generate-keys.sh`) transfer.

    ```bash
    ./ppml/scripts/generate-password.sh used_password_when_generate_keys
    ```
    This scrips will generate 2 files in `password` dir.

    ```bash
    key.txt
    output.bin
    ```
### 2.2 Prepare Docker Container

Pull docker image from Dockerhub
```bash
docker pull intelanalytics/analytics-zoo-ppml-trusted-big-data-ml-scala-graphene:0.10-SNAPSHOT
```

Alternatively, you can build docker image from Dockerfile (this will take some time):

```bash
cd ppml/trusted-big-data-ml/scala/docker-graphene
./build-docker-image.sh
```

### 2.3 Run Trusted Big Data and ML on Single Node

#### 2.3.1 Start PPML Container

Enter `analytics-zoo/ppml/trusted-big-data-ml/scala/docker-graphene` dir.

1. Copy `keys` and `password`
    ```bash
    cd ppml/trusted-big-data-ml/scala/docker-graphene
    # copy keys and password into current directory
    cp -r ../keys .
    cp -r ../password .
    ```
2. To start the container, first modify the paths in deploy-local-spark-sgx.sh, and then run the following commands:
    ```bash
    ./deploy-local-spark-sgx.sh
    sudo docker exec -it spark-local bash
    cd /ppml/trusted-big-data-ml
    ./init.sh
    ```
    
#### 2.3.2 Run Trusted Spark Pi

This example runs a simple Spark PI program, which is an  easy way to verify if the Trusted PPML environment is ready.  

Run the script to run trusted Spark Pi:

```bash
bash start-spark-local-pi-sgx.sh
```

Open another terminal and check the log:

```bash
sudo docker exec -it spark-local cat /ppml/trusted-big-data-ml/spark.local.pi.sgx.log | egrep "###|INFO|Pi"
```

The result should look something like:

>   Pi is roughly 3.1422957114785572

#### 2.3.3 Run Trusted Spark SQL

This example shows how to run trusted Spark SQL (e.g.,  TPC-H queries).

First, download and install [SBT](https://www.scala-sbt.org/download.html) and deploy a [HDFS](https://hadoop.apache.org/docs/r2.7.7/hadoop-project-dist/hadoop-common/ClusterSetup.html) for TPC-H dataset and output, then build the source codes with SBT and generate TPC-H dataset according to the [TPC-H example](https://github.com/intel-analytics/zoo-tutorials/tree/master/tpch-spark). After that, check if there is an  `spark-tpc-h-queries_2.11-1.0.jar` under `tpch-spark/target/scala-2.11`; if so, we have successfully packaged the project.

Copy the TPC-H package to container:

```bash
docker cp tpch-spark/ spark-local:/ppml/trusted-big-data-ml/work
docker cp tpch-spark/start-spark-local-tpc-h-sgx.sh spark-local:/ppml/trusted-big-data-ml/
sudo docker exec -it spark-local bash
cd /ppml/trusted-big-data-ml/
```
Then run the script below:

```bash
sh start-spark-local-tpc-h-sgx.sh [your_hdfs_tpch_data_dir] [your_hdfs_output_dir]
```

Open another terminal and check the log:

```bash
sudo docker exec -it spark-local cat /ppml/trusted-big-data-ml/spark.local.tpc.h.sgx.log | egrep "###|INFO|finished"
```

The result should look like:

>   ----------------22 finished--------------------

#### 2.3.4 Run Trusted Deep Learning

This example shows how to run trusted deep learning (using an BigDL LetNet program).

First, download the MNIST Data from [here](http://yann.lecun.com/exdb/mnist/). Use `gzip -d` to unzip all the downloaded files (train-images-idx3-ubyte.gz, train-labels-idx1-ubyte.gz, t10k-images-idx3-ubyte.gz, t10k-labels-idx1-ubyte.gz) and put them into folder `/ppml/trusted-big-data-ml/work/data`.

Then run the following script:  

```bash
bash start-spark-local-train-sgx.sh
```

Open another terminal and check the log:
```bash
sudo docker exec -it spark-local cat /ppml/trusted-big-data-ml/spark.local.sgx.log | egrep "###|INFO"
```
or
```bash
sudo docker logs spark-local | egrep "###|INFO"
```

The result should look like:

```bash
############# train optimized[P1182:T2:java] ---- end time: 310534 ms return from shim_write(...) = 0x1d
############# ModuleLoader.saveToFile File.saveBytes end, used 827002 ms[P1182:T2:java] ---- end time: 1142754 ms return from shim_write(...) = 0x48
############# ModuleLoader.saveToFile saveWeightsToFile end, used 842543 ms[P1182:T2:java] ---- end time: 1985297 ms return from shim_write(...) = 0x4b
############# model saved[P1182:T2:java] ---- end time: 1985297 ms return from shim_write(...) = 0x19
```

### 2.4 Run Trusted Big Data and ML on Cluster
#### 2.4.1 Configure the Environment

Prerequisite: passwordless ssh login to all the nodes needs to be properly set up first.

```bash
nano environments.sh
```
#### 2.4.2 Start Distributed Big Data and ML Platform

First run the following command to start the service:

```bash
./deploy-distributed-standalone-spark.sh
```

Then run the following command to start the training:

```bash
./start-distributed-spark-train-sgx.sh
```
#### 2.4.3  Stop Distributed Big Data and ML Platform

First, stop the training:

```bash
./stop-distributed-standalone-spark.sh
```

Then stop the service:

```bash
./undeploy-distributed-standalone-spark.sh
```

## 3. Trusted Realtime Compute and ML
With the trusted realtime compute and ML/DL support, users can run standard Flink stream processing and distributed DL model inference (using [Cluster Serving](https://www.usenix.org/conference/opml20/presentation/song)) in a secure and trusted fashion. In this feature, both [Graphene](https://github.com/oscarlab/graphene) and [Occlum](https://github.com/occlum/occlum) are supported, users can choose one of them as LibOS layer.

### 3.1 Prerequisite

Please refer to [Section 2.1 Prerequisite](#prerequisite). For Occlum backend, if your kernel version is below 5.11, please install [enable_rdfsbase](https://github.com/occlum/enable_rdfsbase).

### 3.2 Prepare Docker Container

Pull docker image from Dockerhub

```bash
# For Graphene
docker pull intelanalytics/analytics-zoo-ppml-trusted-realtime-ml-scala-graphene:0.10-SNAPSHOT
```

```bash
# For Occlum
docker pull intelanalytics/analytics-zoo-ppml-trusted-realtime-ml-scala-occlum:0.10-SNAPSHOT
```

Also, you can build docker image from Dockerfile (this will take some time).

```bash
# For Graphene
cd ppml/trusted-realtime-ml/scala/docker-graphene
./build-docker-image.sh
```

```bash
# For Occlum
cd ppml/trusted-realtime-ml/scala/docker-occlum
./build-docker-image.sh
```

### 3.3 Run Trusted Realtime Compute and ML

#### 3.3.1 Configure the Environment

Enter `analytics-zoo/ppml/trusted-realtime-ml/scala/docker-graphene` or `analytics-zoo/ppml/trusted-realtime-ml/scala/docker-occlum` dir.

Modify `environments.sh`. Change MASTER, WORKER IP and file paths (e.g., `keys` and `password`).

```bash
nano environments.sh
```

#### 3.3.2 Start the service

Start Flink service:

```bash
./deploy-flink.sh
```

#### 3.3.3 Run Trusted Flink Program

Submit Flink jobs:

```bash
cd ${FLINK_HOME}
./bin/flink run ./examples/batch/WordCount.jar
```

If Jobmanager is not running on current node, please add `-m ${FLINK_JOB_MANAGER_IP}`.

The result should look like:

```bash
(a,5)    
(action,1) 
(after,1)
(against,1)  
(all,2) 
(and,12) 
(arms,1)   
(arrows,1)  
(awry,1)   
(ay,1)    
(bare,1)  
(be,4)      
(bear,3)      
(bodkin,1) 
(bourn,1)  
```
#### 3.3.4 Run Trusted Cluster Serving

Start Cluster Serving as follows:

```bash
./start-local-cluster-serving.sh
```

After all services are ready, you can directly push inference requests int queue with [Restful API](https://analytics-zoo.github.io/master/#ClusterServingGuide/ProgrammingGuide/#restful-api). Also, you can push image/input into queue with Python API

```python
from zoo.serving.client import InputQueue
input_api = InputQueue()
input_api.enqueue('my-image1', user_define_key={"path: 'path/to/image1'})
```

Cluster Serving service is a long running service in container, you can stop it as follows:

```bash
docker stop trusted-cluster-serving-local
```
