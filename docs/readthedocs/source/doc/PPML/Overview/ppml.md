# PPML (Privacy Preserving Machine Learning)

PPML (Privacy-Preserving Machine Learning) aims at protecting user privacy, meanwhile keeping machine learning applications still useful. However, achieving this goal without impacting existing applications is quite difficult, especially in end-to-end big data scenarios. To resolve this problem, Analytics-Zoo provides an end-to-end PPML platform for Big Data AI based on Intel SGX (Software Guard Extensions). This PPML platform ensures the whole Big Data & AI pipeline are fully protected by secured SGX enclave in hardware level, further more existing Big Data & AI applications, such as Flink, Spark, SparkSQL and machine/deep learning, can be seamlessly migrated into this PPML platform without any code changes.

## PPML for Big Data AI

To take full advantage of big data, especially the value of private or sensitive data, customers need to build a trusted platform under the guidance of privacy laws or regulation, such as [GDPR](https://gdpr-info.eu/) and CCPA (https://oag.ca.gov/privacy/ccpa). This requirement raises big challenges to customers who already have big data and big data applications, such as Spark/SparkSQL, Flink/FlinkSQL and AI applications. Migrating these applications into privacy preserving way requires lots of additional efforts.

With Analytics-Zoo, customers/developers can build a Trusted Platform for big data with a few clicks, and all existing big data & AI applications, such as Flink and Spark applications, can be migrated into this platform without any code changes. In specific, Analytics-Zoo uses serval security technologies

- Confidential Computation with Intel SGX. Intel SGX provides hardware-based isolation and memory encryption with very limited attack surface.
- Seamless migration with LibOS. Based on LibOS projects ([Graphene](https://grapheneproject.io/) and [Occlum](https://occlum.io/)), Analytics-Zoo empowers our customers (e.g., data scientists and big data developers) to build PPML applications on top of large scale dataset without impacting existing applications.
- Secured networks with TLS and encryption. All network traffic are protected by TLS, in some cases, content should be encrypted before transformation.
- File or model protection with encryption. Model and sensitive configuration files will be encrypted before uploading to Trusted platform. These files are only decrypted in SGX enclave.
- Environment & App attestation with SGX attestation. SGX attestation ensures that remote/local SGX env and applications can be verified.

Note: Intel SGX requires hardware support, please [check if your CPU has this feature](https://www.intel.com/content/www/us/en/support/articles/000028173/processors/intel-core-processors.html). In [3rd Gen Intel Xeon Scalable Processors](https://newsroom.intel.com/press-kits/3rd-gen-intel-xeon-scalable/), SGX allows up to 1TB of data to be included in secure enclaves.

### Key features

- Protecting data and model confidentiality
  - Sensitive input/output data (computation, training and inference), e.g., healthcare data
  - Proprietary model, e.g., model trained with self-owned or sensitive data
- Seamless migrate existing big data applications into privacy preserving applications
- Trusted big data & AI Platform based on Intel SGX
  - Trusted Big Data Analytics and ML: Spark batch, SparkSQL, BigDL, TPC-H
  - Trusted Realtime Compute and ML: Flink batch/streaming, Cluster Serving

## Trusted Big Data Analytics and ML

In this section, we will demonstrate how to use Analytics-Zoo to setup trusted Spark in SGX, then run Spark PI example in safe way. Trusted Big Data Analytics and ML supports Spark related applications, such as Spark batch jobs, SparkSQL and BigDL etc. For more examples, please refer to [trusted-big-data-ml](https://github.com/intel-analytics/analytics-zoo/tree/master/ppml/trusted-big-data-ml/scala/docker-graphene).

### Use cases

- Big Data analysis using Spark (Spark SQL, Dataframe, MLlib, etc.)
- Distributed deep learning using BigDL

### Get started

#### Prerequisite: Install SGX Driver & Prepare Scripts

Please check if current platform [has SGX feature](https://www.intel.com/content/www/us/en/support/articles/000028173/processors/intel-core-processors.html). Then, enable SGX feature in BIOS. Note that after SGX is enabled, a portion of memory will be assigned to SGX (this memory cannot be seen/used by OS and other applications).

Check SGX driver with `ls /dev | grep sgx`. If SGX driver is not installed, please install [SGX DCAP driver](https://github.com/intel/SGXDataCenterAttestationPrimitives/tree/master/driver/linux) with [install-graphene-driver.sh](https://github.com/intel-analytics/analytics-zoo/blob/master/ppml/scripts/install-graphene-driver.sh) (need root permission).

```bash
./ppml/scripts/install-graphene-driver.sh
```

#### Step 0: Prepare Environment

Download scripts and dockerfiles in [this link](https://github.com/intel-analytics/analytics-zoo/tree/master/ppml).

##### TLS keys & password & Enclave key

Prepare keys for TLS (test only, need input security password for keys).

```bash
./ppml/scripts/generate-keys.sh
```

This scrips will generate 5 files in `keys` dir (you can replace them with your own TLS keys).

```bash
keystore.pkcs12
server.crt
server.csr
server.key
server.pem
```

Generated `password` to avoid plain text security password transfer.

```bash
./ppml/scripts/generate-password.sh
```

This scrips will generate 2 files in `password` dir.

```bash
key.txt
output.bin
```

You also need to generate your enclave key using the command below, and keep it safely for future remote attestations and to start SGX enclaves more securely. It will generate a file enclave-key.pem in your present working directory, which will be your enclave key. To store the key elsewhere, modify the outputted file path.

```bash
openssl genrsa -3 -out enclave-key.pem 3072
```

##### Docker

Pull docker image from Dockerhub

```bash
docker pull intelanalytics/analytics-zoo-ppml-trusted-realtime-ml-scala-graphene:0.10-SNAPSHOT
```

Also, you can build docker image from Dockerfile (this will take some time).

```bash
cd ppml/trusted-big-data-ml/scala/docker-graphene
./build-docker-image.sh
```

#### Single-Node Trusted Big Data Analytics and ML Platform

Enter `analytics-zoo/ppml/trusted-big-data-ml/scala/docker-graphene` dir. Start Spark service with this command

Prepare `keys` and `password`
```bash
cd ppml/trusted-big-data-ml/scala/docker-graphene
# copy keys and password into current directory
cp -r ../keys .
cp -r ../password .
```

Before you run the following commands to start the container, you need to modify the paths in deploy-local-big-data-ml.sh.
Then run the following commands:
```bash
./deploy-local-big-data-ml.sh
sudo docker exec -it spark-local bash
cd /ppml/trusted-big-data-ml
./init.sh
```

##### **Example 1: Spark Pi on Graphene-SGX**

This example is a simple Spark local PI example, this a very easy way to verify if your SGX environment is ready.  
Run the script to run pi test in spark:

```bash
bash start-spark-local-pi-sgx.sh
```

Open another terminal and check the log:

```bash
sudo docker exec -it spark-local cat /ppml/trusted-big-data-ml/spark.local.pi.sgx.log | egrep "###|INFO|Pi"
```

The result should look like:

>   Pi is roughly 3.1422957114785572

##### **Example 2: BigDL model training on Graphene-SGX**

This example is about how to train a lenet model using BigDL on Graphene-SGX. Before you run the following script, you should download the MNIST Data from [here](http://yann.lecun.com/exdb/mnist/). Use `gzip -d` to unzip all the downloaded files(train-images-idx3-ubyte.gz, train-labels-idx1-ubyte.gz, t10k-images-idx3-ubyte.gz, t10k-labels-idx1-ubyte.gz) and put them into folder `/ppml/trusted-big-data-ml/work/data`. Then run the following script:  

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

The result should look like: <br>
>   ############# train optimized[P1182:T2:java] ---- end time: 310534 ms return from shim_write(...) = 0x1d <br>
>   ############# ModuleLoader.saveToFile File.saveBytes end, used 827002 ms[P1182:T2:java] ---- end time: 1142754 ms return from shim_write(...) = 0x48 <br>
>   ############# ModuleLoader.saveToFile saveWeightsToFile end, used 842543 ms[P1182:T2:java] ---- end time: 1985297 ms return from shim_write(...) = 0x4b <br>
>   ############# model saved[P1182:T2:java] ---- end time: 1985297 ms return from shim_write(...) = 0x19 <br>

##### **Example 3: Spark-SQL on Graphene-SGX** 

Before run TPC-H test in container we created, we should download and install [SBT](https://www.scala-sbt.org/download.html) and deploy a [HDFS](https://hadoop.apache.org/docs/r2.7.7/hadoop-project-dist/hadoop-common/ClusterSetup.html) for TPC-H dataset and output, then build the source codes with SBT and generate TPC-H dataset according to [TPC-H](https://github.com/qiuxin2012/tpch-spark). After packaged, check if we have `spark-tpc-h-queries_2.11-1.0.jar ` under `tpch-spark/target/scala-2.11`, if have, we package successfully.

Copy TPC-H to container: <br>
```bash
docker cp tpch-spark/ spark-local:/ppml/trusted-big-data-ml/work
docker cp tpch-spark/start-spark-local-tpc-h-sgx.sh spark-local:/ppml/trusted-big-data-ml/
sudo docker exec -it spark-local bash
cd /ppml/trusted-big-data-ml/
```

Then run the script to run TPC-H test in spark: <br>
```bash
sh start-spark-local-tpc-h-sgx.sh [your_hdfs_tpch_data_dir] [your_hdfs_output_dir]
```

Open another terminal and check the log: <br>
```bash
sudo docker exec -it spark-local cat /ppml/trusted-big-data-ml/spark.local.tpc.h.sgx.log | egrep "###|INFO|finished"
```

The result should look like: <br>
>   ----------------22 finished--------------------

#### Distributed Trusted Big Data Analytics and ML Platform

##### **Step 1: Configure the environments for master, workers, docker image, security keys/password files, enclave key, and data path.**
Requirement: setup passwordless ssh login to all the nodes.
```bash
nano environments.sh
```
##### **Step 2: Start distributed big data ML**
To start the Spark services for distributed big data ML, run
```bash
./deploy-distributed-standalone-spark.sh
```

Then run the following command to start the training:
```bash
./start-distributed-spark-train-sgx.sh
```

##### **Step 3: Stop distributed big data ML**
When stopping distributed big data ML, stop the training first:
```bash
./stop-distributed-standalone-spark.sh
```
Then stop the spark services:
```bash
./undeploy-distributed-standalone-spark.sh
```

-----------------------------------------------

## Trusted Realtime Compute and ML

In this section, we will demonstrate how to use Analytics-Zoo to setup trusted Flink in SGX, then run real-time model serving in safe way. Trusted Realtime Compute and ML supports Flink related applications, such as Batch/Streaming Flink jobs, FlinkSQL and Cluster Serving. For more examples, please refer to [trusted-realtime-ml](https://github.com/intel-analytics/analytics-zoo/tree/master/ppml/trusted-realtime-ml/scala/docker-graphene).

### User cases

- Real time data computation/analytics using Flink
- Distributed end-to-end serving solution with Cluster Serving

### Get started

#### [Prerequisite: Install SGX Driver & Prepare Scripts](#prerequisite-install-sgx-driver--prepare-scripts)

#### Step 0: Prepare Environment

Download scripts and dockerfiles in [this link](https://github.com/intel-analytics/analytics-zoo/tree/master/ppml).

##### TLS keys & password

Prepare keys for TLS (test only, need input security password for keys).

```bash
./ppml/scripts/generate-keys.sh
```

This scrips will generate 5 files in `keys` dir (you can replace them with your own TLS keys).

```bash
keystore.pkcs12
server.crt
server.csr
server.key
server.pem
```

Generated `password` to avoid plain text security password transfer.

```bash
./ppml/scripts/generate-password.sh
```

This scrips will generate 2 files in `password` dir.

```bash
key.txt
output.bin
```

##### Docker

Pull docker image from Dockerhub

```bash
docker pull intelanalytics/analytics-zoo-ppml-trusted-realtime-ml-scala-graphene:0.10-SNAPSHOT
```

Also, you can build docker image from Dockerfile (this will take some time).

```bash
cd ppml/trusted-big-data-ml/scala/docker-graphene
./build-docker-image.sh
```

#### Step 1: Start  Trusted Realtime Compute and ML Platform

Enter `analytics-zoo/ppml/trusted-realtime-ml/scala/docker-graphene` dir.

Modify `environments.sh`. Change MASTER, WORKER IP and file paths (e.g., `keys` and `password`).

```bash
nano environments.sh
```

Start Flink in SGX

```bash
./deploy-flink.sh
```

After all jobs are done, stop Flink in SGX

```bash
./stop-flink.sh
```

#### Step 2: Run Flink Program

If working env has Flink, then submit jobs to Flink

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

#### Step 3: Run Trusted Cluster Serving

[Analytics-Zoo Cluster serving](https://www.usenix.org/conference/opml20/presentation/song) is a distributed end-to-end inference service based on Flink. Now this feature is available in PPML solution, while all input data and model in inference pipeline are fully protected.

Start Cluster Serving Service

```bash
./start-local-cluster-serving.sh
```

After all services are ready, you can directly push inference requests int queue with [Restful API](https://analytics-zoo.github.io/master/#ClusterServingGuide/ProgrammingGuide/#restful-api). Also, you can push image/input into queue with Python API

```python
from zoo.serving.client import InputQueue
input_api = InputQueue()
input_api.enqueue('my-image1', user_define_key={"path: 'path/to/image1'})
```

Cluster Serving service is a long running service in container, you can stop it with

```bash
docker stop trusted-cluster-servinglocal
```

For distributed/multi-container, please refer to [Distributed Trusted Cluster Serving](https://github.com/intel-analytics/analytics-zoo/tree/master/ppml/trusted-realtime-ml/scala/docker-graphene#in-distributed-mode)
