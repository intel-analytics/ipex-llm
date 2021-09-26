# Trusted Big Data ML
SGX-based Trusted Big Data ML allows the user to run end-to-end big data analytics application and Intel Analytics Zoo and BigDL model training with spark local and distributed cluster on Graphene-SGX.

*Please mind the IP and file path settings. They should be changed to the IP/path of your own sgx server on which you are running the programs.*

## How To Build
Before running the following command, please modify the paths in `build-docker-image.sh`. <br>
Then build the docker image by running this command: <br>
```bash
./build-docker-image.sh
```

## How to Run

### Prerequisite
To launch Trusted Big Data ML applications on Graphene-SGX, you need to install graphene-sgx-driver:
```bash
sudo bash ../../../scripts/install-graphene-driver.sh
```

### Prepare the data
To train a model with ppml in analytics zoo and bigdl, you need to prepare the data first. The Docker image is taking lenet and mnist as example. <br>
You can download the MNIST Data from [here](http://yann.lecun.com/exdb/mnist/). Unzip all the files and put them in one folder(e.g. mnist). <br>
There are four files. **train-images-idx3-ubyte** contains train images, **train-labels-idx1-ubyte** is train label file, **t10k-images-idx3-ubyte** has validation images and **t10k-labels-idx1-ubyte** contains validation labels. For more detail, please refer to the download page. <br>
After you decompress the gzip files, these files may be renamed by some decompress tools, e.g. **train-images-idx3-ubyte** is renamed to **train-images.idx3-ubyte**. Please change the name back before you run the example.  <br>

### Prepare the keys
The ppml in analytics zoo needs secured keys to enable spark security such as Authentication, RPC Encryption, Local Storage Encryption and TLS, you need to prepare the secure keys and keystores. In this tutorial, you can generate keys and keystores with root permission (test only, need input security password for keys).

```bash
sudo ../../../scripts/generate-keys.sh
```

You also need to generate your enclave key using the command below, and keep it safely for future remote attestations and to start SGX enclaves more securely.
It will generate a file `enclave-key.pem` in your present working directory, which will be your enclave key. To store the key elsewhere, modify the outputted file path.
```bash
openssl genrsa -3 -out enclave-key.pem 3072
```
### Prepare the password
Next, you need to store the password you used for key generation, i.e., `generate-keys.sh`, in a secured file:

```bash
sudo bash ../../../scripts/generate-password.sh used_password_when_generate_keys
```

### Run the PPML as Docker containers

#### In spark local mode
##### Start the container to run spark applications in ppml
Before you run the following commands to start the container, you need to modify the paths in `./deploy-local-spark-sgx.sh`. <br>
Then run the following commands: <br>
```bash
./deploy-local-spark-sgx.sh
sudo docker exec -it spark-local bash
cd /ppml/trusted-big-data-ml
```

##### Example 1: Spark PI on Graphene-SGX
```bash
./init.sh
vim start-spark-local-pi-sgx.sh
```
If you run ./init.sh meeting failure, please try to exit the container and then follow the commands of [docs of PPML](https://github.com/intel-analytics/analytics-zoo/blob/master/docs/readthedocs/source/doc/PPML/Overview/ppml.md#21-prerequisite), and recreated the keys and password directory following those commands. Then modify those PATHs in the ./deploy-local-spark-sgx.sh, and rerun the ./deploy-local-spark-sgx.sh to build one new container. <br>
Add these code in `start-spark-local-pi-sgx.sh`: <br>
```bash
#!/bin/bash

set -x

SGX=1 ./pal_loader /opt/jdk8/bin/java \
        -cp '/ppml/trusted-big-data-ml/work/spark-2.4.6/examples/jars/spark-examples_2.11-2.4.6.jar:/ppml/trusted-big-data-ml/work/bigdl-jar-with-dependencies.jar:/ppml/trusted-big-data-ml/work/spark-2.4.6/conf/:/ppml/trusted-big-data-ml/work/spark-2.4.6/jars/*' \
        -Xmx10g \
        -Dbigdl.mklNumThreads=1 \
        -XX:ActiveProcessorCount=24 \
        org.apache.spark.deploy.SparkSubmit \
        --master 'local[4]' \
        --conf spark.driver.port=10027 \
        --conf spark.scheduler.maxRegisteredResourcesWaitingTime=5000000 \
        --conf spark.worker.timeout=600 \
        --conf spark.starvation.timeout=250000 \
        --conf spark.rpc.askTimeout=600 \
        --conf spark.blockManager.port=10025 \
        --conf spark.driver.host=127.0.0.1 \
        --conf spark.driver.blockManager.port=10026 \
        --conf spark.io.compression.codec=lz4 \
        --class org.apache.spark.examples.SparkPi \
        --executor-cores 4 \
        --total-executor-cores 4 \
        --executor-memory 10G \
        /ppml/trusted-big-data-ml/work/spark-2.4.6/examples/jars/spark-examples_2.11-2.4.6.jar | tee spark.local.pi.sgx.log
```

Then run the script to run pi test in spark: <br>
```bash
sh start-spark-local-pi-sgx.sh
```

Open another terminal and check the log:
```bash
sudo docker exec -it spark-local cat /ppml/trusted-big-data-ml/spark.local.pi.sgx.log | egrep "###|INFO|Pi"
```

The result should look like: <br>
>   Pi is roughly 3.1422957114785572

##### Example 2: Analytics Zoo model training on Graphene-SGX
```bash
./init.sh
./start-spark-local-train-sgx.sh
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

##### Example Test 3
Before run TPC-H test in container we created, we should download and install [SBT](https://www.scala-sbt.org/download.html) and deploy a [HDFS](https://hadoop.apache.org/docs/r1.2.1/) for TPC-H output, then build and package TPC-H dataset according to [TPC-H](https://github.com/qiuxin2012/tpch-spark) with your needs. After packaged, check if we have `spark-tpc-h-queries_2.11-1.0.jar ` under `/tpch-spark/target/scala-2.11`, if have, we package successfully.


Copy TPC-H to container: <br>
```bash
docker cp tpch-spark/ spark-local:/ppml/trusted-big-data-ml/work
sudo docker exec -it spark-local bash
cd ppml/trusted-big-data-ml/
./init.sh
vim start-spark-local-tpc-h-sgx.sh
```

Please modify HDFS_NAMENODE_IP in this script and then add these code in the `start-spark-local-tpc-h-sgx.sh` file: <br>
```bash
#!/bin/bash

set -x

SGX=1 ./pal_loader /opt/jdk8/bin/java \
        -cp '/ppml/trusted-big-data-ml/work/tpch-spark/target/scala-2.11/spark-tpc-h-queries_2.11-1.0.jar:/ppml/trusted-big-data-ml/work/tpch-spark/dbgen/*:/ppml/trusted-big-data-ml/work/bigdl-jar-with-dependencies.jar:/ppml/trusted-big-data-ml/work/spark-2.4.6/conf/:/ppml/trusted-big-data-ml/work/spark-2.4.6/jars/*' \
        -Xmx10g \
        -Dbigdl.mklNumThreads=1 \
        -XX:ActiveProcessorCount=24 \
        org.apache.spark.deploy.SparkSubmit \
        --master 'local[4]' \
        --conf spark.driver.port=10027 \
        --conf spark.scheduler.maxRegisteredResourcesWaitingTime=5000000 \
        --conf spark.worker.timeout=600 \
        --conf spark.starvation.timeout=250000 \
        --conf spark.rpc.askTimeout=600 \
        --conf spark.blockManager.port=10025 \
        --conf spark.driver.host=127.0.0.1 \
        --conf spark.driver.blockManager.port=10026 \
        --conf spark.io.compression.codec=lz4 \
        --conf spark.sql.shuffle.partitions=8 \
        --class main.scala.TpchQuery \
        --executor-cores 4 \
        --total-executor-cores 4 \
        --executor-memory 10G \
        /ppml/trusted-big-data-ml/work/tpch-spark/target/scala-2.11/spark-tpc-h-queries_2.11-1.0.jar \
        hdfs://HDFS_NAMENODE_IP:8020/dbgen hdfs://HDFS_NAMENODE_IP:8020/tmp/output | tee spark.local.tpc.h.sgx.log
```

Then run the script to run TPC-H test in spark: <br>
```bash
sh start-spark-local-tpc-h-sgx.sh
```

Open another terminal and check the log: <br>
```bash
sudo docker exec -it spark-local cat /ppml/trusted-big-data-ml/spark.local.tpc.h.sgx.log | egrep "###|INFO|finished"
```

The result should look like: <br>
>   ----------------22 finished--------------------

##### Other Spark workloads are also supported, please follow the 3 examples to submit your workload with spark on Graphene-SGX


#### In spark standalone cluster mode

Pay attention to the filenames here. They can be quite confusing.

##### Setup passwordless ssh login to all the nodes.
##### Configure the environments for master, workers, docker image, security keys/password files, enclave key, and data path.
```bash
nano environment.sh
```
##### Start distributed big data ML
To start the Spark services for distributed big data ML, run
```bash
./deploy-distributed-standalone-spark.sh
```

Then run the following command to start the training:
```bash
./start-distributed-spark-train-sgx.sh
```

##### Stop distributed big data ML
When stopping distributed big data ML, stop the training first:
```bash
./stop-distributed-standalone-spark.sh
```
Then stop the spark services:
```bash
./undeploy-distributed-standalone-spark.sh
```

##### Other Spark workloads are also supported, please follow the 3 examples to submit your workload with spark on Graphene-SGX

Note that in the distributed scenario, you need to run them in the container named `spark-driver` instead of `spark-local` for these examples to work.

##### Troubleshooting
You can run the script `sudo bash distributed-check-status.sh` after starting distributed cluster serving to check whether the components have been correctly started.

To test a specific component, pass one or more argument to it among the following:
"master", and "worker". For example, run the following command to check the status of the Spark job master.

```bash
./distributed-check-status.sh master
```

To test all components, you can either pass no argument or pass the "all" argument.

```bash
./distributed-check-status.sh
```
If all is well, the following results should be displayed:

```
(1/2) Detecting Master state...
Master initialization successful.
(2/2) Detecting Worker state...
Worker initialization successful.
```

It is suggested to run this script once after starting distributed cluster serving to verify that all components are up and running.
