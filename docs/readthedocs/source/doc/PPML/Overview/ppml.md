# PPML User Guide

## 1. Privacy Preserving Machine Learning
Protecting privacy and confidentiality is critical for large-scale data analysis and machine learning. Analytics Zoo ***PPML*** combines various low level hardware and software security technologies (e.g., Intel SGX, LibOS such as Graphene and Occlum, Federated Learning, etc.), so that users can continue to apply standard Big Data and AI technologies (such as Apache Spark, Apache Flink, Tensorflow, PyTorch, etc.) without sacrificing privacy.

## 1.1 PPML for Big Data AI
Analytics Zoo/BigDL provides a distributed PPML platform for protecting the *end-to-end Big Data AI pipeline* (from data ingestion, data analysis, all the way to machine learning and deep learning). In particular, it extends the single-node [Trusted Execution Environment](https://en.wikipedia.org/wiki/Trusted_execution_environment) to provide a *Trusted Cluster Environment*, so as to run unmodified Big Data analysis and ML/DL programs in a secure fashion on (private or public) cloud:

 * Compute and memory protected by SGX Enclaves
 * Network communication protected by remote attestation and TLS
 * Storage (e.g., data and model) protected by encryption
 * Optional federated learning support

That is, even when the program runs in an untrusted cloud environment, all the data and models are protected (e.g., using encryption) on disk and network, and the compute and memory are also protected using SGX Enclaves, so as to preserve the confidentiality and privacy during data analysis and machine learning.

In the current release, two types of trusted Big Data AI applications are supported:

1. Big Data analytics and ML/DL (supporting [Apache Spark](https://spark.apache.org/) and [BigDL](https://github.com/intel-analytics/BigDL))
2. Realtime compute and ML/DL (supporting [Apache Flink](https://flink.apache.org/) and BigDL [Cluster Serving](https://www.usenix.org/conference/opml20/presentation/song))

## 2. Trusted Big Data Analytics and ML
With the trusted Big Data analytics and ML/DL support, users can run standard Spark data analysis (such as Spark SQL, Dataframe, MLlib, etc.) and distributed deep learning (using BigDL) in a secure and trusted fashion.

### 2.1 Prerequisite

Download scripts and dockerfiles from [this link](https://github.com/intel-analytics/analytics-zoo). And do the following commands:
```bash
cd analytics-zoo/ppml/
```

1. Install SGX Driver

    Please check if the current HW processor supports [SGX](https://www.intel.com/content/www/us/en/support/articles/000028173/processors/intel-core-processors.html). Then, enable SGX feature in BIOS. Note that after SGX is enabled, a portion of memory will be assigned to SGX (this memory cannot be seen/used by OS and other applications).

    Check SGX driver with `ls /dev | grep sgx`. If SGX driver is not installed, please install [SGX DCAP driver](https://github.com/intel/SGXDataCenterAttestationPrimitives/tree/master/driver/linux):

    ```bash
    cd scripts/
    ./install-graphene-driver.sh
    cd ..
    ```

2. Generate key for SGX enclave

   Generate the enclave key using the command below, and keep it safely for future remote attestations and to start SGX enclaves more securely. It will generate a file `enclave-key.pem` in the current working directory, which will be the  enclave key. To store the key elsewhere, modify the output file path.

    ```bash
    cd scripts/
    openssl genrsa -3 -out enclave-key.pem 3072
    cd ..
    ```

3. Prepare keys for TLS with root permission (test only, need input security password for keys). Please also install jdk/openjdk and set the environment path of java path to get keytool.

    ```bash
    cd scripts/
    ./generate-keys.sh
    cd ..
    ```
    When entering pass phrase or password, you could input the same password by yourself; and these passwords could also be used for the next step of generating password. Password should be longer than 6 bits and containing number and letter, and one sample password is "3456abcd". These passwords would be used for future remote attestations and to start SGX enclaves more securely. And This scripts will generate 6 files in `./ppml/scripts/keys` dir (you can replace them with your own TLS keys).

    ```bash
    keystore.jks
    keystore.pkcs12
    server.crt
    server.csr
    server.key
    server.pem
    ```

4. Generate `password` to avoid plain text security password (used for key generation in `generate-keys.sh`) transfer.

    ```bash
    cd scripts/
    ./generate-password.sh used_password_when_generate_keys
    cd ..
    ```
    This scrips will generate 2 files in `./ppml/scripts/password` dir.

    ```bash
    key.txt
    output.bin
    ```
### 2.2 Trusted Big Data Analytics and ML on JVM

#### 2.2.1 Prepare Docker Image

Pull docker image from Dockerhub
```bash
docker pull intelanalytics/bigdl-ppml-trusted-big-data-ml-scala-graphene:0.14.0-SNAPSHOT
```

Alternatively, you can build docker image from Dockerfile (this will take some time):

```bash
cd trusted-big-data-ml/scala/docker-graphene
./build-docker-image.sh
```

#### 2.2.2 Run Trusted Big Data and ML on Single Node

##### 2.2.2.1 Start PPML Container

Enter `analytics-zoo/ppml/trusted-big-data-ml/scala/docker-graphene` dir.

1. Copy `keys` and `password`
    ```bash
    cd trusted-big-data-ml/scala/docker-graphene
    # copy keys and password into current directory
    cp -r ../.././../scripts/keys/ .
    cp -r ../.././../scripts/password/ .
    ```
2. Prepare the data
   To train a model with ppml in analytics zoo and bigdl, you need to prepare the data first. The Docker image is taking lenet and mnist as example. <br>
   You can download the MNIST Data from [here](http://yann.lecun.com/exdb/mnist/). Unzip all the files and put them in one folder(e.g. mnist). <br>
   There are four files. **train-images-idx3-ubyte** contains train images, **train-labels-idx1-ubyte** is train label file, **t10k-images-idx3-ubyte** has validation images    and **t10k-labels-idx1-ubyte** contains validation labels. For more detail, please refer to the download page. <br>
   After you decompress the gzip files, these files may be renamed by some decompress tools, e.g. **train-images-idx3-ubyte** is renamed to **train-images.idx3-ubyte**. Please change the name back before you run the example.  <br>
   
3. To start the container, first modify the paths in deploy-local-spark-sgx.sh, and then run the following commands:
    ```bash
    ./deploy-local-spark-sgx.sh
    sudo docker exec -it spark-local bash
    cd /ppml/trusted-big-data-ml
    ./init.sh
    ```
    **ENCLAVE_KEY_PATH** means the absolute path to the "enclave-key.pem", according to the above commands, the path would be like "analytics-zoo/ppml/scripts/enclave-key.pem". <br>
    **DATA_PATH** means the absolute path to the data(like mnist) that would used later in the spark program. According to the above commands, the path would be like "analytics-zoo/ppml/trusted-big-data-ml/scala/docker-graphene/mnist" <br>
    **KEYS_PATH** means the absolute path to the keys you just created and copied to. According to the above commands, the path would be like "analytics-zoo/ppml/trusted-big-data-ml/scala/docker-graphene/keys" <br>
    **LOCAL_IP** means your local IP address. <br>

##### 2.2.2.2 Run Your Spark Program with Analytics Zoo PPML on SGX

To run your pyspark program, first you need to prepare your own pyspark program and put it under the trusted directory in SGX  `/ppml/trusted-big-data-ml/work`. Then run with `ppml-spark-submit.sh` using the command:

```bash
./ppml-spark-submit.sh work/YOUR_PROMGRAM.py | tee YOUR_PROGRAM-sgx.log
```

When the program finishes, check the results with the log `YOUR_PROGRAM-sgx.log`.

##### 2.2.2.3 Run Trusted Spark Examples with Analytics Zoo PPML SGX

##### 2.2.2.3.1 Run Trusted Spark Pi

This example runs a simple Spark PI program, which is an easy way to verify if the Trusted PPML environment is ready.  

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

##### 2.2.2.3.2 Run Trusted Spark SQL

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
bash start-spark-local-tpc-h-sgx.sh [your_hdfs_tpch_data_dir] [your_hdfs_output_dir]
```

Open another terminal and check the log:

```bash
sudo docker exec -it spark-local cat /ppml/trusted-big-data-ml/spark.local.tpc.h.sgx.log | egrep "###|INFO|finished"
```

The result should look like:

>   ----------------22 finished--------------------

##### 2.2.2.3.3 Run Trusted Deep Learning

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

#### 2.2.3 Run Trusted Big Data and ML on Cluster

##### 2.2.3.1 Configure the Environment

Prerequisite: passwordless ssh login to all the nodes needs to be properly set up first.

```bash
nano environments.sh
```
##### 2.2.3.2 Start Distributed Big Data and ML Platform

First run the following command to start the service:

```bash
./deploy-distributed-standalone-spark.sh
```

Then run the following command to start the training:

```bash
./start-distributed-spark-train-sgx.sh
```
##### 2.2.3.3  Stop Distributed Big Data and ML Platform

First, stop the training:

```bash
./stop-distributed-standalone-spark.sh
```

Then stop the service:

```bash
./undeploy-distributed-standalone-spark.sh
```

### 2.3 Trusted Big Data Analytics and ML with Python

#### 2.3.1 Prepare Docker Image

Pull docker image from Dockerhub

```bash
docker pull intelanalytics/bigdl-ppml-trusted-big-data-ml-python-graphene:0.14-SNAPSHOT
```

Alternatively, you can build docker image from Dockerfile (this will take some time):

```bash
cd ppml/trusted-big-data-ml/python/docker-graphene
./build-docker-image.sh
```

#### 2.3.2 Run Trusted Big Data and ML on Single Node

##### 2.3.2.1 Start PPML Container

Enter `analytics-zoo/ppml/trusted-big-data-ml/python/docker-graphene` directory.

1. Copy `keys` and `password` to current directory

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

##### 2.3.2.2 Run Your Pyspark Program with Analytics Zoo PPML on SGX

To run your pyspark program, first you need to prepare your own pyspark program and put it under the trusted directory in SGX  `/ppml/trusted-big-data-ml/work`. Then run with `ppml-spark-submit.sh` using the command:

```bash
./ppml-spark-submit.sh work/YOUR_PROMGRAM.py | tee YOUR_PROGRAM-sgx.log
```

When the program finishes, check the results with the log `YOUR_PROGRAM-sgx.log`.

##### 2.3.2.3 Run Python and Pyspark Examples with Analytics Zoo PPML on SGX

##### 2.3.2.3.1 Run Trusted Python Helloworld

This example runs a simple native python program, which is an easy way to verify if the Trusted PPML environment is correctly set up.

Run the script to run trusted Python Helloworld:

```bash
bash work/start-scripts/start-python-helloworld-sgx.sh
```

Open another terminal and check the log:

```bash
sudo docker exec -it spark-local cat /ppml/trusted-big-data-ml/test-helloworld-sgx.log | egrep "Hello World"
```

The result should look something like:

> Hello World

##### 2.3.2.3.2 Run Trusted Python Numpy

This example shows how to run trusted native python numpy.

Run the script to run trusted Python Numpy:

```bash
bash work/start-scripts/start-python-numpy-sgx.sh
```

Open another terminal and check the log:

```bash
sudo docker exec -it spark-local cat /ppml/trusted-big-data-ml/test-numpy-sgx.log | egrep "numpy.dot"
```

The result should look something like:

>  numpy.dot: 0.034211914986371994 sec

##### 2.3.2.3.3 Run Trusted Spark Pi

This example runs a simple Spark PI program.

Run the script to run trusted Spark Pi:

```bash
bash work/start-scripts/start-spark-local-pi-sgx.sh
```

Open another terminal and check the log:

```bash
sudo docker exec -it spark-local cat /ppml/trusted-big-data-ml/test-pi-sgx.log | egrep "roughly"
```

The result should look something like:

> Pi is roughly 3.146760

##### 2.3.2.3.4 Run Trusted Spark Wordcount

This example runs a simple Spark Wordcount program.

Run the script to run trusted Spark Wordcount:

```bash
bash work/start-scripts/start-spark-local-wordcount-sgx.sh
```

Open another terminal and check the log:

```bash
sudo docker exec -it spark-local cat /ppml/trusted-big-data-ml/test-wordcount-sgx.log | egrep "print"
```

The result should look something like:

> print("Hello: 1
>
> print(sys.path);: 1

##### 2.3.2.3.5 Run Trusted Spark SQL

This example shows how to run trusted Spark SQL.

First, make sure that the paths of resource in `/ppml/trusted-big-data-ml/work/spark-2.4.6/examples/src/main/python/sql/basic.py` are the same as the paths of `people.json`  and `people.txt`.

Run the script to run trusted Spark SQL:

```bash
bash work/start-scripts/start-spark-local-sql-sgx.sh
```

Open another terminal and check the log:

```bash
sudo docker exec -it spark-local cat /ppml/trusted-big-data-ml/test-sql-basic-sgx.log | egrep "Justin"
```

The result should look something like:

>| 19| Justin|
>
>| Justin|
>
>| Justin| 20|
>
>| 19| Justin|
>
>| 19| Justin|
>
>| 19| Justin|
>
>Name: Justin
>
>| Justin|

##### 2.3.2.3.6 Run Trusted Spark BigDL

This example shows how to run trusted Spark BigDL.

Run the script to run trusted Spark BigDL and it would take some time to show the final results:

```bash
bash work/start-scripts/start-spark-local-bigdl-sgx.sh
```

Open another terminal and check the log:

```bash
sudo docker exec -it spark-local cat /ppml/trusted-big-data-ml/test-bigdl-lenet-sgx.log | egrep "Accuracy"
```

The result should look something like:

> creating: createTop1Accuracy
>
> 2021-06-18 01:39:45 INFO DistriOptimizer$:180 - [Epoch 1 60032/60000][Iteration 469][Wall Clock 457.926565s] Top1Accuracy is Accuracy(correct: 9488, count: 10000, accuracy: 0.9488)
>
> 2021-06-18 01:46:20 INFO DistriOptimizer$:180 - [Epoch 2 60032/60000][Iteration 938][Wall Clock 845.747782s] Top1Accuracy is Accuracy(correct: 9696, count: 10000, accuracy: 0.9696)

##### 2.3.2.3.7 Run Trusted Spark XGBoost Regressor

This example shows how to run trusted Spark XGBoost Regressor.

First, make sure that `Boston_Housing.csv` is under `work/data` directory or the same path in the `start-spark-local-xgboost-regressor-sgx.sh`. Replace the value of `RABIT_TRACKER_IP` with your own IP address in the script.

Run the script to run trusted Spark XGBoost Regressor and it would take some time to show the final results:

```bash
bash work/start-scripts/start-spark-local-xgboost-regressor-sgx.sh
```

Open another terminal and check the log:

```bash
sudo docker exec -it spark-local cat /ppml/trusted-big-data-ml/test-zoo-xgboost-regressor-sgx.log | egrep "prediction" -A19
```

The result should look something like:

> | features|label| prediction|
>
> +--------------------+-----+------------------+
>
> |[41.5292,0.0,18.1...| 8.5| 8.51994514465332|
>
> |[67.9208,0.0,18.1...| 5.0| 5.720333099365234|
>
> |[20.7162,0.0,18.1...| 11.9|10.601168632507324|
>
> |[11.9511,0.0,18.1...| 27.9| 26.19390106201172|
>
> |[7.40389,0.0,18.1...| 17.2|16.112293243408203|
>
> |[14.4383,0.0,18.1...| 27.5|25.952226638793945|
>
> |[51.1358,0.0,18.1...| 15.0| 14.67484188079834|
>
> |[14.0507,0.0,18.1...| 17.2|16.112293243408203|
>
> |[18.811,0.0,18.1,...| 17.9| 17.42863655090332|
>
> |[28.6558,0.0,18.1...| 16.3| 16.0191593170166|
>
> |[45.7461,0.0,18.1...| 7.0| 5.300708770751953|
>
> |[18.0846,0.0,18.1...| 7.2| 6.346951007843018|
>
> |[10.8342,0.0,18.1...| 7.5| 6.571983814239502|
>
> |[25.9406,0.0,18.1...| 10.4|10.235769271850586|
>
> |[73.5341,0.0,18.1...| 8.8| 8.460335731506348|
>
> |[11.8123,0.0,18.1...| 8.4| 9.193297386169434|
>
> |[11.0874,0.0,18.1...| 16.7|16.174896240234375|
>
> |[7.02259,0.0,18.1...| 14.2| 13.38729190826416|

##### 2.3.2.3.8 Run Trusted Spark XGBoost Classifier

This example shows how to run trusted Spark XGBoost Classifier.

Before running the example, download the sample dataset from [pima-indians-diabetes](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv) dataset. After downloading the dataset, make sure that `pima-indians-diabetes.data.csv` is under `work/data` directory or the same path in the `start-spark-local-xgboost-classifier-sgx.sh`. Replace `path_of_pima_indians_diabetes_csv` with your path of `pima-indians-diabetes.data.csv`  and the value of `RABIT_TRACKER_IP` with your own IP address in the script.

Run the script to run trusted Spark XGBoost Classifier and it would take some time to show the final results:

```bash
bash start-spark-local-xgboost-classifier-sgx.sh
```

Open another terminal and check the log:

```bash
sudo docker exec -it spark-local cat /ppml/trusted-big-data-ml/test-xgboost-classifier-sgx.log | egrep "prediction" -A7
```

The result should look something like:

> | f1|  f2| f3| f4|  f5| f6|  f7| f8|label|    rawPrediction|     probability|prediction|
>
> +----+-----+----+----+-----+----+-----+----+-----+--------------------+--------------------+----------+
>
> |11.0|138.0|74.0|26.0|144.0|36.1|0.557|50.0| 1.0|[-0.8209581375122...|[0.17904186248779...|    1.0|
>
> | 3.0|106.0|72.0| 0.0| 0.0|25.8|0.207|27.0| 0.0|[-0.0427864193916...|[0.95721358060836...|    0.0|
>
> | 6.0|117.0|96.0| 0.0| 0.0|28.7|0.157|30.0| 0.0|[-0.2336160838603...|[0.76638391613960...|    0.0|
>
> | 2.0| 68.0|62.0|13.0| 15.0|20.1|0.257|23.0| 0.0|[-0.0315906107425...|[0.96840938925743...|    0.0|
>
> | 9.0|112.0|82.0|24.0| 0.0|28.2|1.282|50.0| 1.0|[-0.7087597250938...|[0.29124027490615...|    1.0|
>
> | 0.0|119.0| 0.0| 0.0| 0.0|32.4|0.141|24.0| 1.0|[-0.4473398327827...|[0.55266016721725...|    0.0|

##### 2.3.2.3.9 Run Trusted Spark Orca Data

This example shows how to run trusted Spark Orca Data.

Before running the example, download the [NYC Taxi](https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv) dataset in Numenta Anoomaly Benchmark for demo. After downloading the dataset, make sure that `nyc_taxi.csv` is under `work/data` directory or the same path in the `start-spark-local-orca-data-sgx.sh`. Replace  `path_of_nyc_taxi_csv` with your path of `nyc_taxi.csv` in the script.

Run the script to run trusted Spark Orca Data and it would take some time to show the final results:

```bash
bash start-spark-local-orca-data-sgx.sh
```

Open another terminal and check the log:

```bash
sudo docker exec -it spark-local cat /ppml/trusted-big-data-ml/test-orca-data-sgx.log | egrep -a "INFO data|Stopping" -A10
```

The result should contain the content look like:

>INFO data collected: [        timestamp value
>
>0   2014-07-01 00:00:00 10844
>
>1   2014-07-01 00:30:00  8127
>
>2   2014-07-01 01:00:00  6210
>
>3   2014-07-01 01:30:00  4656
>
>4   2014-07-01 02:00:00  3820
>
>...          ...  ...
>
>10315 2015-01-31 21:30:00 24670
>
>10316 2015-01-31 22:00:00 25721
>
>10317 2015-01-31 22:30:00 27309
>
>10318 2015-01-31 23:00:00 26591
>
>\--
>
>INFO data2 collected: [        timestamp value      datetime hours awake
>
>0  2014-07-01 00:00:00 10844 2014-07-01 00:00:00   0   1
>
>1  2014-07-01 00:30:00  8127 2014-07-01 00:30:00   0   1
>
>2  2014-07-01 03:00:00  2369 2014-07-01 03:00:00   3   0
>
>3  2014-07-01 04:30:00  2158 2014-07-01 04:30:00   4   0
>
>4  2014-07-01 05:00:00  2515 2014-07-01 05:00:00   5   0
>
>...         ...  ...         ...  ...  ...
>
>5215 2015-01-31 17:30:00 23595 2015-01-31 17:30:00   17   1
>
>5216 2015-01-31 18:30:00 27286 2015-01-31 18:30:00   18   1
>
>5217 2015-01-31 19:00:00 28804 2015-01-31 19:00:00   19   1
>
>5218 2015-01-31 19:30:00 27773 2015-01-31 19:30:00   19   1
>
>\--
>
>Stopping orca context

##### 2.3.2.3.10 Run Trusted Spark Orca Learn Tensorflow Basic Text Classification

This example shows how to run trusted Spark Orca learn Tensorflow basic text classification.

Run the script to run trusted Spark Orca learn Tensorflow basic text classification and it would take some time to show the final results. To run this example in standalone mode, replace `-e SGX_MEM_SIZE=32G \` with `-e SGX_MEM_SIZE=64G \` in `start-distributed-spark-driver.sh`

```bash
bash start-spark-local-orca-tf-text.sh
```

Open another terminal and check the log:

```bash
sudo docker exec -it spark-local cat test-orca-tf-text.log | egrep "results"
```

The result should be similar to:

>INFO results: {'loss': 0.6932533979415894, 'acc Top1Accuracy': 0.7544000148773193}

#### 2.3.3 Run Trusted Big Data and ML on Cluster

##### 2.3.3.1 Configure the Environment

Prerequisite: passwordless ssh login to all the nodes needs to be properly set up first.

```bash
nano environments.sh
```

##### 2.3.3.2 Start Distributed Big Data and ML Platform

First run the following command to start the service:

```bash
./deploy-distributed-standalone-spark.sh
```

Then start the service:

```bash
./start-distributed-spark-driver.sh
```

After that, you can run previous examples on cluster by replacing `--master 'local[4]'` in the start scripts with

```bash
--master 'spark://your_master_url' \
--conf spark.authenticate=true \
--conf spark.authenticate.secret=your_secret_key \
```

##### 2.3.3.3 Stop Distributed Big Data and ML Platform

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

### 3.2 Prepare Docker Image

Pull docker image from Dockerhub

```bash
# For Graphene
docker pull intelanalytics/bigdl-ppml-trusted-realtime-ml-scala-graphene:0.14.0-SNAPSHOT
```

```bash
# For Occlum
docker pull intelanalytics/bigdl-ppml-trusted-realtime-ml-scala-occlum:0.14.0-SNAPSHOT
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
input_api.enqueue('my-image1', user_define_key={"path": 'path/to/image1'})
```

Cluster Serving service is a long running service in container, you can stop it as follows:

```bash
docker stop trusted-cluster-serving-local
```
