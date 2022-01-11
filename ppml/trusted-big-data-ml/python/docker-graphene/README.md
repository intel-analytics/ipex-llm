# Trusted Big Data ML with Python

SGX-based Trusted Big Data ML allows the user to run end-to-end big data analytics application and Intel BigDL model training with spark local and distributed cluster on Graphene-SGX.

*Please mind the IP and file path settings. They should be changed to the IP/path of your own sgx server on which you are running the programs.*

## Before Running the Code

#### 1. Build docker image

Before running the following command, please modify the paths in `build-docker-image.sh`. Then build the docker image with the following command.

```bash
./build-docker-image.sh
```

#### 2. Prepare data, key and password

*  ##### Prepare the Data

  To train a model with ppml in bigdl, you need to prepare the data first. The Docker image is taking lenet and mnist as example.You can download the MNIST Data from [here](http://yann.lecun.com/exdb/mnist/). Unzip all the files and put them in one folder(e.g. mnist). 

  There are four files. **train-images-idx3-ubyte** contains train images, **train-labels-idx1-ubyte** is train label file, **t10k-images-idx3-ubyte** has validation images and **t10k-labels-idx1-ubyte** contains validation labels. For more detail, please refer to the download page. 

  After you decompress the gzip files, these files may be renamed by some decompress tools, e.g. **train-images-idx3-ubyte** is renamed to **train-images.idx3-ubyte**. Please change the name back before you run the example.

* ##### Prepare the Key

  The ppml in bigdl needs secured keys to enable spark security such as Authentication, RPC Encryption, Local Storage Encryption and TLS, you need to prepare the secure keys and keystores. In this tutorial, you can generate keys and keystores with root permission (test only, need input security password for keys).

  ```bash
  sudo ../../../scripts/generate-keys.sh
  ```

  You also need to generate your enclave key using the command below, and keep it safely for future remote attestations and to start SGX enclaves more securely.

  It will generate a file `enclave-key.pem` in your present working directory, which will be your enclave key. To store the key elsewhere, modify the outputted file path.

  ```bash
  openssl genrsa -3 -out enclave-key.pem 3072
  ```

* ##### Prepare the Password

  Next, you need to store the password you used for key generation, i.e., `generate-keys.sh`, in a secured file.

  ```bash
  sudo bash ../../../scripts/generate-password.sh used_password_when_generate_keys
  ```

## Run Your Pyspark Program

#### 1. Start the container to run native python examples

Before you run the following commands to start the container, you need to modify the paths in `deploy-local-spark-sgx.sh` and then run the following commands.

```bash
./deploy-local-spark-sgx.sh
sudo docker exec -it spark-local bash
cd /ppml/trusted-big-data-ml
./init.sh
```

 #### 2. Run your pyspark program

To run your pyspark program, first you need to prepare your own pyspark program and put it under the trusted directory in SGX  `/ppml/trusted-big-data-ml/work`. Then run with `ppml-spark-submit.sh` using the command:

```bash
./ppml-spark-submit.sh work/YOUR_PROMGRAM.py | tee YOUR_PROGRAM-sgx.log
```

When the program finishes, check the results with the log `YOUR_PROGRAM-sgx.log`.

## Run Native Python Examples

#### 1. Start the container to run native python examples

Before you run the following commands to start the container, you need to modify the paths in `deploy-local-spark-sgx.sh` and then run the following commands.

```bash
./deploy-local-spark-sgx.sh
sudo docker exec -it spark-local bash
cd /ppml/trusted-big-data-ml
./init.sh
```

 #### 2. Run native python examples

##### Example 1: `helloworld.py`

Run the example with SGX with the following command in the terminal.

```bash
SGX=1 ./pal_loader bash -c "python ./work/examples/helloworld.py" | tee test-helloworld-sgx.log
```
Then check the output with the following command.

```bash
cat test-helloworld-sgx.log | egrep "Hello World"
```

The result should be 

> Hello World

##### Example 2: `test-numpy.py`

Run the example with SGX with the following command in the terminal.

```bash
SGX=1 ./pal_loader bash -c "python ./work/examples/test-numpy.py" | tee test-numpy-sgx.log
```

Then check the output with the following command.

```bash
cat test-numpy-sgx.log | egrep "numpy.dot"
```

The result should be similar to

> numpy.dot: 0.034211914986371994 sec

## Run as Spark Local Mode

#### 1. Start the container to run spark applications in spark local mode

Before you run the following commands to start the container, you need to modify the paths in `deploy-local-spark-sgx.sh` and then run the following commands.

```bash
./deploy-local-spark-sgx.sh
sudo docker exec -it spark-local bash
cd /ppml/trusted-big-data-ml
./init.sh
```

 #### 2. Run pyspark examples

##### Example 1: `pi.py`

Run the example with SGX spark local mode with the following command in the terminal. 

```bash
SGX=1 ./pal_loader bash -c "/opt/jdk8/bin/java \
        -cp '/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*' \
        -Xmx1g org.apache.spark.deploy.SparkSubmit \
        --master 'local[4]' \
        --conf spark.python.use.daemon=false \
        /ppml/trusted-big-data-ml/work/spark-3.1.2/examples/src/main/python/pi.py" 2>&1 | tee test-pi-sgx.log
```

Then check the output with the following command.

```bash
cat test-pi-sgx.log | egrep "roughly"
```

The result should be similar to

>Pi is roughly 3.146760

##### Example 2: `test-wordcount.py`

Run the example with SGX spark local mode with the following command in the terminal. 

```bash
SGX=1 ./pal_loader bash -c "export PYSPARK_PYTHON=/usr/bin/python && /opt/jdk8/bin/java \
        -cp '/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*' \
        -Xmx1g org.apache.spark.deploy.SparkSubmit \
        --master 'local[4]' \
        --conf spark.python.use.daemon=false \
        /ppml/trusted-big-data-ml/work/spark-3.1.2/examples/src/main/python/wordcount.py ./work/examples/helloworld.py" 2>&1 | tee test-wordcount-sgx.log
```

Then check the output with the following command.

```bash
cat test-wordcount-sgx.log | egrep -a "import.*: [0-9]*$"
```

The result should be similar to

> import: 1

##### Example 3: Basic SQL

Before running the example, make sure that the paths of resource in `/ppml/trusted-big-data-ml/work/spark-3.1.2/examples/src/main/python/sql/basic.py` are the same as the paths of `people.json`  and `people.txt`.

Run the example with SGX spark local mode with the following command in the terminal. 

```bash
SGX=1 ./pal_loader bash -c "export PYSPARK_PYTHON=/usr/bin/python && \
        /opt/jdk8/bin/java \
        -cp '/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*' \
        -Xmx1g org.apache.spark.deploy.SparkSubmit \
        --master 'local[4]' \
        --conf spark.python.use.daemon=false \
        --conf spark.python.worker.reuse=false \
        /ppml/trusted-big-data-ml/work/spark-3.1.2/examples/src/main/python/sql/basic.py" 2>&1 | tee test-sql-basic-sgx.log
```

Then check the output with the following command.

```bash
cat test-sql-basic-sgx.log | egrep "Justin"
```

The result should be similar to

> | 19|  Justin|
>
> |  Justin| 
>
> |  Justin|       20|
>
> | 19|  Justin|
>
> | 19|  Justin|
>
> | 19|  Justin|
>
> Name: Justin
>
> |  Justin|

##### Example 4: Bigdl lenet

Run the example with SGX spark local mode with the following command in the terminal. 

```bash
SGX=1 ./pal_loader bash -c "/opt/jdk8/bin/java -cp \
  '/ppml/trusted-big-data-ml/work/bigdl-0.14.0-SNAPSHOT/jars/*:/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*' \
  -Xmx2g \
  org.apache.spark.deploy.SparkSubmit \
  --master 'local[4]' \
  --conf spark.python.use.daemon=false \
  --conf spark.python.worker.reuse=false \
  --conf spark.driver.memory=8g \
  --conf spark.rpc.message.maxSize=190 \
  --conf spark.network.timeout=10000000 \
  --conf spark.executor.heartbeatInterval=10000000 \
  --properties-file /ppml/trusted-big-data-ml/work/bigdl-0.14.0-SNAPSHOT/conf/spark-bigdl.conf \
  --py-files /ppml/trusted-big-data-ml/work/bigdl-0.14.0-SNAPSHOT/python/bigdl-orca-spark_3.1.2-0.14.0-SNAPSHOT-python-api.zip,/ppml/trusted-big-data-ml/work/bigdl-0.14.0-SNAPSHOT/python/bigdl-dllib-spark_3.1.2-0.14.0-SNAPSHOT-python-api.zip,/ppml/trusted-big-data-ml/work/bigdl-0.14.0-SNAPSHOT/examples/dllib/lenet/lenet.py \
  --driver-cores 2 \
  --total-executor-cores 2 \
  --executor-cores 2 \
  --executor-memory 8g \
  /ppml/trusted-big-data-ml/work/bigdl-0.14.0-SNAPSHOT/examples/dllib/lenet/lenet.py \
  --dataPath /ppml/trusted-big-data-ml/work/data/mnist \
  --maxEpoch 2" 2>&1 | tee test-bigdl-lenet-sgx.log
```

Then check the output with the following command.

```bash
cat test-bigdl-lenet-sgx.log | egrep "Accuracy"
```

The result should be similar to

>creating: createTop1Accuracy
>
>2021-06-18 01:39:45 INFO DistriOptimizer$:180 - [Epoch 1 60032/60000][Iteration 469][Wall Clock 457.926565s] Top1Accuracy is Accuracy(correct: 9488, count: 10000, accuracy: 0.9488)
>
>2021-06-18 01:46:20 INFO DistriOptimizer$:180 - [Epoch 2 60032/60000][Iteration 938][Wall Clock 845.747782s] Top1Accuracy is Accuracy(correct: 9696, count: 10000, accuracy: 0.9696)

##### Example 5: XGBoost Regressor

Before running the example, make sure that `Boston_Housing.csv` is under `work/data` directory or the same path in the command. Run the example with SGX spark local mode with the following command in the terminal. Replace `your_IP_address` with your IP address and `path_of_boston_housing_csv` with your path of `Boston_Housing.csv`.

```bash
SGX=1 ./pal_loader bash -c "export RABIT_TRACKER_IP=your_IP_address && /opt/jdk8/bin/java -cp \
    '/ppml/trusted-big-data-ml/work/analytics-zoo-0.12.0-SNAPSHOT/lib/analytics-zoo-bigdl_0.13.0-spark_2.4.6-0.12.0-SNAPSHOT-jar-with-dependencies.jar:/ppml/trusted-big-data-ml/work/spark-2.4.6/conf/:/ppml/trusted-big-data-ml/work/spark-2.4.6/jars/*' \
  -Xmx2g \
  org.apache.spark.deploy.SparkSubmit \
  --master 'local[4]' \
  --conf spark.driver.memory=2g \
  --conf spark.executor.extraClassPath=/ppml/trusted-big-data-ml/work/analytics-zoo-0.12.0-SNAPSHOT/lib/analytics-zoo-bigdl_0.13.0-spark_2.4.6-0.12.0-SNAPSHOT-jar-with-dependencies.jar \
  --conf spark.driver.extraClassPath=/ppml/trusted-big-data-ml/work/analytics-zoo-0.12.0-SNAPSHOT/lib/analytics-zoo-bigdl_0.13.0-spark_2.4.6-0.12.0-SNAPSHOT-jar-with-dependencies.jar \
  --properties-file /ppml/trusted-big-data-ml/work/analytics-zoo-0.12.0-SNAPSHOT/conf/spark-analytics-zoo.conf \
  --jars /ppml/trusted-big-data-ml/work/analytics-zoo-0.12.0-SNAPSHOT/lib/analytics-zoo-bigdl_0.13.0-spark_2.4.6-0.12.0-SNAPSHOT-jar-with-dependencies.jar \
  --py-files /ppml/trusted-big-data-ml/work/analytics-zoo-0.12.0-SNAPSHOT/lib/analytics-zoo-bigdl_0.13.0-spark_2.4.6-0.12.0-SNAPSHOT-python-api.zip \
  --executor-memory 2g \
  /ppml/trusted-big-data-ml/work/examples/pyzoo/xgboost/xgboost_example.py \
  --file-path path_of_boston_housing_csv" | tee test-zoo-xgboost-regressor-sgx.log
```

Then check the output with the following command.

```bash
cat test-zoo-xgboost-regressor-sgx.log | egrep "prediction" -A19
```

The result should be similar to

>|      features|label|    prediction|
>
>+--------------------+-----+------------------+
>
>|[41.5292,0.0,18.1...| 8.5| 8.51994514465332|
>
>|[67.9208,0.0,18.1...| 5.0| 5.720333099365234|
>
>|[20.7162,0.0,18.1...| 11.9|10.601168632507324|
>
>|[11.9511,0.0,18.1...| 27.9| 26.19390106201172|
>
>|[7.40389,0.0,18.1...| 17.2|16.112293243408203|
>
>|[14.4383,0.0,18.1...| 27.5|25.952226638793945|
>
>|[51.1358,0.0,18.1...| 15.0| 14.67484188079834|
>
>|[14.0507,0.0,18.1...| 17.2|16.112293243408203|
>
>|[18.811,0.0,18.1,...| 17.9| 17.42863655090332|
>
>|[28.6558,0.0,18.1...| 16.3| 16.0191593170166|
>
>|[45.7461,0.0,18.1...| 7.0| 5.300708770751953|
>
>|[18.0846,0.0,18.1...| 7.2| 6.346951007843018|
>
>|[10.8342,0.0,18.1...| 7.5| 6.571983814239502|
>
>|[25.9406,0.0,18.1...| 10.4|10.235769271850586|
>
>|[73.5341,0.0,18.1...| 8.8| 8.460335731506348|
>
>|[11.8123,0.0,18.1...| 8.4| 9.193297386169434|
>
>|[11.0874,0.0,18.1...| 16.7|16.174896240234375|
>
>|[7.02259,0.0,18.1...| 14.2| 13.38729190826416|

##### Example 6: XGBoost Classifier

Before running the example, download the sample dataset from [pima-indians-diabetes](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv) dataset manually or with following command. 

```bash
wget https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
```

After downloading the dataset, make sure that `pima-indians-diabetes.data.csv` is under `work/data` directory or the same path in the command. Run the example with SGX spark local mode with the following command in the terminal. Replace `your_IP_address` with your IP address and `path_of_pima_indians_diabetes_csv` with your path of `pima-indians-diabetes.data.csv`.

```bash
SGX=1 ./pal_loader bash -c "export RABIT_TRACKER_IP=your_IP_address && /opt/jdk8/bin/java -cp \
  '/ppml/trusted-big-data-ml/work/analytics-zoo-0.12.0-SNAPSHOT/lib/analytics-zoo-bigdl_0.13.0-spark_2.4.6-0.12.0-SNAPSHOT-jar-with-dependencies.jar:/ppml/trusted-big-data-ml/work/spark-2.4.6/conf/:/ppml/trusted-big-data-ml/work/spark-2.4.6/jars/*' \
  -Xmx2g \
  org.apache.spark.deploy.SparkSubmit \
  --master 'local[4]' \
  --conf spark.driver.memory=2g \
  --conf spark.executor.extraClassPath=/ppml/trusted-big-data-ml/work/analytics-zoo-0.12.0-SNAPSHOT/lib/analytics-zoo-bigdl_0.13.0-spark_2.4.6-0.12.0-SNAPSHOT-jar-with-dependencies.jar \
  --conf spark.driver.extraClassPath=/ppml/trusted-big-data-ml/work/analytics-zoo-0.12.0-SNAPSHOT/lib/analytics-zoo-bigdl_0.13.0-spark_2.4.6-0.12.0-SNAPSHOT-jar-with-dependencies.jar \
  --properties-file /ppml/trusted-big-data-ml/work/analytics-zoo-0.12.0-SNAPSHOT/conf/spark-analytics-zoo.conf \
  --jars /ppml/trusted-big-data-ml/work/analytics-zoo-0.12.0-SNAPSHOT/lib/analytics-zoo-bigdl_0.13.0-spark_2.4.6-0.12.0-SNAPSHOT-jar-with-dependencies.jar \
  --py-files /ppml/trusted-big-data-ml/work/analytics-zoo-0.12.0-SNAPSHOT/lib/analytics-zoo-bigdl_0.13.0-spark_2.4.6-0.12.0-SNAPSHOT-python-api.zip \
  --executor-memory 2g \
  /ppml/trusted-big-data-ml/work/examples/pyzoo/xgboost/xgboost_classifier.py \
  -f path_of_pima_indians_diabetes_csv" | tee test-xgboost-classifier-sgx.log
```

Then check the output with the following command.

```bash
cat test-xgboost-classifier-sgx.log | egrep "prediction" -A7
```

The result should be similar to

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

##### Example 7: Orca data

Before running the example, download the [NYC Taxi](https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv) dataset in Numenta Anoomaly Benchmark for demo manually or with following command. 

```bash
wget https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv
```

After downloading the dataset, make sure that `nyc_taxi.csv` is under `work/data` directory or the same path in the command. Run the example with SGX spark local mode with the following command in the terminal. Replace `path_of_nyc_taxi_csv` with your path of `nyc_taxi.csv`.

```bash
SGX=1 ./pal_loader bash -c "/opt/jdk8/bin/java -cp \
  '/ppml/trusted-big-data-ml/work/analytics-zoo-0.12.0-SNAPSHOT/lib/analytics-zoo-bigdl_0.13.0-spark_2.4.6-0.12.0-SNAPSHOT-jar-with-dependencies.jar:/ppml/trusted-big-data-ml/work/spark-2.4.6/conf/:/ppml/trusted-big-data-ml/work/spark-2.4.6/jars/*' \
  -Xmx2g \
  org.apache.spark.deploy.SparkSubmit \
  --master 'local[4]' \
  --conf spark.driver.memory=2g \
  --conf spark.executor.extraClassPath=/ppml/trusted-big-data-ml/work/analytics-zoo-0.12.0-SNAPSHOT/lib/analytics-zoo-bigdl_0.13.0-spark_2.4.6-0.12.0-SNAPSHOT-jar-with-dependencies.jar \
  --conf spark.driver.extraClassPath=/ppml/trusted-big-data-ml/work/analytics-zoo-0.12.0-SNAPSHOT/lib/analytics-zoo-bigdl_0.13.0-spark_2.4.6-0.12.0-SNAPSHOT-jar-with-dependencies.jar \
  --properties-file /ppml/trusted-big-data-ml/work/analytics-zoo-0.12.0-SNAPSHOT/conf/spark-analytics-zoo.conf \
  --jars /ppml/trusted-big-data-ml/work/analytics-zoo-0.12.0-SNAPSHOT/lib/analytics-zoo-bigdl_0.13.0-spark_2.4.6-0.12.0-SNAPSHOT-jar-with-dependencies.jar \
  --py-files /ppml/trusted-big-data-ml/work/analytics-zoo-0.12.0-SNAPSHOT/lib/analytics-zoo-bigdl_0.13.0-spark_2.4.6-0.12.0-SNAPSHOT-python-api.zip \
  --executor-memory 2g \
  /ppml/trusted-big-data-ml/work/examples/pyzoo/orca/data/spark_pandas.py \
  -f path_of_nyc_taxi_csv" | tee test-orca-data-sgx.log
```

Then check the output with the following command.

```bash
cat test-orca-data-sgx.log | egrep -a "INFO data|Stopping" -A10
```

Then the result should contain the similar content as 

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
> 
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

##### Example 8: Orca learn Tensorflow basic text classification

Run the example with SGX spark local mode with the following command in the terminal. To run the example in SGX standalone mode, replace `-e SGX_MEM_SIZE=32G \` with `-e SGX_MEM_SIZE=64G \` in `start-distributed-spark-driver.sh`

```bash
SGX=1 ./pal_loader bash -c "/opt/jdk8/bin/java -cp \
  '/ppml/trusted-big-data-ml/work/analytics-zoo-0.12.0-SNAPSHOT/lib/analytics-zoo-bigdl_0.13.0-spark_2.4.6-0.12.0-SNAPSHOT-jar-with-dependencies.jar:/ppml/trusted-big-data-ml/work/spark-2.4.6/conf/:/ppml/trusted-big-data-ml/work/spark-2.4.6/jars/*' \
  -Xmx3g \
  org.apache.spark.deploy.SparkSubmit \
  --master 'local[4]' \
  --conf spark.driver.memory=3g \
  --conf spark.executor.extraClassPath=/ppml/trusted-big-data-ml/work/analytics-zoo-0.12.0-SNAPSHOT/lib/analytics-zoo-bigdl_0.13.0-spark_2.4.6-0.12.0-SNAPSHOT-jar-with-dependencies.jar \
  --conf spark.driver.extraClassPath=/ppml/trusted-big-data-ml/work/analytics-zoo-0.12.0-SNAPSHOT/lib/analytics-zoo-bigdl_0.13.0-spark_2.4.6-0.12.0-SNAPSHOT-jar-with-dependencies.jar \
  --properties-file /ppml/trusted-big-data-ml/work/analytics-zoo-0.12.0-SNAPSHOT/conf/spark-analytics-zoo.conf \
  --jars /ppml/trusted-big-data-ml/work/analytics-zoo-0.12.0-SNAPSHOT/lib/analytics-zoo-bigdl_0.13.0-spark_2.4.6-0.12.0-SNAPSHOT-jar-with-dependencies.jar \
  --py-files /ppml/trusted-big-data-ml/work/analytics-zoo-0.12.0-SNAPSHOT/lib/analytics-zoo-bigdl_0.13.0-spark_2.4.6-0.12.0-SNAPSHOT-python-api.zip \
  --executor-memory 3g \
  --executor-cores 2 \
  --driver-cores 2 \
  /ppml/trusted-big-data-ml/work/examples/pyzoo/orca/learn/tf/basic_text_classification/basic_text_classification.py \
  --cluster_mode local" | tee test-orca-tf-text-sgx.log
```

Then check the output with the following command.

```bash
cat test-orca-tf-text.log | egrep "results"
```

Then the result should be similar to

> INFO results: {'loss': 0.6932533979415894, 'acc Top1Accuracy': 0.7544000148773193}

## Run as Spark Standalone Mode

#### 1. Start the container to run spark applications in spark standalone mode

Before you run the following commands to start the container, you need to modify the paths in `environment.sh` and then run the following commands.

```bash
./deploy-distributed-standalone-spark.sh
./start-distributed-spark-driver.sh
```

Then use `distributed-check-status.sh` to check master's and worker's status and make sure that both of them are running.

Use the following commands to enter the docker of spark driver.

```bash
sudo docker exec -it spark-driver bash
cd /ppml/trusted-big-data-ml
./init.sh
./start-spark-standalone-driver-sgx.sh
```

#### 2. Run pyspark examples

To run the pyspark examples in spark standalone mode, you only need to replace the following command in spark local mode command:

```bash
--master 'local[4]' \
```

with

```bash
--master 'spark://your_master_url' \
--conf spark.authenticate=true \
--conf spark.authenticate.secret=your_secret_key \
```

and  replace `your_master_url` with your own master url and `your_secret_key` with your own secret key.

## Safety configurations in standalone mode
### Safety Pamameters
The security configuration in the following section ensures that tasks submitted by Spark run safely in standalone mode. These security configurations include:

    a. Authentication for RPC channels
    b. AES-based encryption for RPC connections
    c. SASL-based encrypted communication
    d. Local Storage Encryption
    e. SSL configuration

First, we create a secret on k8s, and `YOUR_SECRET` needed user defined.  
`kubectl create secret generic spark-secret --from-literal secret=$YOUR_SECRET`

Next, we need to create a `secured_password`  
```bash
export secure_password=`openssl rsautl -inkey /ppml/trusted-big-data-ml/work/password/key.txt -decrypt < /ppml/trusted-big-data-ml/work/password/output.bin`
```
#### a. Authentication for RPC channels
For Spark on YARN, Spark will automatically handle generating and distributing the shared secret, in other mechanisms, the `spark.authenticate.secret` parameter needs to be specified.  
`spark.authenticate`: whether Spark authenticates its internal connections.  
`spark.authenticate.secret`: The secret key used authentication  
`spark.kubernetes.executor.secretKeyRef.SPARK_AUTHENTICATE_SECRET` and `spark.kubernetes.driver.secretKeyRef.SPARK_AUTHENTICATE_SECRET`: mount `SPARK_AUTHENTICATE_SECRET` environment variable from a secret for both the Driver and Executors.

```bash
    --conf spark.authenticate=true
    --conf spark.authenticate.secret=$secure_password
    --conf spark.kubernetes.executor.secretKeyRef.SPARK_AUTHENTICATE_SECRET="spark-secret:secret" 
    --conf spark.kubernetes.driver.secretKeyRef.SPARK_AUTHENTICATE_SECRET="spark-secret:secret" 
```

#### b. AES-based encryption for RPC connections
`spark.network.crypto.enabled`: enable AES-based RPC encryption.  
`spark.network.crypto.keyLength`: The length in bits of the encryption key to generate.  
`spark.network.crypto.keyFactoryAlgorithm`: The key factory algorithm to use when generating encryption keys.  
```bash
    --conf spark.network.crypto.enabled=true 
    --conf spark.network.crypto.keyLength=128 
    --conf spark.network.crypto.keyFactoryAlgorithm=PBKDF2WithHmacSHA1
```
#### c. SASL-based encrypted communication
`spark.authenticate.enableSaslEncryption`: enable SASL-based encrypted communication.  
```bash
    --conf spark.authenticate.enableSaslEncryption=true
```
#### d. Local Storage Encryption
`spark.io.encryption.enabled`: enable local disk I/O encryption. Currently supported by all modes except Mesos.  
`spark.io.encryption.keySizeBits`: IO encryption key size in bits.  
`spark.io.encryption.keygen.algorithm`: The algorithm to use when generating the IO encryption key  
```bash
    --conf spark.io.encryption.enabled=true
    --conf spark.io.encryption.keySizeBits=128
    --conf spark.io.encryption.keygen.algorithm=HmacSHA1
```
#### e. SSL Configuration
`spark.ssl.enabled`: enable SSL.  
`spark.ssl.port`: the port where the SSL service will listen on.  
`spark.ssl.keyPassword`: the password to the private key in the key store.  
`spark.ssl.keyStore`: path to the key store file.  
`spark.ssl.keyStorePassword`: password to the key store.  
`spark.ssl.keyStoreType`: the type of the key store.  
`spark.ssl.trustStore`: path to the trust store file.  
`spark.ssl.trustStorePassword`: password for the trust store.  
`spark.ssl.trustStoreType`: the type of the trust store.  
```bash
      --conf spark.ssl.enabled=true
      --conf spark.ssl.port=8043
      --conf spark.ssl.keyPassword=$secure_password
      --conf spark.ssl.keyStore=/ppml/trusted-big-data-ml/work/keys/keystore.jks 
      --conf spark.ssl.keyStorePassword=$secure_password
      --conf spark.ssl.keyStoreType=JKS
      --conf spark.ssl.trustStore=/ppml/trusted-big-data-ml/work/keys/keystore.jks
      --conf spark.ssl.trustStorePassword=$secure_password  
      --conf spark.ssl.trustStoreType=JKS 
```

### Driver on SGX
Traditionally, only the executor runs in sgx mode, the following parameters can make the driver also run in sgx mode.
`spark.kubernetes.sgx.enabled`: enable driver on SGX.  
`spark.kubernetes.sgx.mem`: set `SGX_MEM_SIZE` for driver.  
`spark.kubernetes.sgx.jvm.mem`: set jvm memory for driver.

```bash
    --conf spark.kubernetes.sgx.enabled=true
    --conf spark.kubernetes.sgx.mem=32g
    --conf spark.kubernetes.sgx.jvm.mem=16g
```
### Run Spark-Pi on k8s with Safety Parameters
The following gives an example of running spark-pi on k8s, and references all security configuration parameters.

```bash
#!/bin/bash

secure_password=`openssl rsautl -inkey /ppml/trusted-big-data-ml/work/password/key.txt -decrypt </ppml/trusted-big-data-ml/work/password/output.bin` && \
export TF_MKL_ALLOC_MAX_BYTES=10737418240 && \
  export SPARK_LOCAL_IP=$LOCAL_IP && \
  /opt/jdk8/bin/java \
    -cp '/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*' \
    -Xmx10g \
    org.apache.spark.deploy.SparkSubmit \
    --master $RUNTIME_SPARK_MASTER \
    --deploy-mode client \
    --name spark-pi-sgx \
    --conf spark.driver.host=$SPARK_LOCAL_IP \
    --conf spark.driver.port=54321 \
    --conf spark.driver.memory=10g \
    --conf spark.kubernetes.authenticate.driver.serviceAccountName=spark \
    --conf spark.kubernetes.container.image=$RUNTIME_K8S_SPARK_IMAGE \
    --conf spark.kubernetes.executor.podTemplateFile=/ppml/trusted-big-data-ml/spark-executor-template.yaml \
    --conf spark.kubernetes.executor.deleteOnTermination=false \
    --conf spark.network.timeout=10000000 \
    --conf spark.executor.heartbeatInterval=10000000 \
    --conf spark.python.use.daemon=false \
    --conf spark.python.worker.reuse=false \
    --executor-cores 8 \
    --total-executor-cores 16 \
    --executor-memory 32G \
    --jars local:///ppml/trusted-big-data-ml/work/spark-3.1.2/examples/jars/spark-examples_2.12-3.1.2.jar \
    --conf spark.kubernetes.sgx.enabled=true \
    --conf spark.kubernetes.sgx.mem=32g \
    --conf spark.kubernetes.sgx.jvm.mem=16g \
    --conf spark.kubernetes.sgx.log.level=error \
    --conf spark.authenticate=true \
    --conf spark.authenticate.secret=$secure_password \
    --conf spark.kubernetes.executor.secretKeyRef.SPARK_AUTHENTICATE_SECRET="spark-secret:secret" \
    --conf spark.kubernetes.driver.secretKeyRef.SPARK_AUTHENTICATE_SECRET="spark-secret:secret" \
    --conf spark.authenticate.enableSaslEncryption=true \
    --conf spark.network.crypto.enabled=true \
    --conf spark.network.crypto.keyLength=128 \
    --conf spark.network.crypto.keyFactoryAlgorithm=PBKDF2WithHmacSHA1 \
    --conf spark.io.encryption.enabled=true \
    --conf spark.io.encryption.keySizeBits=128 \
    --conf spark.io.encryption.keygen.algorithm=HmacSHA1 \
    --conf spark.ssl.enabled=true \
    --conf spark.ssl.port=8043 \
    --conf spark.ssl.keyPassword=$secure_password \
    --conf spark.ssl.keyStore=/ppml/trusted-big-data-ml/work/keys/keystore.jks \
    --conf spark.ssl.keyStorePassword=$secure_password \
    --conf spark.ssl.keyStoreType=JKS \
    --conf spark.ssl.trustStore=/ppml/trusted-big-data-ml/work/keys/keystore.jks \
    --conf spark.ssl.trustStorePassword=$secure_password \
    --conf spark.ssl.trustStoreType=JKS \
    --class org.apache.spark.examples.SparkPi \
    --verbose \
    local:///ppml/trusted-big-data-ml/work/spark-3.1.2/examples/jars/spark-examples_2.12-3.1.2.jar 100 2>&1 | tee spark-pi-sgx.log
```
