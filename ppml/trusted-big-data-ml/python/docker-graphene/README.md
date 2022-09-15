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

##### Prepare the Data

  To train a model with ppml in BigDL, you need to prepare the data first. The Docker image is taking lenet and mnist as example.You can download the MNIST Data from [here](http://yann.lecun.com/exdb/mnist/). Unzip all the files and put them in one folder(e.g. mnist).

  There are four files. **train-images-idx3-ubyte** contains train images, **train-labels-idx1-ubyte** is train label file, **t10k-images-idx3-ubyte** has validation images and **t10k-labels-idx1-ubyte** contains validation labels. For more detail, please refer to the download page.

  After you decompress the gzip files, these files may be renamed by some decompress tools, e.g. **train-images-idx3-ubyte** is renamed to **train-images.idx3-ubyte**. Please change the name back before you run the example.

##### Prepare the Key

  The ppml in bigdl needs secured keys to enable spark security such as Authentication, RPC Encryption, Local Storage Encryption and TLS, you need to prepare the secure keys and keystores. In this tutorial, you can generate keys and keystores with root permission (test only, need input security password for keys).

  ```bash
  sudo bash ../../../scripts/generate-keys.sh
  ```

  You also need to generate your enclave key using the command below, and keep it safely for future remote attestations and to start SGX enclaves more securely.

  It will generate a file `enclave-key.pem` in your present working directory, which will be your enclave key. To store the key elsewhere, modify the outputted file path.

  ```bash
  openssl genrsa -3 -out enclave-key.pem 3072
  ```

##### Prepare the Password

  Next, you need to store the password you used for key generation, i.e., `generate-keys.sh`, in a secured file.

  ```bash
  sudo bash ../../../scripts/generate-password.sh used_password_when_generate_keys
  ```

## Run Your PySpark Program

#### 1. Start the container to run native python examples

Before you run the following commands to start the container, you need to modify the paths in `deploy-local-spark-sgx.sh` and then run the following commands.

```bash
./deploy-local-spark-sgx.sh
sudo docker exec -it spark-local bash
cd /ppml/trusted-big-data-ml
```

#### 2. Run your pyspark program

To run your pyspark program, first you need to prepare your own pyspark program and put it under the trusted directory in SGX  `/ppml/trusted-big-data-ml/work`. Then run with `bigdl-ppml-submit.sh` using the command:

```bash
./bigdl-ppml-submit.sh work/YOUR_PROMGRAM.py | tee YOUR_PROGRAM-sgx.log
```

When the program finishes, check the results with the log `YOUR_PROGRAM-sgx.log`.

## Run Native Python Examples

#### 1. Start the container to run native python examples

Before you run the following commands to start the container, you need to modify the paths in `deploy-local-spark-sgx.sh` and then run the following commands.

```bash
./deploy-local-spark-sgx.sh
sudo docker exec -it spark-local bash
cd /ppml/trusted-big-data-ml
```

 #### 2. Run native python examples

##### Example 1: `helloworld.py`

Run the example with SGX with the following command in the terminal.

```bash
/graphene/Tools/argv_serializer bash -c "python ./work/examples/helloworld.py" > /ppml/trusted-big-data-ml/secured-argvs
./init.sh
SGX=1 ./pal_loader bash | tee test-helloworld-sgx.log
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
/graphene/Tools/argv_serializer bash -c "python ./work/examples/test-numpy.py | tee test-numpy-sgx.log" > /ppml/trusted-big-data-ml/secured-argvs
./init.sh
SGX=1 ./pal_loader bash | tee test-numpy-sgx.log
```

Then check the output with the following command.

```bash
cat test-numpy-sgx.log | egrep "numpy.dot"
```

The result should be similar to

> numpy.dot: 0.034211914986371994 sec

## Run Python code with dependencies

It usually happens that your code needs to depend on some other packages. In this section, we will show how to install python dependencies within the ppml running environment without changing the Dockerfile by using **conda** or python **eggs**.


### Install python dependencies using Conda

The code used in this example can be found [here](https://spark.apache.org/docs/latest/api/python/user_guide/python_packaging.html).

#### 1. Prepare conda environment and pacakge

##### Prepare conda environment
Run the following command in the terminal with **conda installed**.

```bash
conda create -y -n pyspark_conda_env -c conda-forge pyarrow pandas conda-pack
```

At this step, please add any dependencies you need when creating the conda environment. We use pyarrow and pandas here because our example code only needs these two packages.

##### Pack the conda environment
Run the following command in the terminal to pack the environment

```bash
conda activate pyspark_conda_env # activate the conda environment
conda pack -f -o pyspark_conda_env.tar.gz
```

#### 2. Start the client container
The detailed instructions on how to build/start the client container can be found [here](https://github.com/intel-analytics/BigDL/blob/main/ppml/docs/prepare_environment.md#start-bigdl-ppml-client-container).

After you have booted up the client container, you can load your app.py and pyspark_conda_env.tar.gz into the container by putting them into the **$DATA_PATH**.

Please be noted that we have mapped **$DATA_PATH** to path **/ppml/trusted-big-data-ml/work/data** within the container.

Alternatively, you can use the **docker cp** command if you only want to try this feature.


#### 3. Submit the task and watch the result
##### Get into the client container
You can get into your container and acquire a terminal by the following command:
```bash
docker exec -it bigdl-ppml-client-k8s /bin/bash
```


##### Submit your task
The following bash command shows an example on how to submit task in cluster mode with sgx enabled. Plsase be noted that this example is only for test purpose, and you should configure the parameters accordingly when using it in your environment.

For detailed explaination of the following command, please check [here](https://github.com/intel-analytics/BigDL/blob/main/ppml/docs/submit_job.md#ppml-cli).

```bash
#!/bin/bash
export secure_password=`openssl rsautl -inkey /ppml/trusted-big-data-ml/work/password/key.txt -decrypt </ppml/trusted-big-data-ml/work/password/output.bin`
bash bigdl-ppml-submit.sh \
        --master $RUNTIME_SPARK_MASTER \
        --deploy-mode cluster \
        --sgx-enabled true \
        --sgx-log-level error \
        --sgx-driver-memory 64g \
        --sgx-driver-jvm-memory 12g \
        --sgx-executor-memory 64g \
        --sgx-executor-jvm-memory 12g \
        --driver-memory 32g \
        --driver-cores 8 \
        --executor-memory 32g \
        --executor-cores 8 \
        --conf spark.kubernetes.container.image=$RUNTIME_K8S_SPARK_IMAGE \
        --num-executors 2 \
        --name spark-test \
        --verbose \
        --archives local:///ppml/trusted-big-data-ml/work/pyspark_conda_env.tar.gz#pyspark_conda_env \
        local:///ppml/trusted-big-data-ml/work/data/app.py
```

We can get the result of the command by either checking the driver pod's log or using the **tee** command to store the output into a file.


```bash
kubectl logs $(kubectl get pods | grep spark-test | grep driver | awk '{print $1}') | egrep "^\[Row"
```

The result should be

>[Row(id=1, mean_udf(v)=1.5), Row(id=2, mean_udf(v)=6.0)]

which, indicates that the dependency **pyarrow** and **pandas** has been successfully loaded into the running environment.


### Install python dependencies using egg

Although the use of eggs is deprecated, the **spark-submit** command allow users to pass dependencies to the executors through python eggs and the `--py-files` option.

In this section, we will show how to use python eggs to install a `demo` package into spark executors.

#### Prepare your Python eggs
You can prepare the eggs you need by either searching [PyPI](https://pypi.org/) or packaging the source code with the [Setuptools](https://setuptools.pypa.io/en/latest/setuptools.html#develop-deploy-the-project-source-in-development-mode).

Here, we will use a very simple egg file for demonstration purpose. The egg named `demo` only contains a function `test()` whose function is to print "Hello World" to the terminal.


#### Submit the task and watch result
Please refer to the above subsections for starting the container and transferring the egg file into the container.

The following bash command shows an example on how to install dependencies within the spark executors running in kubernetes clusters.

```bash
export secure_password=`openssl rsautl -inkey /ppml/trusted-big-data-ml/work/password/key.txt -decrypt </ppml/trusted-big-data-ml/work/password/output.bin`
bash bigdl-ppml-submit.sh \
        --master $RUNTIME_SPARK_MASTER \
        --deploy-mode cluster \
        --sgx-enabled true \
        --sgx-log-level error \
        --sgx-driver-memory 64g \
        --sgx-driver-jvm-memory 12g \
        --sgx-executor-memory 64g \
        --sgx-executor-jvm-memory 12g \
        --driver-memory 32g \
        --driver-cores 8 \
        --executor-memory 32g \
        --executor-cores 8 \
        --conf spark.kubernetes.container.image=$RUNTIME_K8S_SPARK_IMAGE \
        --num-executors 2 \
        --name spark-test \
        --verbose \
        --py-files local:///ppml/trusted-big-data-ml/work/demo-0.1.0-py3.7.egg \
        local:///ppml/trusted-big-data-ml/work/data/app.py
```

The result can be obtained by running the following bash command:

```bash
kubectl logs $(kubectl get pods | grep spark-test | grep driver | awk '{print $1}') | egrep "^Hello"
```

The result should be
>Hello World

which, indicates that the dependency `demo` package has been successfully loaded into the running environment.
## Run as Spark Local Mode

#### 1. Start the container to run spark applications in spark local mode

Before you run the following commands to start the container, you need to modify the paths in `deploy-local-spark-sgx.sh` and then run the following commands.

```bash
./deploy-local-spark-sgx.sh
sudo docker exec -it spark-local bash
cd /ppml/trusted-big-data-ml
```

#### 2. Run PySpark examples

##### Example 1: `pi.py`

Run the example with SGX spark local mode with the following command in the terminal. 

```bash
/graphene/Tools/argv_serializer bash -c "/opt/jdk8/bin/java \
        -cp '/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*' \
        -Xmx1g org.apache.spark.deploy.SparkSubmit \
        --master 'local[4]' \
        --conf spark.network.timeout=10000000 \
        --conf spark.executor.heartbeatInterval=10000000 \
        --conf spark.python.use.daemon=false \
        --conf spark.python.worker.reuse=false \
        /ppml/trusted-big-data-ml/work/spark-3.1.2/examples/src/main/python/pi.py" > /ppml/trusted-big-data-ml/secured-argvs
./init.sh
SGX=1 ./pal_loader bash 2>&1 | tee test-pi-sgx.log
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
/graphene/Tools/argv_serializer bash -c "export PYSPARK_PYTHON=/usr/bin/python && /opt/jdk8/bin/java \
        -cp '/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*' \
        -Xmx1g org.apache.spark.deploy.SparkSubmit \
        --master 'local[4]' \
        --conf spark.network.timeout=10000000 \
        --conf spark.executor.heartbeatInterval=10000000 \
        --conf spark.python.use.daemon=false \
        --conf spark.python.worker.reuse=false \
        /ppml/trusted-big-data-ml/work/spark-3.1.2/examples/src/main/python/wordcount.py ./work/examples/helloworld.py" > /ppml/trusted-big-data-ml/secured-argvs
./init.sh
SGX=1 ./pal_loader bash 2>&1 | tee test-wordcount-sgx.log
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
/graphene/Tools/argv_serializer bash -c "export PYSPARK_PYTHON=/usr/bin/python && \
        /opt/jdk8/bin/java \
        -cp '/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*' \
        -Xmx1g org.apache.spark.deploy.SparkSubmit \
        --master 'local[4]' \
        --conf spark.python.use.daemon=false \
        --conf spark.python.worker.reuse=false \
        /ppml/trusted-big-data-ml/work/spark-3.1.2/examples/src/main/python/sql/basic.py" > /ppml/trusted-big-data-ml/secured-argvs

./init.sh
SGX=1 ./pal_loader bash 2>&1 | tee test-sql-basic-sgx.log
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
/graphene/Tools/argv_serializer bash -c "/opt/jdk8/bin/java -cp \
  '/ppml/trusted-big-data-ml/work/bigdl-2.1.0-SNAPSHOT/jars/*:/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*' \
  -Xmx2g \
  org.apache.spark.deploy.SparkSubmit \
  --master 'local[4]' \
  --conf spark.python.use.daemon=false \
  --conf spark.python.worker.reuse=false \
  --conf spark.driver.memory=8g \
  --conf spark.rpc.message.maxSize=190 \
  --conf spark.network.timeout=10000000 \
  --conf spark.executor.heartbeatInterval=10000000 \
  --properties-file /ppml/trusted-big-data-ml/work/bigdl-2.1.0-SNAPSHOT/conf/spark-bigdl.conf \
  --py-files /ppml/trusted-big-data-ml/work/bigdl-2.1.0-SNAPSHOT/python/bigdl-orca-spark_3.1.2-2.1.0-SNAPSHOT-python-api.zip,/ppml/trusted-big-data-ml/work/bigdl-2.1.0-SNAPSHOT/python/bigdl-dllib-spark_3.1.2-2.1.0-SNAPSHOT-python-api.zip,/ppml/trusted-big-data-ml/work/bigdl-2.1.0-SNAPSHOT/examples/dllib/lenet/lenet.py \
  --driver-cores 2 \
  --total-executor-cores 2 \
  --executor-cores 2 \
  --executor-memory 8g \
  /ppml/trusted-big-data-ml/work/bigdl-2.1.0-SNAPSHOT/examples/dllib/lenet/lenet.py \
  --dataPath /ppml/trusted-big-data-ml/work/data/mnist \
  --maxEpoch 2" > /ppml/trusted-big-data-ml/secured-argvs
  
./init.sh
SGX=1 ./pal_loader bash 2>&1 | tee test-bigdl-lenet-sgx.log
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

##### (Deprecated) Example 5: XGBoost Regressor

Please be noted that the xgboost example listed here is **deprecated** due to the fact that Rabit's network (contains gradient, split and env) is not protected.

The data source `Boston_Housing.csv` can be found at [here](https://github.com/selva86/datasets/blob/master/BostonHousing.csv).

Before running the example, make sure that `Boston_Housing.csv` is under `work/data` directory or the same path in the command. Run the example with SGX spark local mode with the following command in the terminal. Replace `path_of_boston_housing_csv` with your path of `Boston_Housing.csv`.


Note that data in `Boston_Housing.csv` needs to be pre-processed, before training with `xgboost_example.py`.

The data for column "chas" is in type "string" and we need to delete all the **quotation marks(")** so that the `xgboost_example.py` can successfully load the data.

Before changing:
> 0.00632,18,2.31,**"0"**,0.538,6.575,65.2,4.09,1,296,15.3,396.9,4.98,24

After changing:
> 0.00632,18,2.31,**0**,0.538,6.575,65.2,4.09,1,296,15.3,396.9,4.98,24

```bash
/graphene/Tools/argv_serializer bash -c "/opt/jdk8/bin/java -cp \
    '/ppml/trusted-big-data-ml/work/bigdl-2.1.0-SNAPSHOT/jars/*:/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*' \
  -Xmx2g \
  org.apache.spark.deploy.SparkSubmit \
  --master 'local[4]' \
  --conf spark.driver.memory=2g \
  --conf spark.executor.extraClassPath=/ppml/trusted-big-data-ml/work/bigdl-2.1.0-SNAPSHOT/jars/* \
  --conf spark.driver.extraClassPath=/ppml/trusted-big-data-ml/work/bigdl-2.1.0-SNAPSHOT/jars/* \
  --properties-file /ppml/trusted-big-data-ml/work/bigdl-2.1.0-SNAPSHOT/conf/spark-bigdl.conf \
  --py-files /ppml/trusted-big-data-ml/work/bigdl-2.1.0-SNAPSHOT/python/bigdl-orca-spark_3.1.2-2.1.0-SNAPSHOT-python-api.zip \
  --executor-memory 2g \
  /ppml/trusted-big-data-ml/work/bigdl-2.1.0-SNAPSHOT/examples/dllib/nnframes/xgboost/xgboost_example.py \
  --file-path path_of_boston_housing_csv" > /ppml/trusted-big-data-ml/secured-argvs
./init.sh
SGX=1 ./pal_loader bash 2>&1 | tee test-zoo-xgboost-regressor-sgx.log
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

##### (Deprecated) Example 6: XGBoost Classifier

Please be noted that the xgboost example listed here is **deprecated** due to the fact that Rabit's network (contains gradient, split and env) is not protected.

Before running the example, download the sample dataset from [pima-indians-diabetes](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv) dataset manually or with following command.

```bash
wget https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
```

After downloading the dataset, make sure that `pima-indians-diabetes.data.csv` is under `work/data` directory or the same path in the command. Run the example with SGX spark local mode with the following command in the terminal. Replace `path_of_pima_indians_diabetes_csv` with your path of `pima-indians-diabetes.data.csv`.

```bash
/graphene/Tools/argv_serializer bash -c "/opt/jdk8/bin/java -cp \
  '/ppml/trusted-big-data-ml/work/bigdl-2.1.0-SNAPSHOT/jars/*:/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*' \
  -Xmx2g \
  org.apache.spark.deploy.SparkSubmit \
  --master 'local[4]' \
  --conf spark.driver.memory=2g \
  --conf spark.executor.extraClassPath=/ppml/trusted-big-data-ml/work/bigdl-2.1.0-SNAPSHOT/jars/* \
  --conf spark.driver.extraClassPath=/ppml/trusted-big-data-ml/work/bigdl-2.1.0-SNAPSHOT/jars/* \
  --properties-file /ppml/trusted-big-data-ml/work/bigdl-2.1.0-SNAPSHOT/conf/spark-bigdl.conf \
  --py-files /ppml/trusted-big-data-ml/work/bigdl-2.1.0-SNAPSHOT/python/bigdl-orca-spark_3.1.2-2.1.0-SNAPSHOT-python-api.zip \
  --executor-memory 2g \
  /ppml/trusted-big-data-ml/work/bigdl-2.1.0-SNAPSHOT/examples/dllib/nnframes/xgboost/xgboost_classifier.py \
  -f path_of_pima_indians_diabetes_csv" > /ppml/trusted-big-data-ml/secured-argvs
./init.sh
SGX=1 ./pal_loader bash 2>&1 | tee test-xgboost-classifier-sgx.log
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

## Run as Spark on Kubernetes Mode

WARNING: If you want spark standalone mode, please refer to [standalone/README.md][standalone]. But it is not recommended.

Follow the guide below to run Spark on Kubernetes manually. Alternatively, you can also use Helm to set everything up automatically. See [kubernetes/README.md][helmGuide].

### 1. Start the spark client as Docker container
### 1.1 Prepare the keys/password/data/enclave-key.pem
Please refer to the previous section about [preparing data, key and password](#prepare-data).

``` bash
bash ../../../scripts/generate-keys.sh
bash ../../../scripts/generate-password.sh YOUR_PASSWORD
kubectl apply -f keys/keys.yaml
kubectl apply -f password/password.yaml
```
Run `cd kubernetes && bash enclave-key-to-secret.sh` to generate your enclave key and add it to your Kubernetes cluster as a secret.
### 1.2 Prepare the k8s configurations
#### 1.2.1 Create the RBAC
```bash
kubectl create serviceaccount spark
kubectl create clusterrolebinding spark-role --clusterrole=edit --serviceaccount=default:spark --namespace=default
```
#### 1.2.2 Generate k8s config file
```bash
kubectl config view --flatten --minify > /YOUR_DIR/kubeconfig
```
#### 1.2.3 Create k8s secret
```bash
kubectl create secret generic spark-secret --from-literal secret=YOUR_SECRET
```
**The secret created (`YOUR_SECRET`) should be the same as the password you specified in section 1.1**

### 1.3 Start the client container
Configure the environment variables in the following script before running it. Check [Bigdl ppml SGX related configurations](#1-bigdl-ppml-sgx-related-configurations) for detailed memory configurations.
```bash
export K8S_MASTER=k8s://$(sudo kubectl cluster-info | grep 'https.*6443' -o -m 1)
echo The k8s master is $K8S_MASTER .
export ENCLAVE_KEY=/YOUR_DIR/enclave-key.pem
export DATA_PATH=/YOUR_DIR/data
export KEYS_PATH=/YOUR_DIR/keys
export SECURE_PASSWORD_PATH=/YOUR_DIR/password
export KUBECONFIG_PATH=/YOUR_DIR/kubeconfig
export LOCAL_IP=$LOCAL_IP
export DOCKER_IMAGE=intelanalytics/bigdl-ppml-trusted-big-data-ml-python-graphene:2.1.0-SNAPSHOT
sudo docker run -itd \
    --privileged \
    --net=host \
    --name=spark-local-k8s-client \
    --cpuset-cpus="0-4" \
    --oom-kill-disable \
    --device=/dev/sgx/enclave \
    --device=/dev/sgx/provision \
    -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
    -v $ENCLAVE_KEY:/graphene/Pal/src/host/Linux-SGX/signer/enclave-key.pem \
    -v $DATA_PATH:/ppml/trusted-big-data-ml/work/data \
    -v $KEYS_PATH:/ppml/trusted-big-data-ml/work/keys \
    -v $SECURE_PASSWORD_PATH:/ppml/trusted-big-data-ml/work/password \
    -v $KUBECONFIG_PATH:/root/.kube/config \
    -e RUNTIME_SPARK_MASTER=$K8S_MASTER \
    -e RUNTIME_K8S_SERVICE_ACCOUNT=spark \
    -e RUNTIME_K8S_SPARK_IMAGE=$DOCKER_IMAGE \
    -e RUNTIME_DRIVER_HOST=$LOCAL_IP \
    -e RUNTIME_DRIVER_PORT=54321 \
    -e RUNTIME_DRIVER_CORES=1 \
    -e RUNTIME_EXECUTOR_INSTANCES=1 \
    -e RUNTIME_EXECUTOR_CORES=8 \
    -e RUNTIME_EXECUTOR_MEMORY=1g \
    -e RUNTIME_TOTAL_EXECUTOR_CORES=4 \
    -e RUNTIME_DRIVER_CORES=4 \
    -e RUNTIME_DRIVER_MEMORY=1g \
    -e SGX_DRIVER_MEM=32g \
    -e SGX_DRIVER_JVM_MEM=8g \
    -e SGX_EXECUTOR_MEM=32g \
    -e SGX_EXECUTOR_JVM_MEM=12g \
    -e SGX_ENABLED=true \
    -e SGX_LOG_LEVEL=error \
    -e SPARK_MODE=client \
    -e LOCAL_IP=$LOCAL_IP \
    $DOCKER_IMAGE bash
```
run `docker exec -it spark-local-k8s-client bash` to entry the container.

### 1.4 Init the client and run Spark applications on k8s (1.4 can be skipped if you are using 1.5 to submit jobs)

#### 1.4.1 Configure `spark-executor-template.yaml` in the container

We assume you have a working Network File System (NFS) configured for your Kubernetes cluster. Configure the `nfsvolumeclaim` on the last line to the name of the Persistent Volume Claim (PVC) of your NFS.

Please prepare the following and put them in your NFS directory:

- The data (in a directory called `data`),
- The kubeconfig file.

#### 1.4.2 Prepare secured-argvs for client

Note: If you are running this client in trusted env, please skip this step. Then, directly run this command without `/graphene/Tools/argv_serializer bash -c`.

```bash
/graphene/Tools/argv_serializer bash -c "secure_password=`openssl rsautl -inkey /ppml/trusted-big-data-ml/work/password/key.txt -decrypt </ppml/trusted-big-data-ml/work/password/output.bin` && TF_MKL_ALLOC_MAX_BYTES=10737418240 && \
    SPARK_LOCAL_IP=$LOCAL_IP && \
    /opt/jdk8/bin/java \
        -cp '/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*' \
        -Xmx8g \
        org.apache.spark.deploy.SparkSubmit \
        --master $RUNTIME_SPARK_MASTER \
        --deploy-mode $SPARK_MODE \
        --name spark-pi-sgx \
        --conf spark.driver.host=$SPARK_LOCAL_IP \
        --conf spark.driver.port=$RUNTIME_DRIVER_PORT \
        --conf spark.driver.memory=$RUNTIME_DRIVER_MEMORY \
        --conf spark.driver.cores=$RUNTIME_DRIVER_CORES \
        --conf spark.executor.cores=$RUNTIME_EXECUTOR_CORES \
        --conf spark.executor.memory=$RUNTIME_EXECUTOR_MEMORY \
        --conf spark.executor.instances=$RUNTIME_EXECUTOR_INSTANCES \
        --conf spark.kubernetes.authenticate.driver.serviceAccountName=spark \
        --conf spark.kubernetes.container.image=$RUNTIME_K8S_SPARK_IMAGE \
        --conf spark.kubernetes.driver.podTemplateFile=/ppml/trusted-big-data-ml/spark-driver-template.yaml \
        --conf spark.kubernetes.executor.podTemplateFile=/ppml/trusted-big-data-ml/spark-executor-template.yaml \
        --conf spark.kubernetes.executor.deleteOnTermination=false \
        --conf spark.network.timeout=10000000 \
        --conf spark.executor.heartbeatInterval=10000000 \
        --conf spark.python.use.daemon=false \
        --conf spark.python.worker.reuse=false \
        --conf spark.kubernetes.sgx.enabled=$SGX_ENABLED \
        --conf spark.kubernetes.sgx.driver.mem=$SGX_DRIVER_MEM \
        --conf spark.kubernetes.sgx.driver.jvm.mem=$SGX_DRIVER_JVM_MEM \
        --conf spark.kubernetes.sgx.executor.mem=$SGX_EXECUTOR_MEM \
        --conf spark.kubernetes.sgx.executor.jvm.mem=$SGX_EXECUTOR_JVM_MEM \
        --conf spark.kubernetes.sgx.log.level=$SGX_LOG_LEVEL \
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
        local:///ppml/trusted-big-data-ml/work/spark-3.1.2/examples/jars/spark-examples_2.12-3.1.2.jar" > /ppml/trusted-big-data-ml/secured-argvs
```

Init Graphene command.

```bash
./init.sh
```

Note that: you can run your own Spark Appliction after changing `--class` and jar path.

1. `local:///ppml/trusted-big-data-ml/work/spark-3.1.2/examples/jars/spark-examples_2.12-3.1.2.jar` => `your_jar_path`
2. `--class org.apache.spark.examples.SparkPi` => `--class your_class_path`

#### 1.4.3 Spark-Pi example
```bash
SGX=1 ./pal_loader bash 2>&1 | tee spark-pi-sgx-$SPARK_MODE.log
```
### 1.5 Use bigdl-ppml-submit.sh to submit ppml jobs
#### 1.5.1 Spark-Pi on local mode
![image2022-6-6_16-18-10](https://user-images.githubusercontent.com/61072813/174703141-63209559-05e1-4c4d-b096-6b862a9bed8a.png)
```
#!/bin/bash
bash bigdl-ppml-submit.sh \
        --sgx-enabled false \
        --master local[2] \
        --driver-memory 32g \
        --driver-cores 8 \
        --executor-memory 32g \
        --executor-cores 8 \
        --num-executors 2 \
        --class org.apache.spark.examples.SparkPi \
        --name spark-pi \
        --verbose \
        local:///ppml/trusted-big-data-ml/work/spark-3.1.2/examples/jars/spark-examples_2.12-3.1.2.jar 3000
```
#### 1.5.2 Spark-Pi on local sgx mode
![image2022-6-6_16-18-57](https://user-images.githubusercontent.com/61072813/174703165-2afc280d-6a3d-431d-9856-dd5b3659214a.png)
```
#!/bin/bash
bash bigdl-ppml-submit.sh \
        --master local[2] \
        --sgx-enabled true \
        --sgx-log-level error \
        --sgx-driver-memory 64g \
        --sgx-driver-jvm-memory 12g \
        --sgx-executor-memory 64g \
        --sgx-executor-jvm-memory 12g \
        --driver-memory 32g \
        --driver-cores 8 \
        --executor-memory 32g \
        --executor-cores 8 \
        --num-executors 2 \
        --class org.apache.spark.examples.SparkPi \
        --name spark-pi \
        --verbose \
        local:///ppml/trusted-big-data-ml/work/spark-3.1.2/examples/jars/spark-examples_2.12-3.1.2.jar 3000
```
#### 1.5.3 Spark-Pi on client mode
![image2022-6-6_16-19-43](https://user-images.githubusercontent.com/61072813/174703216-70588315-7479-4b6c-9133-095104efc07d.png)
```
#!/bin/bash
 
export secure_password=`openssl rsautl -inkey /ppml/trusted-big-data-ml/work/password/key.txt -decrypt </ppml/trusted-big-data-ml/work/password/output.bin`
bash bigdl-ppml-submit.sh \
        --master $RUNTIME_SPARK_MASTER \
        --deploy-mode client \
        --sgx-enabled true \
        --sgx-log-level error \
        --sgx-driver-memory 64g \
        --sgx-driver-jvm-memory 12g \
        --sgx-executor-memory 64g \
        --sgx-executor-jvm-memory 12g \
        --driver-memory 32g \
        --driver-cores 8 \
        --executor-memory 32g \
        --executor-cores 8 \
        --num-executors 2 \
        --conf spark.kubernetes.container.image=$RUNTIME_K8S_SPARK_IMAGE \
        --class org.apache.spark.examples.SparkPi \
        --name spark-pi \
        --verbose \
        local:///ppml/trusted-big-data-ml/work/spark-3.1.2/examples/jars/spark-examples_2.12-3.1.2.jar 3000
```
#### 1.5.4 Spark-Pi on cluster mode
![image2022-6-6_16-20-0](https://user-images.githubusercontent.com/61072813/174703234-e45b8fe5-9c61-4d17-93ef-6b0c961a2f95.png)
```
#!/bin/bash

export secure_password=`openssl rsautl -inkey /ppml/trusted-big-data-ml/work/password/key.txt -decrypt </ppml/trusted-big-data-ml/work/password/output.bin`
bash bigdl-ppml-submit.sh \
        --master $RUNTIME_SPARK_MASTER \
        --deploy-mode cluster \
        --sgx-enabled true \
        --sgx-log-level error \
        --sgx-driver-memory 64g \
        --sgx-driver-jvm-memory 12g \
        --sgx-executor-memory 64g \
        --sgx-executor-jvm-memory 12g \
        --driver-memory 32g \
        --driver-cores 8 \
        --executor-memory 32g \
        --executor-cores 8 \
        --conf spark.kubernetes.container.image=$RUNTIME_K8S_SPARK_IMAGE \
        --num-executors 2 \
        --class org.apache.spark.examples.SparkPi \
        --name spark-pi \
        --verbose \
        local:///ppml/trusted-big-data-ml/work/spark-3.1.2/examples/jars/spark-examples_2.12-3.1.2.jar 3000
```

#### 1.5.5 bigdl-ppml-submit.sh explanations

bigdl-ppml-submit.sh is used to simplify the steps in 1.4

1. To use bigdl-ppml-submit.sh, first set the following required arguments: 
```
--master $RUNTIME_SPARK_MASTER \
--deploy-mode cluster \
--driver-memory 32g \
--driver-cores 8 \
--executor-memory 32g \
--executor-cores 8 \
--sgx-enabled true \
--sgx-log-level error \
--sgx-driver-memory 64g \
--sgx-driver-jvm-memory 12g \
--sgx-executor-memory 64g \
--sgx-executor-jvm-memory 12g \
--conf spark.kubernetes.container.image=$RUNTIME_K8S_SPARK_IMAGE \
--num-executors 2 \
--name spark-pi \
--verbose \
--class org.apache.spark.examples.SparkPi \
local:///ppml/trusted-big-data-ml/work/spark-3.1.2/examples/jars/spark-examples_2.12-3.1.2.jar 3000
```
if you are want to enable sgx, don't forget to set the sgx-related arguments
```
--sgx-enabled true \
--sgx-log-level error \
--sgx-driver-memory 64g \
--sgx-driver-jvm-memory 12g \
--sgx-executor-memory 64g \
--sgx-executor-jvm-memory 12g \
```
you can update the application arguments to anything you want to run
```
--class org.apache.spark.examples.SparkPi \
local:///ppml/trusted-big-data-ml/work/spark-3.1.2/examples/jars/spark-examples_2.12-3.1.2.jar 3000
```

2. If you want to enable the spark security configurations as in 2.Spark security configurations, export secure_password to enable it.
```
export secure_password=`openssl rsautl -inkey /ppml/trusted-big-data-ml/work/password/key.txt -decrypt </ppml/trusted-big-data-ml/work/password/output.bin`
```

3. The following spark properties are set by default in bigdl-ppml-submit.sh. If you want to overwrite them or add new spark properties, just append the spark properties to bigdl-ppml-submit.sh as arguments.
```
--conf spark.driver.host=$LOCAL_IP \
--conf spark.driver.port=$RUNTIME_DRIVER_PORT \
--conf spark.network.timeout=10000000 \
--conf spark.executor.heartbeatInterval=10000000 \
--conf spark.python.use.daemon=false \
--conf spark.python.worker.reuse=false \
--conf spark.kubernetes.authenticate.driver.serviceAccountName=spark \
--conf spark.kubernetes.driver.podTemplateFile=/ppml/trusted-big-data-ml/spark-driver-template.yaml \
--conf spark.kubernetes.executor.podTemplateFile=/ppml/trusted-big-data-ml/spark-executor-template.yaml \
--conf spark.kubernetes.executor.deleteOnTermination=false \
```


### Configuration Explainations

#### 1. Bigdl ppml SGX related configurations

<img title="" src="../../../../docs/readthedocs/image/ppml_memory_config.png" alt="ppml_memory_config.png" data-align="center">

The following parameters enable spark executor running on SGX.  
`spark.kubernetes.sgx.enabled`: true -> enable spark executor running on sgx, false -> native on k8s without SGX.  
`spark.kubernetes.sgx.driver.mem`: Spark driver SGX epc memeory.  
`spark.kubernetes.sgx.driver.jvm.mem`: Spark driver JVM memory, Recommended setting is less than half of epc memory.  
`spark.kubernetes.sgx.executor.mem`: Spark executor SGX epc memeory.  
`spark.kubernetes.sgx.executor.jvm.mem`: Spark executor JVM memory, Recommended setting is less than half of epc memory.  
`spark.kubernetes.sgx.log.level`: Spark executor on SGX log level, Supported values are error,all and debug.  
The following is a recommended configuration in client mode.
```bash
    --conf spark.kubernetes.sgx.enabled=true
    --conf spark.kubernetes.sgx.driver.mem=32g
    --conf spark.kubernetes.sgx.driver.jvm.mem=10g
    --conf spark.kubernetes.sgx.executor.mem=32g
    --conf spark.kubernetes.sgx.executor.jvm.mem=12g
    --conf spark.kubernetes.sgx.log.level=error
    --conf spark.driver.memory=10g
    --conf spark.executor.memory=1g
```
The following is a recommended configuration in cluster mode.
```bash
    --conf spark.kubernetes.sgx.enabled=true
    --conf spark.kubernetes.sgx.driver.mem=32g
    --conf spark.kubernetes.sgx.driver.jvm.mem=10g
    --conf spark.kubernetes.sgx.executor.mem=32g
    --conf spark.kubernetes.sgx.executor.jvm.mem=12g
    --conf spark.kubernetes.sgx.log.level=error
    --conf spark.driver.memory=1g
    --conf spark.executor.memory=1g
```
When SGX is not used, the configuration is the same as spark native.
```bash
    --conf spark.driver.memory=10g
    --conf spark.executor.memory=12g
```
#### 2. Spark security configurations
Below is an explanation of these security configurations, Please refer to [Spark Security](https://spark.apache.org/docs/3.1.2/security.html) for detail.  
##### 2.1 Spark RPC
###### 2.1.1 Authentication
`spark.authenticate`: true -> Spark authenticates its internal connections, default is false.  
`spark.authenticate.secret`: The secret key used authentication.  
`spark.kubernetes.executor.secretKeyRef.SPARK_AUTHENTICATE_SECRET` and `spark.kubernetes.driver.secretKeyRef.SPARK_AUTHENTICATE_SECRET`: mount `SPARK_AUTHENTICATE_SECRET` environment variable from a secret for both the Driver and Executors.  
`spark.authenticate.enableSaslEncryption`: true -> enable SASL-based encrypted communication, default is false.  
```bash
    --conf spark.authenticate=true
    --conf spark.authenticate.secret=$secure_password
    --conf spark.kubernetes.executor.secretKeyRef.SPARK_AUTHENTICATE_SECRET="spark-secret:secret" 
    --conf spark.kubernetes.driver.secretKeyRef.SPARK_AUTHENTICATE_SECRET="spark-secret:secret" 
    --conf spark.authenticate.enableSaslEncryption=true
```

###### 2.1.2 Encryption
`spark.network.crypto.enabled`: true -> enable AES-based RPC encryption, default is false.  
`spark.network.crypto.keyLength`: The length in bits of the encryption key to generate.  
`spark.network.crypto.keyFactoryAlgorithm`: The key factory algorithm to use when generating encryption keys.  
```bash
    --conf spark.network.crypto.enabled=true 
    --conf spark.network.crypto.keyLength=128 
    --conf spark.network.crypto.keyFactoryAlgorithm=PBKDF2WithHmacSHA1
```
###### 2.1.3. Local Storage Encryption
`spark.io.encryption.enabled`: true -> enable local disk I/O encryption, default is false.  
`spark.io.encryption.keySizeBits`: IO encryption key size in bits.  
`spark.io.encryption.keygen.algorithm`: The algorithm to use when generating the IO encryption key.  
```bash
    --conf spark.io.encryption.enabled=true
    --conf spark.io.encryption.keySizeBits=128
    --conf spark.io.encryption.keygen.algorithm=HmacSHA1
```
###### 2.1.4 SSL Configuration
`spark.ssl.enabled`: true -> enable SSL.  
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
[helmGuide]: https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/python/docker-graphene/kubernetes/README.md
[standalone]: https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/python/docker-graphene/standalone/README.md

