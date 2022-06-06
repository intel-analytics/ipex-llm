In this tutorial, we prepare a PyTorch Fashion-MNIST example which completed by BigDL and supports running on yarn cluster. In particular, we will show you:
* How to prepare for the environment before submit the program to Yarn;
* How to run the BigDL program on Yarn through multiple ways;

BigDL Orca example code can be found here https://github.com/intel-analytics/BigDL/tree/main/docs/docs/tutorials/tutorial_example/Fashion_MNIST/

# Key Concepts
* OrcaContext
A BigDL Orca program usually starts with the initialization of OrcaContext. We can specify the `runtime` (default is spark, ray is also a first-class backend) and `cluster_mode` arguments to create or get a SparkContext or RayContext with optimized configurations for BigDL performance.

* Orca Estimator
After initializing the OrcaContext, we could simply create an Estimatornow, and the Estimator will replicate the model on each node in the cluster, feed the data partition on each node to the local model replica, and synchronize the model parameters using various backend technologies.

Let's get started!
# Prepare Environment
Before submitting the BigDL program to Yarn Cluster, we need to setup the environment as the following steps:
* Setup JAVA Environment
* Setup Spark Environment
* Setup Hadoop Environment
* Install All Needed Python Libraries 
* For CDH Users

## Setup JAVA Environment
We need to download and install JDK in the environment, and properly set the environment variable JAVA_HOME, which is required by Spark. JDK8 is highly recommended.

```bash
# For Ubuntu
sudo apt-get install openjdk-8-jre
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/

# For CentOS
su -c "yum install java-1.8.0-openjdk"
export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.282.b08-1.el7_9.x86_64/jre

export PATH=$PATH:$JAVA_HOME/bin
java -version  # Verify the version of JDK.
```

## Setup Spark Environment
We need to set environment variables `${SPARK_HOME}` as follows:
```bash
export SPARK_HOME=the folder path where you extract the Spark package
```

## Setup Hadoop Environment
Check the Hadoop setup and configurations of our cluster. Make sure we correctly set the environment variable HADOOP_CONF_DIR, which is needed to initialize Spark on YARN:

```bash
export HADOOP_CONF_DIR=the directory of the hadoop and yarn configurations
```

## Install Needed Python Libraries
We need first to use conda to prepare the Python environment on the local machine where we submit our application. Create a conda environment, install BigDL and all the needed Python libraries in the created conda environment:
``` bash
conda create -n bigdl python=3.7  # "bigdl" is conda environment name, you can use any name you like.
conda activate bigdl
# Use conda or pip to install all the needed Python dependencies in the created conda environment.
pip install bigdl
pip install conda-pack # Use conda-pack to pack the conda environment to an archive
```
### Options
* When you are running a program on Ray backend, please install Ray as below:
```bash
pip install ray[default]
```
* When you are running a TensorFlow2 model, please install TensorFlow as below:
```bash
pip install tensorflow==2.6.0 keras==2.6.0 # When you are running a tensorflow model.
```

## For CDH Users
If your CDH cluster has already installed Spark, the CDH’s Spark might be conflict with the pyspark installed by pip required by BigDL. Thus before running BigDL applications, you should unset all the Spark related environment variables. You can use `env | grep SPARK` to find all the existing Spark environment variables.

Also, a CDH cluster’s `HADOOP_CONF_DIR` should be `/etc/hadoop/conf` on CDH by default.

# Quick Start
In the following part, we will show you three ways that BigDL supports to submit and run our programs on Yarn:
* Use a built-in function
* Use a BigDL provided script
* Use a spark-submit script

# Run Programs with built-in function
This is the easiest and most recommended way to run BigDL on YARN, we only need to prepare the environment on the driver machine, all dependencies would be automatically packaged and distributed to the whole Yarn cluster.

## Yarn Client
When running programs on yarn-client mode by built-in function, we need to set `--cluster_mode` argument as `yarn-client` or `yarn` to create an OrcaContext on yarn for yarn-client mode.

Let's run the program as a python script:
```bash
python train.py --cluster_mode yarn-client
```

## Yarn Cluster
When running programs on yarn cluster mode, we need,
* Set `--cluster_mode` parameter to `yarn-cluster`, which could create an OrcaContext on yarn for yarn-cluster mode.
* Set `--no-download` to bind downloading dataset when creating PyTorch DataLoader.
* Set `--remote_dir` parameter to the datasets path on remote resources. We recommand you to use datasets from remote resources like HDFS and S3 to instead of downloading these datasets, since there may get connection errors on `yarn-cluster` mode.
* Set `--data_dir` to a local file to store datasets and load by `data_creator` function. BigDL provides a function to reach datasets from remote resources, which could help executors on yarn to load data.

Now, we could refer to the following script to run the example.
```bash
python train.py --cluster_mode yarn-cluster --remote_dir hdfs://path/to/remote_data --data_dir /tmp/dataset
```

# Run Programs with BigDL Provided Scripts
For `spark-submit` users, BigDL provided a `bigdl-submit` script which could automatically detect and setup configuration and jars files from the current conda environment. Before submitting our program to Yarn with `bigdl-submit`, we need to pack the Conda environment to an archive, which captures the Conda environment for Python and stores both Python interpreter and all its relevant dependencies.
```bash
conda pack -o environment.tar.gz#environment 
```
### Note:
* If `environment.tar.gz` is not under the same directory with script.py, we should modify its path in `--archives` in the running command below;
* Please make sure the `cluster_mode` parameter in `init_orca_context( )` must be `spark-submit` (default) when using `bigdl-submit` to run programs.

## Yarn Client
When running programs on yarn-client mode with `bigdl-submit`, we need:
* Set the Python environment as the local Python path. For `yarn-client` mode, the Spark driver is running on local and it will use the Python interpreter in the current active conda environment.
```bash
export PYSPARK_DRIVER_PYTHON='which python' # Do not set in yarn cluster modes.
```
* Set the executor Python environment to the path in the Conda archive, since executors will use the Python interpreter in the conda archive.
```bash
export PYSPARK_PYTHON=environment/bin/python
```
* Set the `archives` argument to the path of the Conda archive, which will be upload to remote clusters(i.e. HDFS) and distributed between all executors.
```bash
--archives environment.tar.gz#environment
```
* Set the `py-files` argument for dependency libraries.
```
--py-files ./model.py
```

Now, let's submit and execute the BigDL program with the following script:
```bash
bigdl-submit \
    --master yarn \
    --deploy-mode client \
    --executor-memory 10g \
    --driver-memory 10g \
    --executor-cores 8 \
    --num-executors 2 \
    --py-files ./model.py \
    --archives environment.tar.gz#environment \
    train.py
```

## Yarn Cluster
When running programs on yarn-cluster mode with `bigdl-submit`, we need:
* Set the `archives` argument to the path of the Conda archive, which will be upload to remote resources(i.e. HDFS) and distributed between executors.
```bash
--archives environment.tar.gz#environment
```
* For yarn-cluster mode, both driver and executors will use the Python interpreter in `environment.tar.gz`, 
```bash
--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=environment/bin/python \
--conf spark.executorEnv.PYSPARK_PYTHON=environment/bin/python
```
* Set the `py-files` argument to load dependency libraries to cluster and distribute between executors.
```
--py-files ./model.py
```

When you complete all required preparations, you could submit and run the program as below: 
```bash
bigdl-submit \
    --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=environment/bin/python \
    --conf spark.executorEnv.PYSPARK_PYTHON=environment/bin/python \
    --master yarn \
    --deploy-mode cluster \
    --executor-memory 10g \
    --driver-memory 10g \
    --executor-cores 8 \
    --num-executors 2 \
    --archives environment.tar.gz#environment \
    --py-files ./model.py \
    train.py --no-download --remote_dir hdfs://path/to/remote_data --data_dir /tmp/dataset
```

# Run Programs with Spark Submit Script
When the dirver node on the Yarn Cluster is not able to install conda environment, it's recommended for users to use `spark-submit` script to submit and execute the program instead of `bigdl-submit`. Before running with `spark-submit`, we need:
* Install all the dependency files that BigDL required (refer to prepare environment part) on the node which could install conda;
* Pack the conda environment to an archive on the node with conda then send it to the driver node; 
* Download and unzip a BigDL assembly package from [BigDL Release Page](https://bigdl.readthedocs.io/en/latest/doc/release.html);
* Set the unzipped BigDL package as `${BIGDL_HOME}`;
* Set environment variables `${SPARK_VERSION}` and `${BIGDL_VERSION}` as the version in your environment.

### Note:
* Please make sure the `cluster_mode` parameter in `init_orca_context( )` must be `spark-submit` (default) when using `bigdl-submit` to run programs.

## Yarn Client
When running with `spark-submit` script, we need to make preparation with the following steps:
### Runtime Environment Preparation
* Set the driver Python environment to the local Python path, since Spark driver is running on local and it will use the Python interpreter in the current active Conda environment.
```bash
export PYSPARK_DRIVER_PYTHON='which python' # Do not set in yarn cluster modes.
```
* Set the executor Python environment to the path in Conda archive, since executors will use the Python interpreter and relevant libraries in the conda archive.
```bash
export PYSPARK_PYTHON=environment/bin/python
```
### Spark-Submit Script Preparation
* Set the `archives` argument to the path of the archive which was sent from the other node;
```bash
--archives environment.tar.gz#environment \
```
* Set the `properties-file` argument to override spark configuration by BigDL configuration file;
```bash
--properties-file ${BIGDL_HOME}/conf/spark-bigdl.conf \
```
* Set the `py-files` argument as the BigDL Python zip file to load and distribute required dependency libraries in the cluster;
```bash
--py-files ${BIGDL_HOME}/python/bigdl-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,model.py \
```
* Set the `spark.driver.extraClassPath` argument to register the BigDL jars files to the classpath of driver;
```bash
--conf spark.driver.extraClassPath=${BIGDL_HOME}/jars/* \
```
* Set the `spark.executor.extraClassPath` argument to register the BigDL jars files to the classpath of executors;
```bash
--conf spark.executor.extraClassPath=${BIGDL_HOME}/jars/* \
```

Once all preparation works are completed, you could submit and execute the program with `spark-submit` script as below:
```bash
spark-submit \
    --properties-file ${BIGDL_HOME}/conf/spark-bigdl.conf \
    --py-files ${BIGDL_HOME}/python/bigdl-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,model.py \
    --conf spark.driver.extraClassPath=${BIGDL_HOME}/jars/* \
    --conf spark.executor.extraClassPath=${BIGDL_HOME}/jars/* \
    --master yarn \
    --deploy-mode client \
    --executor-memory 10g \
    --driver-memory 10g \
    --executor-cores 8 \
    --num-executors 2 \
    --archives environment.tar.gz#environment \
    train.py
```

## Yarn-Cluster
When running programs on yarn cluster mode with `spark-submit` script, you could refer to the following steps:
### Spark-Submit Script Preparation
* Set the `archives` argument to the path of the Conda archive, which will be upload to remote clusters(i.e. HDFS) and distributed between all executors.
```bash
--archives environment.tar.gz#environment
```
* For yarn-cluster mode, both driver and executors will use the Python interpreter in `environment.tar.gz`, 
```bash
--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=environment/bin/python \
--conf spark.executorEnv.PYSPARK_PYTHON=environment/bin/python
```
* Set the `py-files` argument to load and distributed BigDL and other dependency libraries on yarn cluster.
```bash
--py-files ${BIGDL_HOME}/python/bigdl-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,model.py
```
* This tutorial was mainly based on a BigDL Orca example, please download jars files from [BigDL Dllib Jars](https://repo1.maven.org/maven2/com/intel/analytics/bigdl/bigdl-dllib-spark_2.4.6/2.0.0/bigdl-dllib-spark_2.4.6-2.0.0-jar-with-dependencies.jar) and [BigDL Orca Jars](https://repo1.maven.org/maven2/com/intel/analytics/bigdl/bigdl-orca-spark_2.4.6/2.0.0/bigdl-orca-spark_2.4.6-2.0.0-jar-with-dependencies.jar) separately, then set the `jars` argument as below to register and transfer the BigDL jars files to the cluster.
```bash
--jars ${BIGDL_HOME}/bigdl-dllib-spark_${SPAKR_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar,${BIGDL_HOME}/bigdl-orca-spark_${SPAKR_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar
```
### Note:
* Do not set `${PYSPARK_DRIVER_PYTHON}` when running programs on yarn-cluster mode.

```bash
spark-submit \
    --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=environment/bin/python \
    --conf spark.executorEnv.PYSPARK_PYTHON=environment/bin/python \
    --archives environment.tar.gz#environment \
    --py-files ${BIGDL_HOME}/python/bigdl-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,model.py \
    --jars ${BIGDL_HOME}/bigdl-dllib-spark_${SPAKR_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar,${BIGDL_HOME}/bigdl-orca-spark_${SPAKR_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar \
    --master yarn \
    --deploy-mode cluster \
    --executor-memory 10g \
    --driver-memory 10g \
    --executor-cores 4 \
    --num-executors 2 \
    train.py
```