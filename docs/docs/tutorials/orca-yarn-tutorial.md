In this tutorial, we will show you:
* How to prepare for the environment before submit your program to Yarn;
* How to run the BigDL program on Yarn on multiple ways;

# Key Concepts (To Do)

* OrcaContext

* Orca Estimator


Let's get started!
# Prepare Environment
Before submitting the BigDL program to Yarn Cluster, we need to setup the environment as the following steps:
* Setup JAVA Environment
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
If your CDH cluster has already installed Spark, the CDH’s Spark might be conflict with the pyspark installed by pip required by BigDL. Thus before running BigDL applications, you should unset all the Spark related environment variables. You can use env | grep SPARK to find all the existing Spark environment variables.

Also, a CDH cluster’s `HADOOP_CONF_DIR` should be `/etc/hadoop/conf` on CDH by default.


# Submit and Execute
BigDL mainly supports 3 ways to submit our job and run it on Yarn:
* Use a built-in function
* Use a BigDL provided script
* Use a spark-submit script

# Run Programs with built-in function
This is the easiest and most recommended way to run BigDL on YARN, we only need to prepare the environment on the driver machine, all dependencies would be automatically packaged and distributed to the whole Yarn cluster.

Note: 
* When running on Yarn-Client mode, the `cluster_mode` of OrcaContext should be `yarn` or `yarn-client`. 
* When running on Yarn-Cluster mode, the `cluster_mode` of OrcaContext should be `yarn-cluster`.

Let's run the program as a python script:
```bash
python script.py
```

# Run Programs with BigDL Provided Scripts
For `spark-submit` users, BigDL provided a `bigdl-submit` script which could automatically detect and setup configuration and jars files from the current conda environment. Before submitting our program to Yarn with `bigdl-submit`, we need to pack the Conda environment to an archive, which captures the Conda environment for Python and stores both Python interpreter and all its relevant dependencies.
```bash
conda pack -o environment.tar.gz#env 
```
Note: If `environment.tar.gz` is not under the same directory with script.py, we should modify its path in `--archives` in the running command below.

## Yarn Client
When running programs with `bigdl-submit`, we need:
* Set the Python environment as the local Python location. For `yarn-client` mode, the Spark driver is running on local and it will use the Python interpreter in the current active conda environment.
```bash
export PYSPARK_DRIVER_PYTHON='which python' # python location on driver
```
* Set the `archives` argument to the location of the Conda archive, which will be upload to remote clusters(i.e. HDFS) and distributed between all executors.
```bash
--archives environment.tar.gz#environment
```
* Set the executor Python environment to the location in the Conda archive, since executors will use the Python interpreter in the conda archive.
```bash
PYSPARK_PYTHON=environment/bin/python
```

Now, let's submit and execute a BigDL program with the following script:
```bash
PYSPARK_PYTHON=environment/bin/python bigdl-submit \
    --master yarn \
    --deploy-mode client \
    --executor-memory 10g \
    --driver-memory 10g \
    --executor-cores 8 \
    --num-executors 2 \
    --archives environment.tar.gz#env \
    script.py
```

# Run with Spark Submit Script
When the dirver node on the Yarn Cluster is not able to install conda environment, it's recommended for us to use `spark-submit` script to run our program instead of `bigdl-submit`. Before running with `spark-submit`, we need:
* Install all the dependency files that BigDL required (refer to prepare environment part) on the node which could install conda;
* Pack the conda environment to an archive on the node with conda then send it to the driver node; 
* Download and unzip a BigDL assembly package from BigDL Release Page;
* Set the location of unzipped BigDL package as `${BIGDL_HOME}`;

## Yarn Client
When running with `spark-submit` script, we need make preparison with the following steps:
* 1. Set the driver Python environment to the local Python location, since Spark driver is running on local and it will use the Python interpreter in the current active Conda environment.
```bash
export PYSPARK_DRIVER_PYTHON='which python' # python location on driver
```
* 2. Set the executor Python environment to the location in the Conda archive, since executors will use the Python interpreter and relevant libraries in the conda archive.
```bash
PYSPARK_PYTHON=environment/bin/python
```
* 3. Set the `archives` argument to the location of the archive which was sent from the other node;
```bash
--archives environment.tar.gz#env
```
* 4. Set the `properties-file` argument to override spark configuration by BigDL configuration file;
```bash
--properties-file ${BIGDL_HOME}/conf/spark-bigdl.conf
```
* 5. Set the `py-files` argument as the BigDL Python zip file to distribute dependency libraries;
```bash
--py-files ${BIGDL_HOME}/python -name bigdl-spark_*-python-api.zip
```
* 6. Set the `spark.driver.extraClassPath` argument to register the BigDL jars files to the classpath of driver;
```bash
--conf spark.driver.extraClassPath=${BIGDL_HOME}/jars/*
```
* 7. Set the `spark.executor.extraClassPath` argument to register the BigDL jars files to the classpath of executors;
```bash
--conf spark.executor.extraClassPath=${BIGDL_HOME}/jars/*
```

Now, let's submit and execute the BigDL program with `spark-submit` script:
```bash
PYSPARK_PYTHON=environment/bin/python spark-submit \
    --properties-file ${BIGDL_HOME}/conf/spark-bigdl.conf \
    --py-files ${BIGDL_HOME}/python -name bigdl-spark_*-python-api.zip \
    --conf spark.driver.extraClassPath=${BIGDL_HOME}/jars/* \
    --conf spark.executor.extraClassPath=${BIGDL_HOME}/jars/* \
    --master yarn \
    --deploy-mode client \
    --executor-memory 10g \
    --driver-memory 10g \
    --executor-cores 8 \
    --num-executors 2 \
    --archives environment.tar.gz#env \
    script.py
```