# Running BigDL-Orca Program on YARN

This tutorial provides a step-by-step guide on how to run BigDL-Orca programs on Apache Hadoop (YARN) clusters, using a [PyTorch Fashin-MNIST program](https://github.com/intel-analytics/BigDL/tree/main/docs/docs/tutorials/tutorial_example/Fashion_MNIST/) as a working example.

# 1. Key Concepts
## 1.1 Init_orca_context
A BigDL Orca program usually starts with the initialization of OrcaContext. For every BigDL Orca program, you should call `init_orca_context` at the beginning of the program as below:

```python
from bigdl.orca import init_orca_context

init_orca_context(...)
```

In `init_orca_context`, the user may specify necessary runtime configurations for the Orca program, including:

* Cluster mode: Users can specify the computing environment for the program (a local machine, K8s cluster, Hadoop/YARN cluster, etc.), `init_orca_context` will create or get an OrcaContext with optimized configurations for BigDL performance, which will automatically provision the runtime Python environment and distributed execution engine.

* Runtime: Users can specify the backend for the program (spark and ray, etc.) to create SparkContext and/or RayContext, the cluster mode would work based on the specified runtime backend.

* Physical resources: Users can specify the amount of physical resources to be allocated for the program on the underlying cluster, including the number of nodes in the cluster, the cores and memory allocated for each node, etc.

For more details, please see [OrcaContext](https://bigdl.readthedocs.io/en/latest/doc/Orca/Overview/orca-context.html) .

## 1.2 Yarn-Client & Yarn-Cluster
The difference between yarn-client and yarn-cluster is where you run your Spark driver. 

For yarn-client, the Spark driver runs in the client process, and the application master is only used for requesting resources from YARN, while for yarn-cluster the Spark driver runs inside an application master process which is managed by YARN on the cluster. If you are running on a production environment(especially yarn-cluster), you should load data from a network file system (e.g. HDFS).

For more details, please see [Launching Spark on YARN](https://spark.apache.org/docs/latest/running-on-yarn.html).

# 2. Prepare Environment
Before running the BigDL program on Yarn, we need to setup the environment as the following steps:
* Setup JAVA & Hadoop Environment
* Install All Needed Python Libraries 
* For CDH Users

## 2.1 Setup JAVA & Hadoop Environment
We need to download and install JDK in the environment, and properly set the environment variable JAVA_HOME, which is required by Spark. JDK8 is highly recommended.

### 2.1.1 Setup JAVA Environment
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

### 2.1.2 Setup Hadoop Environment
Check the Hadoop setup and configurations of our cluster. Make sure we correctly set the environment variable `HADOOP_CONF_DIR`, which is needed to initialize Spark on YARN:
```bash
export HADOOP_CONF_DIR=/path/to/hadoop/conf
```

## 2.2 Install Needed Python Libraries
### 2.2.1 Install and Create Conda Environment
We need first to use conda to prepare the Python environment on the local machine where we submit our application. Please download and install Conda with the following commands:
```bash
# Download Anaconda installation script 
wget -P /tmp https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh

# Execute the script to install conda
bash /tmp/Anaconda3-2020.02-Linux-x86_64.sh

# Please type this command in your terminal to activate Conda environment
source ~/.bashrc
``` 
Create a conda environment, install BigDL and all the needed Python libraries in the created conda environment:
``` bash
conda create -n bigdl python=3.7  # "bigdl" is conda environment name, you can use any name you like.
conda activate bigdl
```

### 2.2.2 Use Conda to Install Python Libraries
You can install the latest release version of BigDL (built on top of Spark 2.4.6 by default) as follows:
```bash
pip install bigdl
```
You can install the latest nightly build of BigDL as follows:
```bash
pip install --pre --upgrade bigdl
```
You can install BigDL built on top of Spark 3.1.2 as follows:
```bash
pip install bigdl-spark3  # Install the latest release version
pip install --pre --upgrade bigdl-spark3  # Install the latest nightly build version
```
When you are running a program on Ray backend, please install BigDL with Ray as below:
```bash
pip install bigdl[ray]
```
Install BigDL will automatically install `pyspark` and `torch`. Please see more details in [Python User Guide](https://bigdl.readthedocs.io/en/latest/doc/UserGuide/python.html).

### 2.2.3 Install Extra Python Libraries
* When you are running a TensorFlow2 model, please install TensorFlow as below:
```bash
pip install tensorflow==2.6.0 keras==2.6.0 # When you are running a tensorflow model.
```

## 2.3 For CDH Users
If your CDH cluster has already installed Spark, the CDH’s Spark might be conflict with the pyspark installed by pip required by BigDL. Thus before running BigDL applications, you should unset all the Spark related environment variables. You can use `env | grep SPARK` to find all the existing Spark environment variables.

Also, a CDH cluster’s `HADOOP_CONF_DIR` should be `/etc/hadoop/conf` on CDH by default.

# 3. Prepare Dataset & Dependency
## 3.1 Prepare Dataset  
In this tutorial, the Fasion-MNIST program supports downloading Fashion-MNIST data directly when running program or loading data from network file systems (e.g. HDFS and S3).
### 3.1.1 Use Remote Dataset
When you are running the program on yarn-cluster, you should use a network file system instead of downloading dataset, since it's possible to get a network error when downloading dataset.

First you need to download the Fashion-MNIST dataset manually on your local machine.
```bash
# PyTorch official dataset download link
git clone https://github.com/zalandoresearch/fashion-mnist.git

mv /path/to/fashion-mnist/data/fashion /path/to/local/data/FashionMNIST/raw 
```
After downloading the dataset, you need to upload it to a network file system (HDFS or S3).
```bash
hdfs dfs -put /path/to/local/data/FashionMNIST hdfs://path/to/remote/data
```
Please set the dataset path on the network file system through `--remote-dir` argument when submmiting and running the program on yarn-cluster, the dataset would be downloaded and processing on the driver.
```bash
python train.py --cluster_mode yarn-cluster --remote-dir hdfs://path/to/remote/data
```

## 3.2 Prepare Python Dependency
Spark allows to upload Python files(`.py`), and zipped Python packages(`.zip`) to the executors by setting `--py-files` option in Spark scripts or `extra_python_lib` in `init_orca_context`, which is a straightforward method to load and distribute additional custom Python dependency to the cluster. If you depend on a nested directory of python files, we recommend packaging them into a zip file.
### 3.2.1 Use Zip File 
To import from a zip file, you need to create a package like `orca_exmaple`, with the `__init__.py` and all dependency Python files. 
```bash
orca_example.zip
|
| -- orca_example <dir>
     | -- __init__.py
     | -- model.py
``` 
Add the code as below in `__init__.py` that allowed `train.py` to access modules you need.
```python
# In the __init__.py
from orca_example.model import import model_creator, optimizer_creator
```
Then zip the whole package.
```bash
zip -q -r orca_example.zip orca_example
```
### 3.2.2 Load Python Dependency
You should import the modules as below in `train.py`. The zip file will be automatically unzipped after uploading and distributing in the cluster, so you need to load modules from the unzipped file. 
```python
# Import dependency from zip file
from orca_example import model_creator, optimizer_creator
```
When running this example using `bigdl-submit` or `spark-submit`, you need to upload the zipped file in Spark scripts through `--py-files` to the cluster.`init_orca_context`.
```bash
--py-files /path/to/orca_example.zip
```
When using `python` command to run the example, please set the `extra_python_lib` in `init_orca_context` to the path of zipped Python file. Note you should import modules after creating OrcaContext, since the zip file will be uploaded through `init_orca_context`.
```python
from bigdl.orca import init_orca_context

# Please switch the `cluster_mode` option to `yarn-cluster` if you need to run program on yarn-cluster mode.
init_orca_context(cluster_mode="yarn-client", cores=4, memory="10g", num_nodes=2, 
                  extra_python_lib="/path/to/orca_example.zip")

# Note please import modules after init_orca_context()
from orca_example import model_creator, optimizer_creator
```

# 4. Run Jobs on Yarn
In the following part, we will show you three ways that BigDL supports to submit and run our programs on Yarn:
* Use `python` command
* Use `bigdl-submit`
* Use `spark-submit`

## 4.1 Use `python` Command
This is the easiest and most recommended way to run BigDL on YARN, you only need to prepare the environment on the driver machine, all dependencies would be automatically packaged and distributed to the whole Yarn cluster. In this way, you can easily switch your job between local (for test) and YARN (for production) by changing the `cluster_mode` to be yarn-client or yarn-cluster, `init_orca_context` will submit the job to YARN with client or cluster mode respectively.

### 4.1.1 Yarn Client
When running every Orca program on yarn-client mode using built-in function, we should call `init_orca_context` to create an OrcaContext at the very beginning of our programs.
```python
from bigdl.orca import init_orca_context

init_orca_context(cluster_mode="yarn-client", cores=4, memory="10g", num_nodes=2, 
                  extra_python_lib="/path/to/orca_example.zip")
```
To execute the tutorial provided Fashion-MINIST example, we should set `--cluster_mode` argument as `yarn-client` to create an OrcaContext and run the program on yarn-client mode as below:
```bash
python train.py --cluster_mode yarn-client
```

### 4.1.2 Yarn Cluster
Fisrt, please call `init_orca_context` to create an OrcaContext at the very beginning of Orca programs. To run programs on yarn cluster mode using built-in function, we should switch the `cluster_mode` to `yarn-cluster` in `init_orca_context` as below:
```python
from bigdl.orca import init_orca_context

init_orca_context(cluster_mode="yarn-cluster", cores=4, memory="10g", num_nodes=2, 
                  extra_python_lib="/path/to/orca_example.zip")
```
When running the program provided by tutorial, we should:
* Set `--cluster_mode` parameter to `yarn-cluster`, which could create an OrcaContext and run the Fasion-MNIST program on yarn-cluster mode.
* Set `--remote_dir` parameter to the datasets path on remote resources. We recommand you to use datasets from remote resources like HDFS and S3 to instead of downloading these datasets, since there may get connection errors on `yarn-cluster` mode.

Now, please run the example as the following script:
```bash
python train.py --cluster_mode yarn-cluster --remote_dir hdfs://path/to/remote/data
```

### 4.1.3 Jupyter Notebook
You can simply run the program in a Jupyter Notebook. Please note that jupyter cannot run on yarn-cluster, as the driver is not running on the local node.
```bash
# Start a jupyter notebook
jupyter notebook --notebook-dir=/path/to/notebook/directory --ip=* --no-browser
```

## 4.2 Use `bigdl-submit`
For Spark script users, BigDL provides a `bigdl-submit` script, you can find it in `${BIGDL_HOME}/scripts/bigdl-submit` and adjust the configurations according to your cluster settings. 
```bash
# bigdl-submit script
spark-submit \
  --properties-file ${BIGDL_CONF} \
  --jars ${BIGDL_JARS} \
  --conf spark.driver.extraClassPath=${BIGDL_JARS} \
  --conf spark.executor.extraClassPath=${BIGDL_JARS} \
  $*
```
In the `bigdl-submit` script:
* It could automatically detect and setup configuration and jars files from the current conda environment, which will be set as `${BIGDL_CONF}` and `${BIGDL_JARS}` seperately. There is no need for you to setup these varaibles manully;
* `--properties-file`: upload the BigDL configuration properties to the cluster;
* `--jars`: register and distribute BigDL dependency jars files to the Spark job;
* `--conf spark.driver.extraClassPath`: register and upload the BigDL jars files to the driver classpath;
* `--conf spark.executor.extraClassPath`: register and upload the BigDL jars files to the classpath of executors.

Before submitting the Fasion-MNIST program to Yarn with `bigdl-submit`, we need to pack the Conda environment to an archive, which captures the Conda environment for Python and stores both Python interpreter and all its relevant dependencies.
```bash
conda pack -o environment.tar.gz#environment 
```
### Note:
* If `environment.tar.gz` is not under the same directory with `train.py`, we should modify its path in `--archives` in the running command.
```
--archives /path/to/environment.tar.gz#environment
```

### 4.2.1 Yarn Client
The first thing we need to make sure is that the `cluster_mode` in `init_orca_context` must be `bigdl-submit` when using `bigdl-submit` to run Orca programs on yarn-client mode.
```python
from bigdl.orca import init_orca_context

init_orca_context(cluster_mode="bigdl-submit")
```
Now, let's submit and execute the BigDL program on yarn-client with the following `bigdl-submit` script:
```bash
bigdl-submit \
    --master yarn \
    --deploy-mode client \
    --executor-memory 10g \
    --driver-memory 10g \
    --executor-cores 8 \
    --num-executors 2 \
    --py-files orca_example.zip \
    --archives environment.tar.gz#environment \
    --conf spark.pyspark.driver.python=/path/to/python \
    --conf spark.pyspark.python=environment/bin/python \
    train.py --cluster_mode bigdl-submit
```
In the `bigdl-submit` script for running Orca programs on yarn-client mode:
* `--master`: the spark master, set it to yarn when running programs on Yarn;
* `--deploy-mode`: submit and execute programs on yarn-client or yarn-cluster, set it to client when running programs on yarn-client mode;
* `--archives`: set this option to the path of the Conda archive, the archive will be uploaded to remote clusters(i.e. HDFS) and distributed between executors;
* `--py-files`: upload Python dependency file(s) to cluster and distribute between executors;
* `--conf spark.pyspark.driver.python`: set the driver Python environment as the local Python path. For `yarn-client` mode, the Spark driver is running on local and it will use the Python interpreter in the current active conda environment;
* `--conf spark.pyspark.python`: set the executor Python environment to the Python path in the Conda archive, since executors will use the Python interpreter in the conda archive.
* `--cluster_mode`: set the cluster_mode in `init_orca_context` to get or create an OrcaContext.

### 4.2.2 Yarn Cluster
To run every Orca program on yarn cluster with `bigdl-submit`, we need to call `init_orca_context` at the very beginning of the program and specify the `cluster_mode` to `bigdl-submit` to create or get an OrcaContext for yarn-cluster mode.
```python
from bigdl.orca import init_orca_context

init_orca_context(cluster_mode="bigdl-submit")
```
When you complete all required preparations, please submit and run the program as below: 
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
    --py-files orca_example.zip \
    train.py --cluster_mode bigdl-submit --remote_dir hdfs://path/to/remote/data
```
In the `bigdl-submit` script for running Orca programs on yarn-cluster mode:
* `--master`: the spark master, set it to yarn when running programs on Yarn;
* `--deploy-mode`: submit and execute programs on yarn-client or yarn-cluster, set it to cluster when running programs on yarn-cluster mode;
* `--archives`: set the option to the path of the Conda archive, which will be upload to remote resources(i.e. HDFS) and distributed between executors;
* `--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON`: set the Python environment for the Application Master process launched on YARN, it will use the Python interpreter in `environment.tar.gz`;
* `--conf spark.executorEnv.PYSPARK_PYTHON`: set the Python environment for executors on Yarn, it will use the Python interpreter in `environment.tar.gz`;
* `--py-files`: upload Python dependency file(s) to cluster and distribute between executors;
* `--cluster_mode`: set the cluster_mode in `init_orca_context` to get or create an OrcaContext;
* `--remote_dir`: loda data from remote datasets path. For yarn-cluster mode, we recommend that you should use the datasets stored in remote resources like HDFS or S3 instead of downloading to avoid possible network connection errors.

## 4.3 Use `spark-submit`
When the client node where users submit the job on Yarn is not able to install BigDL and other dependency libraries using conda, it's recommended for users to use `spark-submit` script instead of `bigdl-submit`. Before running with `spark-submit`, we need:
* Setup spark environment first since we cannot use pyspark on driver, please set environment variables `${SPARK_HOME}` as follows:
```bash
export SPARK_HOME=/path/to/spark # the folder path where you extract the Spark package
```
* Install all the dependency files that BigDL required (please refer to prepare environment part), pack the Conda environment to an archive and send it to the Spark driver on a different node which could use conda;
* Download and unzip a BigDL assembly package from [BigDL Release Page](https://bigdl.readthedocs.io/en/latest/doc/release.html), set the unzipped BigDL file path as `${BIGDL_HOME}`;
* Set environment variables `${SPARK_VERSION}` and `${BIGDL_VERSION}` as the version in your environment.

### 4.3.1 Yarn Client
we need to call `init_orca_context` at the very beginning of the program and specify the `cluster_mode` to `spark-submit` to create or get an OrcaContext for yarn-cluster mode.
```python
from bigdl.orca import init_orca_context

init_orca_context(cluster_mode="spark-submit")
```
Please submit and execute the program with `spark-submit` script on yarn-client mode as below:
```bash
spark-submit \
    --properties-file ${BIGDL_HOME}/conf/spark-bigdl.conf \
    --py-files ${BIGDL_HOME}/python/bigdl-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,orca_example.zip \
    --conf spark.pyspark.driver.python=/path/to/python \
    --conf spark.pyspark.python=environment/bin/python \
    --conf spark.driver.extraClassPath=${BIGDL_HOME}/jars/* \
    --conf spark.executor.extraClassPath=${BIGDL_HOME}/jars/* \
    --master yarn \
    --deploy-mode client \
    --executor-memory 10g \
    --driver-memory 10g \
    --executor-cores 8 \
    --num-executors 2 \
    --archives environment.tar.gz#environment \
    train.py --cluster_mode spark-submit
```
In this script:
* `--master`: the spark master, set it to yarn when running programs on Yarn;
* `--deploy-mode`: submit and execute programs on yarn-client or yarn-cluster, set it to client when running programs on yarn-client mode;
* `--archives`: set the option to the path of the archive which was sent from the other node;
* `--properties-file`: upload the BigDL configuration properties to the cluster;
* `--py-files`: upload Python dependency file(s) to cluster and distribute between executors;
* `--conf spark.pyspark.driver.python`: set the driver Python environment to the local Python path, since Spark driver is running on local and it will use the Python interpreter in the current active Conda environment;
* `--conf spark.pyspark.python`: set the executor Python environment to the Python path in Conda archive, since executors will use the Python interpreter and relevant libraries in the conda archive;
* `--conf spark.driver.extraClassPath`: load and register the BigDL jars files to the driver's classpath;
* `--conf spark.executor.extraClassPath`: load and register the BigDL jars files to the executors' classpath.
* `--cluster_mode`: set the cluster_mode in `init_orca_context` to get or create an OrcaContext.

### 4.3.2 Yarn-Cluster
We need to call `init_orca_context` first to create an OrcaContext at the very beginning of the program.
```python
from bigdl.orca import init_orca_context

init_orca_context(cluster_mode="spark-submit")
```

This tutorial is mainly based on a BigDL Orca example, please download jars files from [BigDL Dllib Jars](https://repo1.maven.org/maven2/com/intel/analytics/bigdl/bigdl-dllib-spark_2.4.6/2.0.0/bigdl-dllib-spark_2.4.6-2.0.0-jar-with-dependencies.jar) and [BigDL Orca Jars](https://repo1.maven.org/maven2/com/intel/analytics/bigdl/bigdl-orca-spark_2.4.6/2.0.0/bigdl-orca-spark_2.4.6-2.0.0-jar-with-dependencies.jar) separately and distribute them by `--jars` option in the `spark-submit` script when running the program.

Please run the example on yarn-cluster mode with the following script:
```bash
spark-submit \
    --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=environment/bin/python \
    --conf spark.executorEnv.PYSPARK_PYTHON=environment/bin/python \
    --archives environment.tar.gz#environment \
    --py-files ${BIGDL_HOME}/python/bigdl-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,orca_example.zip \
    --jars ${BIGDL_HOME}/bigdl-dllib-spark_${SPAKR_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar,${BIGDL_HOME}/bigdl-orca-spark_${SPAKR_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar \
    --master yarn \
    --deploy-mode cluster \
    --executor-memory 10g \
    --driver-memory 10g \
    --executor-cores 4 \
    --num-executors 2 \
    train.py --cluster_mode spark-submit --remote_dir hdfs://path/to/remote/data
```
In the `spark-submit` script for yarn-cluster:
* `--master`: the spark master, set it to yarn when running programs on Yarn;
* `--deploy-mode`: submit and execute programs on yarn-client or yarn-cluster, set it to cluster when running programs on yarn-cluster mode;
* `--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON`: set the Python environment for the Application Master process launched on YARN, it will use the Python interpreter in `environment.tar.gz`; 
* `--conf spark.executorEnv.PYSPARK_PYTHON`: set the Python environment for executors on Yarn, it will use the Python interpreter in `environment.tar.gz`;
* `--archives`: set the option to the path of the Conda archive, which will be upload to remote clusters(i.e. HDFS) and distributed between all executors;
* `--py-files`: upload Python dependency file(s) to cluster and distribute between executors;
* `--jars`: register and distribute BigDL dependency jars files to the Spark job;
* `--cluster_mode`: set the cluster_mode in `init_orca_context` to get or create an OrcaContext;
* `--remote_dir`: load dataset from remote path. For yarn-cluster mode, we recommend that you use the datasets stored in remote resources like HDFS instead of downloading to avoid possible network connection errors.