# Hadoop/YARN User Guide

Hadoop version: Hadoop >= 2.7 or CDH 5.X, Hadoop 3.X or CHD 6.X are not supported

---

You can run Analytics Zoo programs on standard Hadoop/YARN clusters without any changes to the cluster (i.e., no need to pre-install Analytics Zoo or any Python libraries in the cluster).

### **1. Prepare Environment**

- You need to first use [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to prepare the Python environment _**on the local client machine**_. Create a conda environment and install all the needed Python libraries in the created conda environment:

  ```bash
  conda create -n zoo python=3.7  # "zoo" is conda environment name, you can use any name you like.
  conda activate zoo

  # Use conda or pip to install all the needed Python dependencies in the created conda environment.
  ```

- You need to download and install JDK in the environment, and properly set the environment variable `JAVA_HOME`, which is required by Spark. __JDK8__ is highly recommended.

  You may take the following commands as a reference for installing [OpenJDK](https://openjdk.java.net/install/):

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

- Check the Hadoop setup and configurations of your cluster. Make sure you properly set the environment variable `HADOOP_CONF_DIR`, which is needed to initialize Spark on YARN:

  ```bash
  export HADOOP_CONF_DIR=the directory of the hadoop and yarn configurations
  ```

### **1.1 Setup for CDH**

CDH Version: CDH 5.X, CDH 6.X is not supported

---
### **2. YARN Client Mode**

- Install Analytics Zoo in the created conda environment via pip:

  ```bash
  pip install analytics-zoo
  ```

  View the [Python User Guide](./python.md) for more details.
  

- We recommend using `init_orca_context` at the very beginning of your code to initiate and run Analytics Zoo on standard Hadoop/YARN clusters in [YARN client mode](https://spark.apache.org/docs/latest/running-on-yarn.html#launching-spark-on-yarn):

  ```python
  from zoo.orca import init_orca_context

  sc = init_orca_context(cluster_mode="yarn-client", cores=4, memory="10g", num_nodes=2)
  ```

  By specifying cluster_mode to be "yarn-client", `init_orca_context` would automatically prepare the runtime Python environment, detect the current Hadoop configurations from `HADOOP_CONF_DIR` and initiate the distributed execution engine on the underlying YARN cluster. View [Orca Context](../Orca/Overview/orca-context.md) for more details.
  

- You can then simply run your Analytics Zoo program in a Jupyter notebook:

  ```bash
  jupyter notebook --notebook-dir=./ --ip=* --no-browser
  ```

  or as a normal Python script (e.g. script.py):

  ```bash
  python script.py
  ```

---
### **3. YARN Cluster Mode**

Follow the steps below if you need to run Analytics Zoo in [YARN cluster mode](https://spark.apache.org/docs/latest/running-on-yarn.html#launching-spark-on-yarn).

- Download and extract [Spark](https://spark.apache.org/downloads.html). You are recommended to use [Spark 2.4.3](https://archive.apache.org/dist/spark/spark-2.4.3/spark-2.4.3-bin-hadoop2.7.tgz). Set the environment variable `SPARK_HOME`:

  ```bash
  export SPARK_HOME=the root directory where you extract the downloaded Spark package
  ```

- Download and extract [Analytics Zoo](../release.md). Make sure the Analytics Zoo package you download is built with the compatible version with your Spark. Set the environment variable `ANALYTICS_ZOO_HOME`:

  ```bash
  export ANALYTICS_ZOO_HOME=the root directory where you extract the downloaded Analytics Zoo package
  ```

- Pack the current conda environment to `environment.tar.gz` (you can use any name you like):

  ```bash
  conda pack -o environment.tar.gz
  ```

- _You need to write your Analytics Zoo program as a Python script._ In the script, you can call `init_orca_context` and specify cluster_mode to be "spark-submit":

  ```python
  from zoo.orca import init_orca_context

  sc = init_orca_context(cluster_mode="spark-submit")
  ```

- Use `spark-submit` to submit your Analytics Zoo program (e.g. script.py):

  ```bash
  PYSPARK_PYTHON=./environment/bin/python ${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
      --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./environment/bin/python \
      --master yarn-cluster \
      --executor-memory 10g \
      --driver-memory 10g \
      --executor-cores 8 \
      --num-executors 2 \
      --archives environment.tar.gz#environment \
      script.py
  ```

  You can adjust the configurations according to your cluster settings.
