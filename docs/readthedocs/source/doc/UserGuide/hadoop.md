# Hadoop/YARN User Guide

Hadoop version: Apache Hadoop >= 2.7 (3.X included) or [CDH](https://www.cloudera.com/products/open-source/apache-hadoop/key-cdh-components.html) 5.X. CDH 6.X have not been tested and thus currently not supported.

---

For _**Scala users**_, please see [Scala User Guide](./scala.md) for how to run BigDL on Hadoop/YARN clusters.  

For _**Python users**_, you can run BigDL programs on standard Hadoop/YARN clusters without any changes to the cluster (i.e., no need to pre-install BigDL or other Python libraries on all nodes in the cluster).

### **1. Prepare Python Environment**

- You need to first use [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to prepare the Python environment _**on the local machine**_ where you submit your application. Create a conda environment, install BigDL and all the needed Python libraries in the created conda environment:

  ```bash
  conda create -n bigdl python=3.7  # "bigdl" is conda environment name, you can use any name you like.
  conda activate bigdl

  pip install bigdl

  # Use conda or pip to install all the needed Python dependencies in the created conda environment.
  ```
  View the [Python User Guide](./python.md) for more details for BigDL installation.

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

- **For CDH users**

  If your CDH cluster has already installed Spark, the CDH's Spark might be conflict with the pyspark installed by pip required by BigDL.

  Thus before running BigDL applications, you should unset all the Spark related environment variables. You can use `env | grep SPARK` to find all the existing Spark environment variables.

  Also, a CDH cluster's `HADOOP_CONF_DIR` should be `/etc/hadoop/conf` on CDH by default.

---
### **2. Run on YARN with built-in function**

_**This is the easiest and most recommended way to run BigDL on YARN,**_ as you don't need to care about environment preparation and Spark related commands. In this way, you can easily switch your job between local (for test) and YARN (for production) by changing the "cluster_mode".

- Call `init_orca_context` at the very beginning of your code to initiate and run BigDL on standard [Hadoop/YARN clusters](https://spark.apache.org/docs/latest/running-on-yarn.html#launching-spark-on-yarn):

  ```python
  from bigdl.orca import init_orca_context

  sc = init_orca_context(cluster_mode="yarn-client", cores=4, memory="10g", num_nodes=2)
  ```

  `init_orca_context` would automatically prepare the runtime Python environment, detect the current Hadoop configurations from `HADOOP_CONF_DIR` and initiate the distributed execution engine on the underlying YARN cluster. View [Orca Context](../Orca/Overview/orca-context.md) for more details.    
  
  By specifying "cluster_mode" to be `yarn-client` or `yarn-cluster`, `init_orca_context` will submit the job to YARN with client and cluster mode respectively.  
  
  The difference between `yarn-client` and `yarn-cluster` is where you run your Spark driver. For `yarn-client`, the Spark driver will run on the node where you start Python, while for `yarn-cluster` the Spark driver will run on a random node in the YARN cluster. So if you are running with `yarn-cluster`, you should change the application's data loading from local file to a network file system (e.g. HDFS).  

- You can then simply run your BigDL program in a Jupyter notebook. Note that _**jupyter cannot run on yarn-cluster**_, as the driver is not running on the local node.

  ```bash
  jupyter notebook --notebook-dir=./ --ip=* --no-browser
  ```

  Or you can run your BigDL program as a normal Python script (e.g. script.py) and in this case both `yarn-client` and `yarn-cluster` are supported.

  ```bash
  python script.py
  ```

---
### **3. Run on YARN with spark-submit**

Follow the steps below if you need to run BigDL with [spark-submit](https://spark.apache.org/docs/latest/running-on-yarn.html#launching-spark-on-yarn).  

- Pack the current active conda environment to `environment.tar.gz` (you can use any name you like) in the current working directory:

  ```bash
  conda pack -o environment.tar.gz
  ```

- _**You need to write your BigDL program as a Python script.**_ In the script, you need to call `init_orca_context` at the very beginning of your code and specify "cluster_mode" to be `spark-submit`:

  ```python
  from bigdl.orca import init_orca_context

  sc = init_orca_context(cluster_mode="spark-submit")
  ```

- Use `spark-submit-with-bigdl` to submit your BigDL program (e.g. script.py). You can adjust the configurations according to your cluster settings. Note that if `environment.tar.gz` is not under the same directory with `script.py`, you may need to modify its path in `--archives` in the running command below.

  For `yarn-cluster` mode:
  ```bash
  spark-submit-with-bigdl \
      --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=environment/bin/python \
      --conf spark.executorEnv.PYSPARK_PYTHON=environment/bin/python \
      --master yarn \
      --deploy-mode cluster \
      --executor-memory 10g \
      --driver-memory 10g \
      --executor-cores 8 \
      --num-executors 2 \
      --archives environment.tar.gz#environment \
      script.py
  ```
  Note: For `yarn-cluster`, the Spark driver is running in a YARN container as well and thus both the driver and executors will use the Python interpreter in `environment.tar.gz`. If you want to operate HDFS as some certain user, you can add `spark.yarn.appMasterEnv.HADOOP_USER_NAME=username` to SparkConf.


  For `yarn-client` mode:
  ```bash
  PYSPARK_PYTHON=environment/bin/python spark-submit-with-bigdl \
      --master yarn \
      --deploy-mode client \
      --executor-memory 10g \
      --driver-memory 10g \
      --executor-cores 8 \
      --num-executors 2 \
      --archives environment.tar.gz#environment \
      script.py
  ```
  Note: For `yarn-client`, the Spark driver is running on local and it will use the Python interpreter in the current active conda environment while the executors will use the Python interpreter in `environment.tar.gz`.
