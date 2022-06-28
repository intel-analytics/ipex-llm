# Running BigDL-Orca Program on YARN

This tutorial provides a step-by-step guide on how to run BigDL-Orca programs on Apache Hadoop (YARN) clusters, using a [PyTorch Fashin-MNIST program](https://github.com/intel-analytics/BigDL/tree/main/docs/docs/tutorials/tutorial_example/Fashion_MNIST/) as a working example.

# 1. Key Concepts
## 1.1 Init_orca_context
A BigDL Orca program usually starts with the initialization of OrcaContext. For every BigDL Orca program, you should call `init_orca_context` at the beginning of the program as below:

```python
# Create or get an OrcaContext on Spark Local
from bigdl.orca import init_orca_context

if cluster_mode == "local": # For local machine
    init_orca_context(cluster_mode="local", cores=2, memory="2g", num_nodes=1)
elif cluster_mode.startswith("yarn"): #  For Hadoop/Yarn cluster
    if cluster_mode == "yarn-client":
        init_orca_context(cluster_mode="yarn-client", cores=2, memory="10g", num_nodes=2, driver_cores=2, driver_memory="4g",
                          extra_python_lib="/path/to/custom_module.py,/path/to/custom_module.zip", conf={"spark.driver.extraJavaOptions": "-Dbigdl.failure.retryTimes=1"})
    elif cluster_mode == "yarn-custer":
        init_orca_context(cluster_mode="yarn-cluster", cores=2, memory="10g", num_nodes=2, driver_cores=2, driver_memory="4g", 
                          extra_python_lib="/path/to/custom_module.py,/path/to/custom_module.zip", conf={"spark.driver.extraJavaOptions": "-Dbigdl.failure.retryTimes=1"})
elif cluster_mode == "bigdl-submit":
    init_orca_context(cluster_mode="bigdl-submit")
elif cluster_mode == "spark-submit":
    init_orca_context(cluster_mode="spark-submit")
```
`init_orca_context` will create or get an OrcaContext with optimized configurations for BigDL performance, which will automatically provision the runtime Python environment and distributed execution engine.

In `init_orca_context`, you may specify necessary runtime configurations for the Orca program, including:
* `cluster_mode`: The mode for Spark cluster. One of `local`, `yarn-client`, `yarn-cluster`, `k8s-client`, `k8s-cluster`, `standalone`, `spark-submit`, `bigdl-submit`, etc. When you are running programs on Yarn, you should specify `cluster_mode` as below:
    * `local`: when you are running Orca programs on Spark local.
    * When you are using `python` command to run Orca programs on Yarn:
        * `yarn-client`: Run Orca programs on yarn-client mode.
        * `yarn-cluster`: Run Orca programs on yarn-client mode.
    * `bigdl-submit`: when you are using `bigdl-submit` command to run Orca programs on Yarn.
    * `spark-submit`: when you are using `spark-submit` command to run Orca programs on Yarn.
* When you are using `python` command to run Orca programs on Yarn, you may also specify the following arguments: 
    * `cores`: The number of cores for each executor.  Default to be `2`.
    * `memory`: The memory for each executor. Default to be `2g`.
    * `num_nodes`: The number of Spark executors. Default to be `1`.
    * `driver_cores`: The number of cores for the Spark driver. Default to be `4`.
    * `driver_memory`: The memory for the Spark driver. Default to be `1g`.
    * `extra_python_lib`: Add `.py`, `.zip` or `.egg` files to distribute with Orca application. If you depend on multiple Python files, we recommend packaging them into a `.zip` or `.egg`, these files will be added to the Python search path on each nodes in the cluster. Default to be `None`.
    * `conf`: You could append extra conf for Spark in key-value format. Default to be `None`.

Note: You only need to specify `cluster_mode` when using `bigdl-submit` or `spark-submit` to run Orca programs.

After the Orca programs finished, you should call `stop_orca_context` at the end of the program to release resources and shutdown the underlying Spark execution engine.
```python
from bigdl.orca import stop_orca_context

stop_orca_context()
```

For more details, please see [OrcaContext](https://bigdl.readthedocs.io/en/latest/doc/Orca/Overview/orca-context.html).

## 1.2 Yarn-Client & Yarn-Cluster
The difference between yarn-client and yarn-cluster is where you run your Spark driver. 

For yarn-client, the Spark driver runs in the client process, and the application master is only used for requesting resources from YARN, while for yarn-cluster the Spark driver runs inside an application master process which is managed by YARN on the cluster.

For more details, please see [Launching Spark on YARN](https://spark.apache.org/docs/latest/running-on-yarn.html#launching-spark-on-yarn ).

# 2. Prepare Environment
Before running the BigDL program on Yarn, we need to setup the environment as the following steps:
* Setup JAVA & Hadoop Environment
* Install All Needed Python Libraries 
* For Spark Installed Cluster(CDH Users)

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
We need first to use conda to prepare the Python environment on the local machine where we submit our application. You could download and install Conda following [Conda User Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) or executing the command as below.
```bash
# Download Anaconda installation script 
wget -P /tmp https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh

# Execute the script to install conda
bash /tmp/Anaconda3-2020.02-Linux-x86_64.sh

# Please type this command in your terminal to activate Conda environment
source ~/.bashrc
``` 

### 2.2.2 Use Conda to Install BigDL
Create a conda environment, install BigDL and all the needed Python libraries in the created conda environment:
``` bash
# "bigdl" is conda environment name, you can use any name you like.
# Please change Python version to 3.6 if you need a Python 3.6 environment.
conda create -n bigdl python=3.7 
conda activate bigdl
```
You can install the latest release version of BigDL (built on top of Spark 2.4.6 by default) as follows:
```bash
pip install bigdl
```
You can install the latest nightly build of BigDL as follows:
```bash
pip install --pre --upgrade bigdl
```
Using Conda to install BigDL will automatically install libraries including:
```bash
conda-pack              0.3.1
numpy                   1.21.6
pandas                  1.2.5
pyarrow                 7.0.0
pyspark                 2.4.6
pytorch-lightning       1.4.2
tensorboard             2.9.1
torch                   1.9.0
torchmetrics            0.9.1
torchvision             0.10.0
redis                   4.3.3
scikit-learn            1.0.2
...
```
You can install BigDL built on top of Spark 3.1.2 as follows:
```bash
pip install bigdl-spark3  # Install the latest release version
pip install --pre --upgrade bigdl-spark3  # Install the latest nightly build version
```
* Note: Using Conda to install `bigdl-spark3` will automatically install `pyspark==3.1.2`.

When you are running a program on Ray backend, please install BigDL Orca with Ray as below:
```bash
pip install bigdl-orca[ray] # Install the latest release version
pip install --pre --upgrade bigdl-orca[ray] # Install the latest nightly build version
```
You can install BigDL Orca with Ray built on top of Spark 3.1.2 as follows:
```bash
pip install bigdl-orca-spark3[ray]  # Install the latest release version
pip install --pre --upgrade bigdl-orca-spark3[ray]  # Install the latest nightly build version
```
* Note: 
    * Using Conda to install `bigdl-orca[ray]` and `bigdl-orca-spark3[ray]` will automatically install `ray[default]==1.9.2` and `pyspark==3.1.2`.
    * `bigdl-orca` and `bigdl-orca-spark3[ray]` will not automatically install `torch`, `torchmetrics` and `torchvision`, you need to manually install these libraries when you are running a PyTorch program. 
        ```bash
        # Install torch version as you need
        pip install torch
        pip install torchmetrics
        pip install torchvision
        ```

Please see more details in [Python User Guide](https://bigdl.readthedocs.io/en/latest/doc/UserGuide/python.html).

### 2.2.3 Install Extra Python Libraries
* When you are running a TensorFlow2 model, please install TF2 as below:
```bash
pip install tensorflow==2.8.0
pip install keras==2.8.0
```
* Please install TF1 when you are running a Tensorflow1 model.
```
pip install tensorflow==1.15.0
pip install keras==2.3.1
```

## 2.3 For Spark Installed Cluster(CDH Users)
If your cluster has already installed Spark, the existing Spark might be conflict with the pyspark installed by pip required by BigDL. Thus before running BigDL applications, you should refer to the following methods to avoid the conflict.

### 2.3.1 Install the Same Version Pyspark
On the client node you run the application, please install the same version of `pyspark` with existing Spark environment.
```bash
pip install pyspark==${SPARK_VERSION}
```

### 2.3.2 Unset Spark Environment Variables
On the client node that you run Orca programs, please use `env | grep SPARK` to find all the existing Spark environment variables.
```bash
env | grep SPARK
```
You may see the following variables.
```
SPARK_HOME=/path/to/spark
SPARK_VERSION=`your spark version`
```
Please unset all the Spark related environment variables on the client node to avoid the conflict.
```bash
unset SPARK_HOME
unset SPARK_VERSION
```
For CDH cluster user, the `HADOOP_CONF_DIR` should be `/etc/hadoop/conf` on CDH by default.

### 2.3.3 Using Arrow to Connect Network File Systems(For CDH Users)
If you are using `pyarrow` to upload saving model to HDFS or download dataset from HDFS on CDH clusters, you need to locate `libhdfs.so` and setup `ARROW_LIBHDFS_DIR` environment variable manually.
* Note: the `pyarrow` will automatically locate `libhdfs.so` in default path of `$HADOOP_HOME/lib/native` from when you are not using Cloudera.
```bash
locate libhdfs.so
```
You will get a list locations of `libhdfs.so`, please pick the one used in CDH. In most cases, it may look like:
```bash
/opt/cloudera/parcels/CDH-${CHD_VERSION}/lib64/libhdfs.so
```
Assign the directory path you locate to the `ARROW_LIBHDFS_DIR` variable.
```bash
export ARROW_LIBHDFS_DIR=/opt/cloudera/parcels/CDH-${CHD_VERSION}/lib64/
```
Please refer to _Part4 Run Jobs on Yarn_ to see how to specify `ARROW_LIBHDFS_DIR` when running programs on Yarn through mutiple ways.

# 3. Prepare Dataset & Customer Module
## 3.1 Prepare Dataset  
It's recommand for you to load data from network file systems (e.g. HDFS and S3) when you are running programs on Yarn cluster. In next part, we will show you how to upload and use a remote storing dataset to run programs on Yarn.  
### 3.1.1 Prepare Remote Dataset
First you need to download the Fashion-MNIST dataset manually on your local machine.
```bash
# PyTorch official dataset download link
git clone https://github.com/zalandoresearch/fashion-mnist.git

mv /path/to/fashion-mnist/data/fashion /path/to/local/data/FashionMNIST/raw 
```
After downloading the dataset, you need to upload it to a network file system (HDFS or S3).
```bash
hdfs dfs -put /path/to/local/data/FashionMNIST hdfs://url:port/path/to/remote/data
```

### 3.1.2 Use Remote Dataset to Run The Example
The Fashion-MNIST example uses a BigDL providing function `get_remote_file_to_local` to download and process data on each executor. You need to specify `remote_path` to download data from the network file system and `local_path` to store remote datasets when calling `get_remote_file_to_local`.  
```python
from bigdl.orca.data.file import get_remote_file_to_local

get_remote_file_to_local(remote_path="hdfs://path/to/data", local_path="/tmp/data")
```
When running the example, you need to set the dataset path option to the network file system through `--remote-dir` argument. 
```bash
python train.py --cluster_mode yarn-client --remote-dir hdfs://path/to/remote/data
```

## 3.2 Prepare Custom Module & Package
Spark allows to upload Python files(`.py`), and zipped Python packages(`.zip`) to the executors by setting `--py-files` option in Spark scripts or `extra_python_lib` in `init_orca_context`, which is a straightforward method to load and distribute additional custom Python dependency to the cluster. 
### 3.2.1 Depend on Few Python Files
* When you are using `bigdl-submit` or `spark-submit` script to run the example, you need to addtionally upload Python dependency files to cluster through `--py-files` option to distribute between executors.
    ```bash
    --py-files /path/to/model.py
    ```
  You could import the modules directly in the program.
  ```python
  from model import model_creator, optimizer_creator
  ```
* When using `python` command to run the example, please set the `extra_python_lib` in `init_orca_context` to the path of the Python files. Note you should import modules after creating OrcaContext, since the Python files will be uploaded through `init_orca_context`.
    ```python
    from bigdl.orca import init_orca_context

    # Please switch the `cluster_mode` option to `yarn-cluster` if you need to run program on yarn-cluster mode.
    init_orca_context(cluster_mode="yarn-client", cores=4, memory="10g", num_nodes=2, 
                    driver_cores=2, driver_memory="4g",
                    extra_python_lib="/path/to/model.py")

    # Note please import modules after init_orca_context()
    from model import model_creator, optimizer_creator
    ```
### 3.2.2 Depend on a Nested Directory of Python Files
* Create `.zip` Files First
    
    If you depend on a nested directory of python files, we recommend packaging them into a zip file, you need to create a package like `orca_exmaple` with all custom modules. 
    ```bash
    | -- orca_example <dir>
        | -- model.py
    ``` 
    Then zip the whole package.
    ```bash
    zip -q -r orca_example.zip orca_example
    ```
* When running this example using `bigdl-submit` or `spark-submit`, you need to upload the zipped file in Spark scripts through `--py-files` to the cluster.
    ```bash
    --py-files /path/to/orca_example.zip
    ```
    You should import custom modules as below in `train.py`. The zip file will be automatically unzipped after uploading and distributing in the cluster, so you need to load modules from the unzipped file. 
    ```python
    # Import dependency from zip file
    from orca_example.model import model_creator, optimizer_creator
    ```
* When using `python` command to run the example, please set the `extra_python_lib` in `init_orca_context` to the path of zipped Python file. Note you should import modules after creating OrcaContext, since the zip file will be uploaded through `init_orca_context`.

    ```python
    from bigdl.orca import init_orca_context

    # Please switch the `cluster_mode` option to `yarn-cluster` if you need to run program on yarn-cluster mode.
    init_orca_context(cluster_mode="yarn-client", cores=4, memory="10g", num_nodes=2, 
                    driver_cores=2, driver_memory="4g",
                    extra_python_lib="/path/to/orca_example.zip")

    # Note please import modules after init_orca_context()
    from orca_example.model import model_creator, optimizer_creator
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
                  driver_cores=2, driver_memory="4g",
                  extra_python_lib="/path/to/orca_example.zip")
```
__Note__:
* Please refer to __Part3.2__ to learn how to prepare for custom modules and upload them to the cluster through `init_orca_context`.
* If you are running this program on CDH, please set `ARROW_LIBHDFS_DIR` as a Spark configuration through `conf` in `init_orca_context` to upload model saving directory to HDFS successfully, please see more details in __Part2.3.3__.   
    ```python
    # For CDH users
    from bigdl.orca import init_orca_context

    conf={"spark.executorEnv.ARROW_LIBHDFS_DIR": "/opt/cloudera/parcels/CDH-${CDH_VERSION}/lib64"}

    init_orca_context(cluster_mode="yarn-client", cores=4, memory="10g", num_nodes=2,
                    driver_cores=2, driver_memory="4g",
                    extra_python_lib="/path/to/orca_example.zip",
                    conf=conf)
    ```

To execute the tutorial provided Fashion-MINIST example, you should:
* Set `--cluster_mode` argument as `yarn-client` to get or create an OrcaContext.
* Set `--remote_dir` parameter to download the datasets path and save model on remote resources.
    ```bash
    python train.py --cluster_mode yarn-client --remote_dir hdfs://url:port/path/to/remote/data
    ```

### 4.1.2 Yarn Cluster
Fisrt, please call `init_orca_context` to create an OrcaContext at the very beginning of Orca programs. To run programs on yarn cluster mode using built-in function, we should switch the `cluster_mode` to `yarn-cluster` in `init_orca_context` as below:
```python
from bigdl.orca import init_orca_context

init_orca_context(cluster_mode="yarn-cluster", cores=4, memory="10g", num_nodes=2,
                  driver_cores=2, driver_memory="4g",
                  extra_python_lib="/path/to/orca_example.zip")
```
__Note__:
* Please refer to __Part3.2__ to learn how to prepare for custom modules and upload them to the cluster through `init_orca_context`.
* If you are running this program on CDH, please set `ARROW_LIBHDFS_DIR` as a Spark configuration through `conf` in `init_orca_context` to upload model saving directory to HDFS successfully, please see more details in __Part2.3.3__.
    ```python
    # For CDH users
    from bigdl.orca import init_orca_context

    conf={"spark.executorEnv.ARROW_LIBHDFS_DIR": "/opt/cloudera/parcels/CDH-${CDH_VERSION}/lib64"}

    init_orca_context(cluster_mode="yarn-cluster", cores=4, memory="10g", num_nodes=2, 
                    driver_cores=2, driver_memory="4g",
                    extra_python_lib="/path/to/orca_example.zip",
                    conf=conf)
    ```
When running the program provided by tutorial, we should:
* Set `--cluster_mode` parameter to `yarn-cluster`, which could create an OrcaContext and run the Fasion-MNIST program on yarn-cluster mode.
* Set `--remote_dir` parameter to download the datasets path and save model on remote resources.

Now, please run the example as the following script:
```bash
python train.py --cluster_mode yarn-cluster --remote_dir hdfs://url:port/path/to/remote/data
```

### 4.1.3 Jupyter Notebook
You can simply run the program in a Jupyter Notebook. Please note that jupyter cannot run on yarn-cluster, as the driver is not running on the local node.
```bash
# Start a jupyter notebook
jupyter notebook --notebook-dir=/path/to/notebook/directory --ip=* --no-browser
```
Then you need to move the example to notebook. Please call `init_orca_context` at the very beginning of the program and set the `cluster_mode` to `yarn-client`.
```python
from bigdl.orca import init_orca_context

init_orca_context(cluster_mode="yarn-client", cores=4, memory="10g", num_nodes=2, 
                  driver_cores=2, driver_memory="4g",
                  extra_python_lib="/path/to/orca_example.zip")
```

## 4.2 Use `bigdl-submit`
BigDL provides an easy-to-use Spark-like `bigdl-submit` script in `${BIGDL_HOME}/scripts`, which could automatically detect configuration and jars files from the current activate Conda environment and setup them as `${BIGDL_CONF}` and `${BIGDL_JARS}` seperately.

Before running the Fasion-MNIST program on Yarn with `bigdl-submit`, you need to prepare following stuff on the client node where you run the program:
* First, call `init_orca_context` at the very beginning of the program and specify the `cluster_mode` to `bigdl-submit` to create or get an OrcaContext.
    ```python
    from bigdl.orca import init_orca_context

    init_orca_context(cluster_mode="bigdl-submit")
    ```
* Pack the Conda environment to an archive, which captures the Conda environment for Python and stores both Python interpreter and all its relevant dependencies.
    ```bash
    conda pack -o environment.tar.gz#environment 
    ```
Note:
* If `environment.tar.gz` is not under the same directory with `train.py`, we should modify its path in `--archives` in the running command.
    ```
    --archives /path/to/environment.tar.gz#environment
    ```

### 4.2.1 Yarn Client
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
    train.py --cluster_mode bigdl-submit --remote_dir hdfs://url:port/path/to/remote/data
```
In the `bigdl-submit` script for running Orca programs on yarn-client mode:
* `--master`: the spark master, set it to yarn when running programs on Yarn;
* `--deploy-mode`: submit and execute programs on yarn-client or yarn-cluster, set it to client when running programs on yarn-client mode;
* `--archives`: set this option to the path of the Conda archive, the archive will be uploaded to remote clusters(i.e. HDFS) and distributed between executors;
* `--py-files`: upload Python dependency file(s) to cluster and distribute between executors;
* `--conf spark.pyspark.driver.python`: set the driver Python environment as the local Python path. For `yarn-client` mode, the Spark driver is running on local and it will use the Python interpreter in the current active conda environment;
* `--conf spark.pyspark.python`: set the executor Python environment to the Python path in the Conda archive, since executors will use the Python interpreter in the conda archive.
* `--cluster_mode`: set the cluster_mode in `init_orca_context` to get or create an OrcaContext.
* `--remote_dir`: loda data and save model on network file systems. For Yarn cluster, we recommend that you should use the datasets stored in network file systems like HDFS or S3 in a production environment.

__Note:__: 
* Please refer to __Part3.2__ to learn how to prepare for custom modules and upload them to the cluster through `bigdl-submit` script.
* If you are running this program on CDH, you need speficy `ARROW_LIBHDFS_DIR` through `--conf spark.executorEnv.ARROW_LIBHDFS_DIR` in the `bigdl-submit` script to make the `libhdfs.so` library to be recognized in the cluster. Please see more details in __Part2.3.3__.
    ```bash
    bigdl-submit \
        --master yarn \
        --deploy-mode client \
        ...
        --conf spark.executorEnv.ARROW_LIBHDFS_DIR=/opt/cloudera/parcels/CDH-${CDH_VERSION}/lib64 \
        train.py --cluster_mode bigdl-submit --remote_dir hdfs://url:port/path/to/remote/data
    ```

### 4.2.2 Yarn Cluster
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
    train.py --cluster_mode bigdl-submit --remote_dir hdfs://url:port/path/to/remote/data
```
In the `bigdl-submit` script for running Orca programs on yarn-cluster mode:
* `--master`: the spark master, set it to yarn when running programs on Yarn;
* `--deploy-mode`: submit and execute programs on yarn-client or yarn-cluster, set it to cluster when running programs on yarn-cluster mode;
* `--archives`: set the option to the path of the Conda archive, which will be upload to remote resources(i.e. HDFS) and distributed between executors;
* `--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON`: set the Python environment for the Application Master process launched on YARN, it will use the Python interpreter in `environment.tar.gz`;
* `--conf spark.executorEnv.PYSPARK_PYTHON`: set the Python environment for executors on Yarn, it will use the Python interpreter in `environment.tar.gz`;
* `--py-files`: upload Python dependency file(s) to cluster and distribute between executors;
* `--cluster_mode`: set the cluster_mode in `init_orca_context` to get or create an OrcaContext;
* `--remote_dir`: loda data and save model on network file systems. For Yarn cluster, we recommend that you should use the datasets stored in network file systems like HDFS or S3 in a production environment.

__Note:__
* Please refer to __Part3.2__ to learn how to prepare for custom modules and upload them to the cluster through `bigdl-submit` script.
* If you are running this program on CDH, you need speficy `ARROW_LIBHDFS_DIR` through `--conf spark.executorEnv.ARROW_LIBHDFS_DIR` in the `bigdl-submit` script to make the `libhdfs.so` library to be recognized in the cluster. Please see more details in __Part2.3.3__.
    ```bash
    bigdl-submit \
        --master yarn \
        --deploy-mode cluster \
        ...
        --conf spark.executorEnv.ARROW_LIBHDFS_DIR=/opt/cloudera/parcels/CDH-${CDH_VERSION}/lib64 \
        train.py --cluster_mode bigdl-submit --remote_dir hdfs://url:port/path/to/remote/data
    ```

## 4.3 Use `spark-submit`
When the client node where users submit the job on Yarn is not able to install BigDL and other dependency libraries using conda, it's recommended for users to use `spark-submit` script instead of `bigdl-submit`. Before running with `spark-submit`, we need:
* On the __client node__ where you submit and run the program:
    * We need to call `init_orca_context` at the very beginning of the program and specify the `cluster_mode` to `spark-submit` to create or get an OrcaContext.
        ```python
        from bigdl.orca import init_orca_context

        init_orca_context(cluster_mode="spark-submit")
        ```
    * Setup spark environment first since we cannot use pyspark on the client node, please set environment variables `${SPARK_HOME}` as follows:
        ```bash
        export SPARK_HOME=/path/to/spark # the folder path where you extract the Spark package
        ```
    * Set environment variables `${SPARK_VERSION}` and `${BIGDL_VERSION}` as the version in your environment.
    * Download and unzip a BigDL assembly package from [BigDL Release Page](https://bigdl.readthedocs.io/en/latest/doc/release.html), set the unzipped BigDL file path as `${BIGDL_HOME}`, please make sure the BigDL version could respond to the Spark version.
* On a __different node__ that could use conda:
    * Install all the dependency files that BigDL required (please refer to __Part2.2 & Part2.3__);
    * Pack the Conda environment to an archive, which captures the Conda environment for Python and stores both Python interpreter and all its relevant dependencies;
        ```bash
        conda pack -o environment.tar.gz#environment
        ```
    * Send the Conda archive to the client node;
        ```bash
        scp /path/to/environment.tar.gz username@client_ip:/path/to/
        ```

### 4.3.1 Yarn Client
Please submit and execute the program with `spark-submit` script on yarn-client mode as below:
```bash
${SPARK_HOME}/bin/spark-submit \
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
    --archives /path/to/environment.tar.gz#environment \
    train.py --cluster_mode spark-submit --remote_dir hdfs://url:port/path/to/remote/data
```
In the `spark-submit` script for running program on yarn-client:
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
* `--remote_dir`: loda data and save model on network file systems. For Yarn cluster, we recommend that you should use the datasets stored in network file systems like HDFS or S3 in a production environment.

__Note:__
* Please refer to __Part3.2__ to learn how to prepare for custom modules and upload them to the cluster through `spark-submit` script.
* If you are running this program on CDH, you need to speficy `ARROW_LIBHDFS_DIR` through `--conf spark.executorEnv.ARROW_LIBHDFS_DIR` in the `spark-submit` script to make the `libhdfs.so` library to be recognized in the cluster. Please see more details in __Part2.3.3__
    ```bash
    spark-submit \
        --master yarn \
        --deploy-mode cluster \
        ...
        --conf spark.executorEnv.ARROW_LIBHDFS_DIR=/opt/cloudera/parcels/CDH-${CDH_VERSION}/lib64 \
        train.py --cluster_mode spark-submit --remote_dir hdfs://url:port/path/to/remote/data
    ```

### 4.3.2 Yarn-Cluster
This tutorial is mainly based on a BigDL Orca example, please download jars files from [BigDL Dllib Jars](https://repo1.maven.org/maven2/com/intel/analytics/bigdl/bigdl-dllib-spark_2.4.6/2.0.0/bigdl-dllib-spark_2.4.6-2.0.0-jar-with-dependencies.jar) and [BigDL Orca Jars](https://repo1.maven.org/maven2/com/intel/analytics/bigdl/bigdl-orca-spark_2.4.6/2.0.0/bigdl-orca-spark_2.4.6-2.0.0-jar-with-dependencies.jar) separately and distribute them by `--jars` option in the `spark-submit` script when running the program.

Please run the example on yarn-cluster mode with the following script:
```bash
${SPARK_HOME}/bin/spark-submit \
    --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=environment/bin/python \
    --conf spark.executorEnv.PYSPARK_PYTHON=environment/bin/python \
    --archives /path/to/environment.tar.gz#environment \
    --py-files ${BIGDL_HOME}/python/bigdl-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,orca_example.zip \
    --jars ${BIGDL_HOME}/bigdl-dllib-spark_${SPAKR_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar,${BIGDL_HOME}/bigdl-orca-spark_${SPAKR_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar \
    --master yarn \
    --deploy-mode cluster \
    --executor-memory 10g \
    --driver-memory 10g \
    --executor-cores 4 \
    --num-executors 2 \
    train.py --cluster_mode spark-submit --remote_dir hdfs://url:port/path/to/remote/data
```
In the `spark-submit` script for running program on yarn-cluster:
* `--master`: the spark master, set it to yarn when running programs on Yarn;
* `--deploy-mode`: submit and execute programs on yarn-client or yarn-cluster, set it to cluster when running programs on yarn-cluster mode;
* `--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON`: set the Python environment for the Application Master process launched on YARN, it will use the Python interpreter in `environment.tar.gz`; 
* `--conf spark.executorEnv.PYSPARK_PYTHON`: set the Python environment for executors on Yarn, it will use the Python interpreter in `environment.tar.gz`;
* `--archives`: set the option to the path of the Conda archive, which will be upload to remote clusters(i.e. HDFS) and distributed between all executors;
* `--py-files`: upload Python dependency file(s) to cluster and distribute between executors;
* `--jars`: register and distribute BigDL dependency jars files to the Spark job;
* `--cluster_mode`: set the cluster_mode in `init_orca_context` to get or create an OrcaContext;
* `--remote_dir`: loda data and save model on network file systems. For Yarn cluster, we recommend that you should use the datasets stored in network file systems like HDFS or S3 in a production environment.

__Note:__
* Please refer to __Part3.2__ to learn how to prepare for custom modules and upload them to the cluster through `spark-submit` script.
* If you are running this program on CDH, you need speficy `ARROW_LIBHDFS_DIR` through `--conf spark.executorEnv.ARROW_LIBHDFS_DIR` in the `spark-submit` script to make the `libhdfs.so` library to be recognized in the cluster. Please see more details in __Part2.3.3__
    ```bash
    spark-submit \
        --master yarn \
        --deploy-mode cluster \
        ...
        --conf spark.executorEnv.ARROW_LIBHDFS_DIR=/opt/cloudera/parcels/CDH-${CDH_VERSION}/lib64 \
        train.py --cluster_mode spark-submit --remote_dir hdfs://url:port/path/to/remote/data
    ```
