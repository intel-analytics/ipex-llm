# Running BigDL-Orca Program on YARN

This tutorial provides a step-by-step guide on how to run BigDL-Orca programs on Apache Hadoop (YARN) clusters, using a [PyTorch Fashin-MNIST program](https://github.com/intel-analytics/BigDL/tree/main/docs/docs/tutorials/tutorial_example/Fashion_MNIST/) as a working example.

# 1. Key Concepts
## 1.1 Init_orca_context
A BigDL Orca program usually starts with the initialization of OrcaContext. For every BigDL Orca program, you should call `init_orca_context` at the beginning of the program as below:

```python
from bigdl.orca import init_orca_context

init_orca_context(cluster_mode, cores, memory, num_nodes, driver_cores, driver_memory, extra_python_lib, conf)
```

In `init_orca_context`, you may specify necessary runtime configurations for running the example on YARN, including:
* `cluster_mode`: a String that specifies the underlying cluster; valid value includes `"local"`, `"yarn-client"`, `"yarn-cluster"`, `"k8s-client"`, `"k8s-cluster"`, `"bigdl-submit"`, `"spark-submit"`, etc.
* `cores`: an Integer that specifies the number of cores for each executor (default to be `2`).
* `memory`: a String that specifies the memory for each executor (default to be `"2g"`).
* `num_nodes`: an Integer that specifies the number of executors (default to be `1`).
* `driver_cores`: an Integer that specifies the number of cores for the driver node (default to be `4`).
* `driver_memory`: a String that specifies the memory for the driver node (default to be `"1g"`).
* `extra_python_lib`: a String that specifies the path to extra Python package, one of `.py`, `.zip` or `.egg` files (default to be `None`).
* `conf`: a Key-Value format to append extra conf for Spark (default to be `None`).

__Note__: 
* All arguments __except__ `cluster_mode` will be ignored when using `bigdl-submit` or `spark-submit` to submit and run Orca programs.

After the Orca programs finished, you should call `stop_orca_context` at the end of the program to release resources and shutdown the underlying distributed runtime engine (such as Spark or Ray).
```python
from bigdl.orca import stop_orca_context

stop_orca_context()
```

For more details, please see [OrcaContext](https://bigdl.readthedocs.io/en/latest/doc/Orca/Overview/orca-context.html).

## 1.2 Yarn-Client & Yarn-Cluster
The difference between yarn-client and yarn-cluster is where you run your Spark driver. 

For yarn-client, the Spark driver runs in the client process, and the application master is only used for requesting resources from YARN, while for yarn-cluster the Spark driver runs inside an application master process which is managed by YARN on the cluster.

For more details, please see [Launching Spark on YARN](https://spark.apache.org/docs/latest/running-on-yarn.html#launching-spark-on-yarn).

## 1.3 Use Distributed Storage When Running on YARN
__Note:__
* When you are running programs on YARN, you should load data from a distributed storage (e.g. HDFS or S3) instead of the local file system.

The Fashion-MNIST example uses a function `get_remote_file_to_local` provided by BigDL to download datasets and create PyTorch Dataloader on each executor. 

```python
import torch
import torchvision
import torchvision.transforms as transforms
from bigdl.orca.data.file import get_remote_file_to_local

def train_data_creator(config, batch_size):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    get_remote_file_to_local(remote_path="hdfs://url:port/path/to/dataset", local_path="/tmp/dataset")

    trainset = torchvision.datasets.FashionMNIST(root="/tmp/dataset", train=True, download=False, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    return trainloader
```

# 2. Prepare Environment
Before running the BigDL program on YARN, you need to setup the environment following the steps below:

## 2.1 Setup JAVA & Hadoop Environment
You need to download and install JDK in the environment, and properly set the environment variable `JAVA_HOME`, which is required by Spark. JDK8 is highly recommended.

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

## 2.2 Install Python Libraries
### 2.2.1 Install Conda
You need first to use conda to prepare the Python environment on the __Client Node__ (where you submit applications). You could download and install Conda following [Conda User Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) or executing the command as below.
```bash
# Download Anaconda installation script 
wget -P /tmp https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh

# Execute the script to install conda
bash /tmp/Anaconda3-2020.02-Linux-x86_64.sh

# Please type this command in your terminal to activate Conda environment
source ~/.bashrc
``` 

### 2.2.2 Use Conda to Install BigDL and Other Python Libraries
Create a conda environment, install BigDL and all the needed Python libraries in the created conda environment:
``` bash
# "env" is conda environment name, you can use any name you like.
# Please change Python version to 3.6 if you need a Python 3.6 environment.
conda create -n env python=3.7 
conda activate env
```
Alternatively, You can install the latest release version of BigDL (built on top of Spark 2.4.6 by default) as follows:
```bash
pip install bigdl
```
You can install the latest nightly build of BigDL as follows:
```bash
pip install --pre --upgrade bigdl
```

__Note:__ 
* Using Conda to install BigDL will automatically install libraries including `conda-pack`, `torch`, `torchmetrics`, `torchvision`, `pandas`, `numpy`, `pyspark==2.4.6`, and etc.
* You can install BigDL built on top of Spark 3.1.2 as follows:
    
    ```bash
    # Install the latest release version
    pip install bigdl-spark3

    # Install the latest nightly build version
    pip install --pre --upgrade bigdl-spark3 
    ```
    Installing bigdl-spark3 will automatically install `pyspark==3.1.2`.
* You also need to install any additional python libraries that your application depends on in this Conda environment.

Please see more details in [Python User Guide](https://bigdl.readthedocs.io/en/latest/doc/UserGuide/python.html).

## 2.3 Notes for CDH Users
* For CDH user, the value of environment variable `HADOOP_CONF_DIR` should be `/etc/hadoop/conf` by default.

* The __Client Node__ (where you submit applications) may have already installed a different version of Spark then the one installed with BigDL. To avoid conflicts, unset all Spark-related environment variables (you may use use `env | grep SPARK` to find all of them):
    ```bash
    unset SPARK_HOME
    unset SPARK_VERSION
    unset ...
    ```

* You also need to properly export the location of `libhdfs.so` if needed. For instance, when using pyarrow, set up the environment variable `ARROW_LIBHDFS_DIR` on the __Client Node__ (where you submit the program to the YARN cluster) as follows:
    ```bash
    export ARROW_LIBHDFS_DIR=/opt/cloudera/parcels/CDH-${CHD_VERSION}/lib64/
    ```

# 3. Prepare Dataset 
To run the example on YARN, you should upload the Fashion-MNIST dataset to a distributed storage (such as HDFS or S3).   

First, please download the Fashion-MNIST dataset manually on your __Client Node__ (where you submit the program to YARN).
```bash
# PyTorch official dataset download link
git clone https://github.com/zalandoresearch/fashion-mnist.git

mv /path/to/fashion-mnist/data/fashion /path/to/local/data/FashionMNIST/raw 
```
Then upload it to a distributed storage.
```bash
# Upload to HDFS
hdfs dfs -put /path/to/local/data/FashionMNIST hdfs://url:port/path/to/remote/data
```

# 4. Prepare Custom Modules
Spark allows to upload Python files (`.py`), and zipped Python packages (`.zip`) to the executors by setting `--py-files` option in Spark scripts or `extra_python_lib` in `init_orca_context`. 

The FasionMNIST example needs to import modules from `model.py`.
* When using `python` command, please specify `extra_python_lib` in `init_orca_context`.
    ```python
    from bigdl.orca import init_orca_context, stop_orca_context
    from model import model_creator, optimizer_creator

    # Please switch the `cluster_mode` to `yarn-cluster` when running on cluster mode.
    init_orca_context(cluster_mode="yarn-client", cores=4, memory="10g", num_nodes=2,
                      driver_cores=2, driver_memory="4g",
                      extra_python_lib="model.py")
    ```

    Please see more details in [Orca Document](https://bigdl.readthedocs.io/en/latest/doc/Orca/Overview/orca-context.html#python-dependencies).

* When using `bigdl-submit` or `spark-submit` script, please specify `--py-files` option in the script.
    ```bash
    --py-files model.py
    ```

    Import custom modules at the beginning of the example:
    ```python
    from bigdl.orca import init_orca_context, stop_orca_context
    from model import model_creator, optimizer_creator

    init_orca_context(cluster_mode="bigdl-submit") # or spark-submit
    ```

    Please see more details in [Spark Document](https://spark.apache.org/docs/latest/submitting-applications.html). 

__Note:__
* You could follow the steps below to use a zipped package instead (recommended if your program depends on a nested directory of Python files) :
    1. Compress the directory into a zipped package.
        ```bash
        zip -q -r FashionMNIST_zipped.zip FashionMNIST
        ```
    2. Please upload the zipped package (`FashionMNIST_zipped.zip`) to YARN.
        * When using `python` command, please specify `extra_python_lib` argument in `init_orca_context`.
      
        * When using `bigdl-submit` or `spark-submit` script, please specify `--py-files` option in the script.
    3. You should import custom modules from the unzipped file as below.
        ```python
        from FashionMNIST.model import model_creator, optimizer_creator
        ```


# 5. Run Jobs on YARN
In the following part, you will learn three ways that BigDL supports to submit and run the example on YARN:
* Use `python` command
* Use `bigdl-submit`
* Use `spark-submit`

## 5.1 Use `python` Command
This is the easiest and most recommended way to run BigDL on YARN.

__Note:__
* You only need to prepare the environment on the __Client Node__ (where you submit applications), all dependencies would be automatically packaged and distributed to YARN cluster.

### 5.1.1 Yarn Client
Please call `init_orca_context` to create an OrcaContext at the very beginning of each Orca program.
```python
from bigdl.orca import init_orca_context

if args.cluster_mode.startswith("yarn"):
    if args.cluster_mode == "yarn-client":
        init_orca_context(cluster_mode="yarn-client", cores=4, memory="10g", num_nodes=2, 
                          driver_cores=2, driver_memory="4g",
                          extra_python_lib="model.py")
    elif args.cluster_mode == "yarn-cluster":
        ...
``` 
Run the example following command below:
```bash
python train.py --cluster_mode yarn-client --remote_dir hdfs://url:port/path/to/remote/data
```
* `--cluster_mode`: set the cluster_mode in `init_orca_context`.
* `--remote_dir`: directory on a distributed storage for the dataset (see __Section 3__) and saving the model.

__Note__:
* Please refer to __Section 4__ for the description of `extra_python_lib`.
* For CDH users, please set `ARROW_LIBHDFS_DIR` (see __Section 2.3__) as a Spark conf in `init_orca_context`.
    ```python
    init_orca_context(cluster_mode, cores, memory, num_nodes, 
                      driver_cores, driver_memory, extra_python_lib, 
                      conf={"spark.executorEnv.ARROW_LIBHDFS_DIR":"/opt/cloudera/parcels/CDH-${CHD_VERSION}/lib64/"})
    ```

### 5.1.2 Yarn Cluster
Please call `init_orca_context` to create an OrcaContext at the very beginning of each Orca program.
```python
from bigdl.orca import init_orca_context

if args.cluster_mode.startswith("yarn"):
    if args.cluster_mode == "yarn-client":
        ...
    elif args.cluster_mode == "yarn-cluster":
        init_orca_context(cluster_mode="yarn-cluster", cores=4, memory="10g", num_nodes=2,
                          driver_cores=2, driver_memory="4g",
                          extra_python_lib="model.py")
```
Run the example following command below:
```bash
python train.py --cluster_mode yarn-cluster --remote_dir hdfs://url:port/path/to/remote/data
```
* `--cluster_mode`: set the cluster_mode in `init_orca_context`.
* `--remote_dir`: directory on a distributed storage for the dataset (see section 3) and saving the model.

__Note__:
* Please refer to __Section 4__ for the description of `extra_python_lib`.
* For CDH users, please set `ARROW_LIBHDFS_DIR` (see __Section 2.3__) as a Spark conf in `init_orca_context`.
    ```python
    init_orca_context(cluster_mode, cores, memory, num_nodes, driver_cores, driver_memory, extra_python_lib, conf={"spark.executorEnv.ARROW_LIBHDFS_DIR":"/opt/cloudera/parcels/CDH-${CHD_VERSION}/lib64/"})
    ```

### 5.1.3 Jupyter Notebook
You can simply run the example in a Jupyter Notebook. 

```bash
# Start a jupyter notebook
jupyter notebook --notebook-dir=/path/to/notebook/directory --ip=* --no-browser
```
You need to migrate the example to notebook. Please call `init_orca_context` at the very beginning of the program and set the `cluster_mode` to `yarn-client`.
```python
from bigdl.orca import init_orca_context

init_orca_context(cluster_mode="yarn-client", cores=4, memory="10g", num_nodes=2, 
                  driver_cores=2, driver_memory="4g",
                  extra_python_lib="model.py")
```
__Note:__
* Jupyter Notebook cannot run on `yarn-cluster`, as the driver is not running on the __Client Node__(the notebook page).

## 5.2 Use `bigdl-submit`
For users who want to use a script instead of Python command, BigDL provides an easy-to-use `bigdl-submit` script (under `${BIGDL_HOME}/scripts`), which could automatically setup configuration and jars files from the current activate Conda environment.

Please call `init_orca_context` at the very beginning of the program.
```python
from bigdl.orca import init_orca_context

if args.cluster_mode == "bigdl-submit":
    init_orca_context(cluster_mode="bigdl-submit")
```

On the __Client Node__ (where you submit applications), before submitting the example:
1. Install and activate Conda environment (see __Section 2.1__).
2. Use Conda to install BigDL and other Python libraries (see __Section 2.2__).
3. Pack the current activate Conda environment to an archive.
    ```bash
    conda pack -o environment.tar.gz
    ```

### 5.2.1 Yarn Client
Submit and run the example on `yarn-client` mode following `bigdl-submit` script below:
```bash
bigdl-submit \
    --master yarn \
    --deploy-mode client \
    --executor-memory 10g \
    --driver-memory 10g \
    --executor-cores 8 \
    --num-executors 2 \
    --py-files model.py \
    --archives /path/to/environment.tar.gz#environment \
    --conf spark.pyspark.driver.python=/path/to/python \
    --conf spark.pyspark.python=environment/bin/python \
    train.py --cluster_mode bigdl-submit --remote_dir hdfs://url:port/path/to/remote/data
```
In the `bigdl-submit` script:
* `--master`: the spark master, set it to yarn;
* `--deploy-mode`: set it to client when running programs on yarn-client mode;
* `--executor-memory`: set the memory for each executor;
* `--driver-memory`: set the memory for the driver node;
* `--executor-cores`: set the cores number for each executor;
* `--num_executors`: set the number of executors;
* `--py-files`: upload extra Python dependency files to YARN;
* `--archives`: upload the Conda archive to YARN;
* `--conf spark.pyspark.driver.python`: set the activate Python location on __Client Node__ as driver's Python environment;
* `--conf spark.pyspark.python`: set the Python location in Conda archive as executors' Python environment;

__Note:__
* `--cluster_mode`: set the cluster_mode in `init_orca_context`.
* `--remote_dir`: directory on a distributed storage for the dataset (see __Section 3__) and saving the model.
* Please refer to __Section 4__ for the description of extra Python dependencies.
* For CDH Users, please speficy `ARROW_LIBHDFS_DIR` (see __Section 2.3__) as a Spark conf in the script.
    ```bash
    bigdl-submit \
    ...
    --conf spark.executorEnv.ARROW_LIBHDFS_DIR=/opt/cloudera/parcels/CDH-${CHD_VERSION}/lib64/ \
    train.py --cluster_mode bigdl-submit --remote_dir hdfs://url:port/path/to/remote/data
    ```

### 5.2.2 Yarn Cluster
Submit and run the program on `yarn-cluster` mode following `bigdl-submit` script below: 
```bash
bigdl-submit \
    --master yarn \
    --deploy-mode cluster \
    --executor-memory 10g \
    --driver-memory 10g \
    --executor-cores 8 \
    --num-executors 2 \
    --py-files model.py \
    --archives /path/to/environment.tar.gz#environment \
    --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=environment/bin/python \
    --conf spark.executorEnv.PYSPARK_PYTHON=environment/bin/python \
    train.py --cluster_mode bigdl-submit --remote_dir hdfs://url:port/path/to/remote/data
```
In the `bigdl-submit` script:
* `--master`: the spark master, set it to `yarn`;
* `--deploy-mode`: set it to `cluster` when running programs on yarn-cluster mode;
* `--executor-memory`: set the memory for each executor;
* `--driver-memory`: set the memory for the driver node;
* `--executor-cores`: set the cores number for each executor;
* `--num_executors`: set the number of executors;
* `--py-files`: upload extra Python dependency files to YARN;
* `--archives`: upload the Conda archive to YARN;
* `--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON`: set the Python location in Conda archive as Python environment of Application Master process;
* `--conf spark.executorEnv.PYSPARK_PYTHON`: set the Python location in Conda archive as Python environment of executors.

__Note:__
* `--cluster_mode`: set the cluster_mode in `init_orca_context`;
* `--remote_dir`: directory on a distributed storage for the dataset (see __Section 3__) and saving the model.
* Please refer to __Section 4__ for the description of extra Python dependencies.
* For CDH Users, please speficy `ARROW_LIBHDFS_DIR` (see __Section 2.3__) as a Spark conf in the script.
    ```bash
    bigdl-submit \
    ...
    --conf spark.executorEnv.ARROW_LIBHDFS_DIR=/opt/cloudera/parcels/CDH-${CHD_VERSION}/lib64/ \
    train.py --cluster_mode bigdl-submit --remote_dir hdfs://url:port/path/to/remote/data
    ```

## 5.3 Use `spark-submit`
When the __Client Node__ (where you submit applications) is not able to install BigDL using Conda, please use `spark-submit` script instead. 

Please call `init_orca_context` at the very beginning of the program.
```python
from bigdl.orca import init_orca_context

# Please set cluster_mode to "spark-submit".
if args.cluster_mode == "spark-submit":
    init_orca_context(cluster_mode="spark-submit")
```

Before submitting application, you need:
* On the __Development Node__ (which could use Conda):
    1. Install and activate Conda environment (see __Section 2.1__).
    2. Use Conda to install BigDL and other Python libraries (see __Section 2.2__).
    3. Pack the current activate Conda environment to an archive;
        ```bash
        conda pack -o environment.tar.gz
        ```
    4. Send the Conda archive to the __Client Node__;
        ```bash
        scp /path/to/environment.tar.gz username@client_ip:/path/to/
        ```
* On the __Client Node__ (where you submit applications):
    1. Setup spark environment variables `${SPARK_HOME}` and `${SPARK_VERSION}`.
        ```bash
        export SPARK_HOME=/path/to/spark # the folder path where you extract the Spark package
        export SPARK_VERSION="your spark version"
        ```
    2. Download and unzip a BigDL assembly package from [BigDL Release Page](https://bigdl.readthedocs.io/en/latest/doc/release.html) (which could match `${SPARK_VERSION}`), then setup `${BIGDL_HOME}` and `${BIGDL_VERSION}`.
        ```bash
        export BIGDL_HOME=path/to/unzipped/BigDL
        export BIGDL_VERSION="download BigDL version"
        ```

### 5.3.1 Yarn Client
Submit and run the program on `yarn-client` mode following `spark-submit` script below: 
```bash
${SPARK_HOME}/bin/spark-submit \
    --master yarn \
    --deploy-mode client \
    --executor-memory 10g \
    --driver-memory 10g \
    --executor-cores 8 \
    --num-executors 2 \
    --archives /path/to/environment.tar.gz#environment \
    --properties-file ${BIGDL_HOME}/conf/spark-bigdl.conf \
    --py-files ${BIGDL_HOME}/python/bigdl-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,model.py \
    --conf spark.pyspark.driver.python=/path/to/python \
    --conf spark.pyspark.python=environment/bin/python \
    --conf spark.driver.extraClassPath=${BIGDL_HOME}/jars/* \
    --conf spark.executor.extraClassPath=${BIGDL_HOME}/jars/* \
    train.py --cluster_mode spark-submit --remote_dir hdfs://url:port/path/to/remote/data
```
In the `spark-submit` script:
* `--master`: the spark master, set it to `yarn`;
* `--deploy-mode`: set it to `client` when running programs on yarn-client mode;
* `--executor-memory`: set the memory for each executor;
* `--driver-memory`: set the memory for the driver node;
* `--executor-cores`: set the cores number for each executor;
* `--num_executors`: set the number of executors;
* `--archives`: upload the Conda archive to YARN;
* `--properties-file`: upload the BigDL configuration properties to YARN;
* `--py-files`: upload extra Python dependency files to YARN;
* `--conf spark.pyspark.driver.python`: set the Python location in Conda archive as driver's Python environment;
* `--conf spark.pyspark.python`: set the Python location in Conda archive as executors' Python environment;
* `--conf spark.driver.extraClassPath`: upload and register the BigDL jars files to the driver's classpath;
* `--conf spark.executor.extraClassPath`: upload and register the BigDL jars files to the executors' classpath;

__Note:__
* `--cluster_mode`: set the cluster_mode in `init_orca_context`;
* `--remote_dir`: directory on a distributed storage for the dataset (see __Section 3__) and saving the model.
* Please refer to __Section 4__ for the description of extra Python dependencies.
* For CDH Users, please speficy `ARROW_LIBHDFS_DIR` (see __Section 2.3__) as a Spark conf in the script.
    ```bash
    ${SPARK_HOME}/bin/spark-submit \
    ...
    --conf spark.executorEnv.ARROW_LIBHDFS_DIR=/opt/cloudera/parcels/CDH-${CHD_VERSION}/lib64/ \
    train.py --cluster_mode spark-submit --remote_dir hdfs://url:port/path/to/remote/data
    ```

### 5.3.2 Yarn-Cluster
On the __Client Node__ (where you submit applications), please download jars files from [BigDL Dllib Jars](https://repo1.maven.org/maven2/com/intel/analytics/bigdl/bigdl-dllib-spark_2.4.6/2.0.0/bigdl-dllib-spark_2.4.6-2.0.0-jar-with-dependencies.jar) and [BigDL Orca Jars](https://repo1.maven.org/maven2/com/intel/analytics/bigdl/bigdl-orca-spark_2.4.6/2.0.0/bigdl-orca-spark_2.4.6-2.0.0-jar-with-dependencies.jar) separately, and put them under `${BIGDL_HOME}`.

__Note:__
* Please register downloaded BigDL jars through `--jars` option in the `spark-submit` script.

Submit and run the program on `yarn-cluster` mode following `spark-submit` script below:
```bash
${SPARK_HOME}/bin/spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --executor-memory 10g \
    --driver-memory 10g \
    --executor-cores 4 \
    --num-executors 2 \
    --archives /path/to/environment.tar.gz#environment \
    --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=environment/bin/python \
    --conf spark.executorEnv.PYSPARK_PYTHON=environment/bin/python \
    --py-files ${BIGDL_HOME}/python/bigdl-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,model.py \
    --jars /path/to/bigdl-dllib-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar,/path/to/bigdl-orca-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar \
    train.py --cluster_mode spark-submit --remote_dir hdfs://url:port/path/to/remote/data
```
In the `spark-submit` script:
* `--master`: the spark master, set it to `yarn`;
* `--deploy-mode`: set it to `cluster` when running programs on yarn-cluster mode;
* `--executor-memory`: set the memory for each executor;
* `--driver-memory`: set the memory for the driver node;
* `--executor-cores`: set the cores number for each executor;
* `--num_executors`: set the number of executors;
* `--archives`: upload the Conda archive to YARN;
* `--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON`: set the Python location in Conda archive as Python environment of Application Master process; 
* `--conf spark.executorEnv.PYSPARK_PYTHON`: set the Python location in Conda archive as  executors' Python environment;
* `--py-files`: upload extra Python dependency files to YARN;
* `--jars`: upload and register BigDL dependency jars files to YARN;

__Note:__
* `--cluster_mode`: set the cluster_mode in `init_orca_context`;
* `--remote_dir`: directory on a distributed storage for the dataset (see __Section 3__) and saving the model.
* Please refer to __Section 4__ for the description of extra Python dependencies.
* For CDH Users, please speficy `ARROW_LIBHDFS_DIR` (see __Section 2.3__) as a Spark conf in the script.
    ```bash
    ${SPARK_HOME}/bin/spark-submit \
    ...
    --conf spark.executorEnv.ARROW_LIBHDFS_DIR=/opt/cloudera/parcels/CDH-${CHD_VERSION}/lib64/ \
    train.py --cluster_mode spark-submit --remote_dir hdfs://url:port/path/to/remote/data
    ```
