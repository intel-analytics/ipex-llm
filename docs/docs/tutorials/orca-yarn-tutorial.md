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
* `num_nodes`: an Integer that specifies the number of Spark executors (default to be `1`).
* `driver_cores`: an Integer that specifies the number of cores for the Spark driver (default to be `4`).
* `driver_memory`: a String that specifies the memory for the Spark driver (default to be `"1g"`).
* `extra_python_lib`: a String that specifies the path to extra Python package, one of `.py`, `.zip` or `.egg` files (default to be `None`).
* `conf`: a Key-Value format to append extra conf for Spark (default to be `None`).

__Note__: 
* All arguments __expect__ `cluster_mode` will be ignored when using `bigdl-submit` or `spark-submit` to submit and run Orca programs.

After the Orca programs finished, you should call `stop_orca_context` at the end of the program to release resources and shutdown the underlying Spark execution engine.
```python
from bigdl.orca import stop_orca_context

stop_orca_context()
```

For more details, please see [OrcaContext](https://bigdl.readthedocs.io/en/latest/doc/Orca/Overview/orca-context.html).

## 1.2 Yarn-Client & Yarn-Cluster
The difference between yarn-client and yarn-cluster is where you run your Spark driver. 

For yarn-client, the Spark driver runs in the client process, and the application master is only used for requesting resources from YARN, while for yarn-cluster the Spark driver runs inside an application master process which is managed by YARN on the cluster.

For more details, please see [Launching Spark on YARN](https://spark.apache.org/docs/latest/running-on-yarn.html#launching-spark-on-yarn).

## 1.3 Use Remote Dataset to Run the Example
__Note:__
* When you are running programs on YARN, you should load data from network file systems (e.g. HDFS and S3).

The Fashion-MNIST example uses a BigDL providing function `get_remote_file_to_local` to download datasets and create PyTorch Dataloader on each executor. 

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

When using `get_remote_file_to_local`, you should:

1. Specify `remote_path` to download dataset from the network file system.
2. Specify `local_path` to store downloading datasets on each executor.

# 2. Prepare Environment
Before running the BigDL program on Yarn, we need to setup the environment following the steps below:
* Setup JAVA & Hadoop Environment
* Install Python Libraries 
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

## 2.2 Install Python Libraries
### 2.2.1 Install Conda
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
* Installing bigdl-spark3 will automatically install `pyspark==3.1.2`.

Please see more details in [Python User Guide](https://bigdl.readthedocs.io/en/latest/doc/UserGuide/python.html).

## 2.3 For CDH Users
If the __client node__ where you submit applications has installed Spark, it might conflict with the `pyspark` installed automatically with BigDL on the __client node__ if they versions not match. Please follow the steps below to solve the conflict.

On the __client node__ that you submit applications, please unset Spark related environment variables to avoid the conflict.
```bash
unset SPARK_HOME
unset SPARK_VERSION
```
You could also use `env | grep SPARK` to find all the existing Spark environment variables, please usnet them all.

__Note:__
* For CDH cluster user, the `HADOOP_CONF_DIR` should be `/etc/hadoop/conf` on CDH by default.
* If you are using `pyarrow` to connnect with network file systems, please setup `ARROW_LIBHDFS_DIR` on the __client node__ where you submit applications.
    ```bash
    locate libhdfs.so
    ```
    You will get a list locations of `libhdfs.so`, please assign the directory path used in CDH to the `ARROW_LIBHDFS_DIR` variable.
    ```bash
    export ARROW_LIBHDFS_DIR=/opt/cloudera/parcels/CDH-${CHD_VERSION}/lib64/
    ```
    Please see __Part5__ to see more details for running the example.

# 3. Prepare Dataset 
To run the example on YARN, you should upload the Fashion-MNIST dataset to network file systems (HDFS or S3).   

First, please download the Fashion-MNIST dataset manually on your __client node__.
```bash
# PyTorch official dataset download link
git clone https://github.com/zalandoresearch/fashion-mnist.git

mv /path/to/fashion-mnist/data/fashion /path/to/local/data/FashionMNIST/raw 
```
Then upload it to the network file system.
```bash
# Upload to HDFS
hdfs dfs -put /path/to/local/data/FashionMNIST hdfs://url:port/path/to/remote/data
```

# 4. Prepare Custom Module

Spark allows to upload Python files(`.py`), and zipped Python packages(`.zip`) to the executors by setting `--py-files` option in Spark scripts or `extra_python_lib` in `init_orca_context`. 

__Note:__
* If you depend on few Python files, you could upload these files to cluster through `--py-files` or `extra_python_lib` directly.

If you depend on a nested directory of python files, you should package them into a zip file, you need to create a package like `FashionMNIST` with all custom modules on the __client node__. 
```bash
| -- FashionMNIST <dir>
    | -- model.py
``` 
Then zip the whole package.
```bash
zip -q -r orca_example.zip FashionMNIST
```

* Use `bigdl-submit` or `spark-submit` script
    
    When running this example using `bigdl-submit` or `spark-submit`, you need to upload the zipped file in Spark scripts through `--py-files` to the cluster.
    ```bash
    --py-files /path/to/orca_example.zip
    ```
    You should import custom modules as below in `train.py`.
    ```python
    # Import dependency from zip file
    from FashionMNIST.model import model_creator, optimizer_creator
    ```

* Use `python` command 

    When using `python` command to run the example, please set the `extra_python_lib` in `init_orca_context` to the path of zipped Python file.

    ```python
    from bigdl.orca import init_orca_context

    init_orca_context(cluster_mode, cores, memory, num_nodes, driver_cores, driver_memory,
                      extra_python_lib="/path/to/orca_example.zip")

    # Note please import modules after init_orca_context()
    from FashionMNIST.model import model_creator, optimizer_creator
    ```
    __Note:__ 
    * You should import modules after creating OrcaContext.

# 5. Run Jobs on Yarn
In the following part, we will show you three ways that BigDL supports to submit and run our programs on Yarn:
* Use `python` command
* Use `bigdl-submit`
* Use `spark-submit`

## 5.1 Use `python` Command
This is the easiest and most recommended way to run BigDL on YARN.

__Note:__
* You only need to prepare the environment on the driver machine, all dependencies would be automatically packaged and distributed to the whole Yarn cluster.

### 5.1.1 Yarn Client
Please call `init_orca_context` to create an OrcaContext at the very beginning of each Orca program.
```python
from bigdl.orca import init_orca_context

init_orca_context(cluster_mode="yarn-client", cores=4, memory="10g", num_nodes=2, 
                  driver_cores=2, driver_memory="4g",
                  extra_python_lib="/path/to/orca_example.zip")
``` 
Run the example following command below:
```bash
python train.py --cluster_mode yarn-client --remote_dir hdfs://url:port/path/to/remote/data
```
* `--cluster_mode`: set the cluster_mode in `init_orca_context` to get or create an OrcaContext.
* `--remote_dir`: load dataset and save model directory on network file systems.

__Note__:
* Please refer to __Part4__ to learn how to prepare for extra custom modules and upload them to the cluster through `init_orca_context`.
* For CDH users, please set `spark.executorEnv.ARROW_LIBHDFS_DIR` as a Spark configuration through `conf` in `init_orca_context`. Please see __Part2.3__ for more details.

### 5.1.2 Yarn Cluster
Please call `init_orca_context` to create an OrcaContext at the very beginning of each Orca program.
```python
from bigdl.orca import init_orca_context

init_orca_context(cluster_mode="yarn-cluster", cores=4, memory="10g", num_nodes=2,
                  driver_cores=2, driver_memory="4g",
                  extra_python_lib="/path/to/orca_example.zip")
```
Run the example following command below:
```bash
python train.py --cluster_mode yarn-cluster --remote_dir hdfs://url:port/path/to/remote/data
```
* `--cluster_mode`: set the cluster_mode in `init_orca_context` to get or create an OrcaContext.
* `--remote_dir`: load dataset and save model directory on network file systems.

__Note__:
* Please refer to __Part4__ to learn how to prepare for extra custom modules and upload them to the cluster through `init_orca_context`.
* For CDH users, please set `spark.executorEnv.ARROW_LIBHDFS_DIR` as a Spark configuration through `conf` in `init_orca_context`. Please see __Part2.3__ for more details.

### 5.1.3 Jupyter Notebook
You can simply run the program in a Jupyter Notebook. 

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
__Note:__
* Jupyter Notebook cannot run on `yarn-cluster`, as the driver is not running on the client node.

## 5.2 Use `bigdl-submit`
BigDL provides an easy-to-use Spark-like `bigdl-submit` script in `${BIGDL_HOME}/scripts`, which could automatically detect configuration and jars files from the current activate Conda environment.

On the __client node__ where you submit the example, before submitting the example:
1. Please call `init_orca_context` at the very beginning of the program.
    ```python
    from bigdl.orca import init_orca_context

    init_orca_context(cluster_mode="bigdl-submit")
    ```
2. Pack the Conda environment to an archive.
    ```bash
    conda pack -o environment.tar.gz#environment 
    ```

### 5.2.1 Yarn Client
Submit and run the BigDL program on `yarn-client` mode following `bigdl-submit` script below:
```bash
bigdl-submit \
    --master yarn \
    --deploy-mode client \
    --executor-memory 10g \
    --driver-memory 10g \
    --executor-cores 8 \
    --num-executors 2 \
    --py-files /path/to/orca_example.zip \
    --archives /path/to/environment.tar.gz#environment \
    --conf spark.pyspark.driver.python=/path/to/python \
    --conf spark.pyspark.python=environment/bin/python \
    train.py --cluster_mode bigdl-submit --remote_dir hdfs://url:port/path/to/remote/data
```
In the `bigdl-submit` script for running Orca programs on yarn-client mode:
* `--master`: the spark master, set it to yarn when running programs on Yarn;
* `--deploy-mode`: submit and execute programs on yarn-client or yarn-cluster, set it to client when running programs on yarn-client mode;
* `--archives`: set this option to the path of the Conda archive, the archive will be uploaded to remote clusters(i.e. HDFS) and distributed between executors;
* `--py-files`: upload Python dependency file(s) to cluster and distribute between executors;
* `--conf spark.pyspark.driver.python`: set the driver Python environment as the local Python path.;For `yarn-client` mode, the Spark driver is running on local and it will use the Python interpreter in the current active conda environment;
* `--conf spark.pyspark.python`: set the executor Python environment to the Python path in the Conda archive, since executors will use the Python interpreter in the conda archive;
* `--cluster_mode`: set the cluster_mode in `init_orca_context` to get or create an OrcaContext.
* `--remote_dir`: load dataset and save model directory on network file systems.

__Note:__
* Please refer to __Part4__ to learn how to prepare for extra custom modules and upload them to the cluster through `bigdl-submit` script.
* If you are running the example on CDH, please speficy `--conf spark.executorEnv.ARROW_LIBHDFS_DIR` in the `bigdl-submit` script. Please see __Part2.3__ for more details.

### 5.2.2 Yarn Cluster
Submit and run the program on `yarn-cluster` mode following `bigdl-submit` script below: 
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
    --archives /path/to/environment.tar.gz#environment \
    --py-files /path/to/orca_example.zip \
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
* `--remote_dir`: load dataset and save model directory on network file systems.

__Note:__
* Please refer to __Part4__ to learn how to prepare for extra custom modules and upload them to the cluster through `bigdl-submit` script.
* If you are running the program on CDH, please speficy `--conf spark.executorEnv.ARROW_LIBHDFS_DIR` in the `bigdl-submit` script. Please see __Part2.3__ for more details.

## 5.3 Use `spark-submit`
When the __client node__ where you submit application is not able to install BigDL using conda, please use `spark-submit` script instead of `bigdl-submit`. Before running with `spark-submit`, you need:
* On the __client node__ where you submit application:
    1. Call `init_orca_context` at the very beginning of the program and specify the `cluster_mode` to `spark-submit`.
        ```python
        from bigdl.orca import init_orca_context

        init_orca_context(cluster_mode="spark-submit")
        ```
    2. Setup spark environment variables `${SPARK_HOME}` and `${SPARK_VERSION}`.
        ```bash
        export SPARK_HOME=/path/to/spark # the folder path where you extract the Spark package
        export SPARK_VERSION="your spark version"
        ```
    3. Download and unzip a BigDL assembly package from [BigDL Release Page](https://bigdl.readthedocs.io/en/latest/doc/release.html) which coule match `${SPARK_VERSION}`, then setup `${BIGDL_HOME}` and `${BIGDL_VERSION}`.
        ```bash
        export BIGDL_HOME=path/to/unzipped/BigDL
        export BIGDL_VERSION="download BigDL version"
        ```

* On a __different node__ that could use conda:
    1. Install all dependency files that BigDL required (please refer to __Part2.2__);
    2. Pack the Conda environment to an archive;
        ```bash
        conda pack -o environment.tar.gz#environment
        ```
    3. Send the Conda archive to the __client node__;
        ```bash
        scp /path/to/environment.tar.gz username@client_ip:/path/to/
        ```

### 5.3.1 Yarn Client
Submit and run the program on `yarn-client` mode following `spark-submit` script below: 
```bash
${SPARK_HOME}/bin/spark-submit \
    --properties-file ${BIGDL_HOME}/conf/spark-bigdl.conf \
    --py-files ${BIGDL_HOME}/python/bigdl-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,/path/to/orca_example.zip \
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
* `--conf spark.executor.extraClassPath`: load and register the BigDL jars files to the executors' classpath;
* `--cluster_mode`: set the cluster_mode in `init_orca_context` to get or create an OrcaContext;
* `--remote_dir`: load dataset and save model directory on network file systems.

__Note:__
* Please refer to __Part4__ to learn how to prepare for extra custom modules and upload them to the cluster through `spark-submit` script.
* If you are running this program on CDH, please speficy `--conf spark.executorEnv.ARROW_LIBHDFS_DIR` in the `spark-submit` script. Please see __Part2.3__ for more details.

### 5.3.2 Yarn-Cluster
On the __client node__, please download jars files from [BigDL Dllib Jars](https://repo1.maven.org/maven2/com/intel/analytics/bigdl/bigdl-dllib-spark_2.4.6/2.0.0/bigdl-dllib-spark_2.4.6-2.0.0-jar-with-dependencies.jar) and [BigDL Orca Jars](https://repo1.maven.org/maven2/com/intel/analytics/bigdl/bigdl-orca-spark_2.4.6/2.0.0/bigdl-orca-spark_2.4.6-2.0.0-jar-with-dependencies.jar) separately and distribute them by `--jars` option in the `spark-submit` script when running the program.

Submit and run the program on `yarn-cluster` mode following `spark-submit` script below:
```bash
${SPARK_HOME}/bin/spark-submit \
    --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=environment/bin/python \
    --conf spark.executorEnv.PYSPARK_PYTHON=environment/bin/python \
    --archives /path/to/environment.tar.gz#environment \
    --py-files ${BIGDL_HOME}/python/bigdl-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,/path/to/orca_example.zip \
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
* `--remote_dir`: load dataset and save model directory on network file systems.

__Note:__
* Please refer to __Part4__ to learn how to prepare for extra custom modules and upload them to the cluster through `spark-submit` script.
* If you are running this program on CDH, please speficy `--conf spark.executorEnv.ARROW_LIBHDFS_DIR` in the `spark-submit` script. Please see __Part2.3__ for more details.
