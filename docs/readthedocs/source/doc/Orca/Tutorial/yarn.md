# Run on Hadoop/YARN Clusters

This tutorial provides a step-by-step guide on how to run BigDL-Orca programs on Apache Hadoop/YARN clusters, using a [PyTorch Fashion-MNIST program](https://github.com/intel-analytics/BigDL/blob/main/python/orca/tutorial/pytorch/FashionMNIST/) as a working example.

The **Client Node** that appears in this tutorial refer to the machine where you launch or submit your applications.

---
## 1. Basic Concepts
### 1.1 init_orca_context
A BigDL Orca program usually starts with the initialization of OrcaContext. For every BigDL Orca program, you should call `init_orca_context` at the beginning of the program as below:

```python
from bigdl.orca import init_orca_context

sc = init_orca_context(cluster_mode, cores, memory, num_nodes,
                       driver_cores, driver_memory, extra_python_lib, conf)
```

In `init_orca_context`, you may specify necessary runtime configurations for running the example on YARN, including:
* `cluster_mode`: one of `"yarn-client"`, `"yarn-cluster"`, `"bigdl-submit"` or `"spark-submit"` when you run on Hadoop/YARN clusters.
* `cores`: the number of cores for each executor (default to be `2`).
* `memory`:  memory for each executor (default to be `"2g"`).
* `num_nodes`: the number of executors (default to be `1`).
* `driver_cores`: the number of cores for the driver node (default to be `4`).
* `driver_memory`: the memory for the driver node (default to be `"2g"`).
* `extra_python_lib`: the path to extra Python packages, separated by comma (default to be `None`). `.py`, `.zip` or `.egg` files are supported.
* `conf`: a dictionary to append extra conf for Spark (default to be `None`).

__Note__: 
* All the arguments __except__ `cluster_mode` will be ignored when using [`bigdl-submit`](#use-bigdl-submit) or [`spark-submit`](#use-spark-submit) to submit and run Orca programs, in which case you are supposed to specify these configurations via the submit command.

After Orca programs finish, you should always call `stop_orca_context` at the end of the program to release resources and shutdown the underlying distributed runtime engine (such as Spark or Ray).
```python
from bigdl.orca import stop_orca_context

stop_orca_context()
```

For more details, please see [OrcaContext](../Overview/orca-context.md).

### 1.2 Yarn-Client & Yarn-Cluster
The difference between yarn-client mode and yarn-cluster mode is where you run your Spark driver. 

For yarn-client, the Spark driver runs in the client process, and the application master is only used for requesting resources from YARN, while for yarn-cluster the Spark driver runs inside an application master process which is managed by YARN in the cluster.

Please see more details in [Launching Spark on YARN](https://spark.apache.org/docs/latest/running-on-yarn.html#launching-spark-on-yarn).

For **yarn-client** mode, you can directly find the driver logs in the console. 

For **yarn-cluster** mode, an `application_time_id` will be returned (`application_1668477395550_1045` in the following log) when the application master process completes.

```bash
23/02/15 15:30:26 INFO yarn.Client: Application report for application_1668477395550_1045 (state: FINISHED)
23/02/15 15:30:26 INFO yarn.Client:
         client token: N/A
         diagnostics: N/A
         ApplicationMaster host: ...
         ApplicationMaster RPC port: 46652
         queue: ...
         start time: 1676446090408
         final status: SUCCEEDED
         tracking URL: http://.../application_1668477395550_1045/
         user: ...
```

Visit the tracking URL and then click `logs` in the table `ApplicationMaster` to see the driver logs.

### 1.3 Distributed storage on YARN
__Note__:
* When you run programs on YARN, you are highly recommended to load/write data from/to a distributed storage (e.g. [HDFS](https://hadoop.apache.org/docs/r1.2.1/hdfs_design.html) or [S3](https://aws.amazon.com/s3/)) instead of the local file system.

The Fashion-MNIST example in this tutorial uses a utility function `get_remote_dir_to_local` provided by BigDL to download datasets and create the PyTorch DataLoader on each executor.

```python
import torch
import torchvision
import torchvision.transforms as transforms
from bigdl.orca.data.file import get_remote_dir_to_local

def train_data_creator(config, batch_size):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    get_remote_dir_to_local(remote_path="hdfs://path/to/dataset", local_path="/tmp/dataset")

    trainset = torchvision.datasets.FashionMNIST(root="/tmp/dataset", train=True,
                                                 download=False, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    return trainloader
```

---
## 2. Prepare Environment
Before running BigDL Orca programs on YARN, you need to properly setup the environment following the steps in this section.

__Note__:
* When using [`python` command](#use-python-command) or [`bigdl-submit`](#use-bigdl-submit), we would directly use the corresponding `pyspark` (which is a dependency of BigDL Orca) for the Spark environment. Thus to avoid possible conflicts, you *DON'T* need to download Spark by yourself or set the environment variable `SPARK_HOME` unless you use [`spark-submit`](#use-spark-submit). 


### 2.1 Setup JAVA & Hadoop Environment
- See [here](../Overview/install.md#install-java) to prepare Java in your cluster.

- Check the Hadoop setup and configurations of your cluster. Make sure you correctly set the environment variable `HADOOP_CONF_DIR`, which is needed to initialize Spark on YARN:
    ```bash
    export HADOOP_CONF_DIR=/path/to/hadoop/conf
    ```

### 2.2 Install Python Libraries
- See [here](../Overview/install.md#install-anaconda) to install conda and prepare the Python environment on the __Client Node__.

- See [here](../Overview/install.md#install-bigdl-orca) to install BigDL Orca in the created conda environment. Note that if you use [`spark-submit`](#use-spark-submit), please __SKIP__ this step and __DO NOT__ install BigDL Orca with pip install command in the conda environment.

- You should install all the other Python libraries that you need in your program in the conda environment as well. `torch`, `torchvision` and `tqdm` are needed to run the Fashion-MNIST example:
    ```bash
    pip install torch torchvision tqdm
    ```


### 2.3 Run on CDH
* For [CDH](https://www.cloudera.com/products/open-source/apache-hadoop/key-cdh-components.html) users, the environment variable `HADOOP_CONF_DIR` should be `/etc/hadoop/conf` by default.

* The __Client Node__ may have already installed a different version of Spark than the one installed with BigDL. To avoid conflicts, unset all Spark-related environment variables (you may use use `env | grep SPARK` to find all of them):
    ```bash
    unset SPARK_HOME
    unset ...
    ```

---
## 3. Prepare Dataset 
To run the Fashion-MNIST example provided by this tutorial on YARN, you should upload the Fashion-MNIST dataset to a distributed storage (such as HDFS or S3) beforehand.   

First, download the Fashion-MNIST dataset manually on your __Client Node__. Note that PyTorch `FashionMNIST Dataset` requires unzipped files located in `FashionMNIST/raw/` under the dataset folder.
```bash
# PyTorch official dataset download link
git clone https://github.com/zalandoresearch/fashion-mnist.git

# Copy the dataset files to the folder FashionMNIST/raw
cp /path/to/fashion-mnist/data/fashion/* /path/to/local/data/FashionMNIST/raw

# Extract FashionMNIST archives
gzip -d /path/to/local/data/FashionMNIST/raw/*
```
Then upload it to a distributed storage. Sample command to upload data to HDFS is as follows:
```bash
hdfs dfs -put /path/to/local/data/FashionMNIST hdfs://path/to/remote/data
```
In the given example, you can specify the argument `--data_dir` to be the directory on a distributed storage for the Fashion-MNIST dataset. The directory should contain `FashionMNIST/raw/train-images-idx3-ubyte` and `FashionMNIST/raw/t10k-images-idx3`.

---
## 4. Prepare Custom Modules
Spark allows to upload Python files (`.py`), and zipped Python packages (`.zip`) across the cluster by setting `--py-files` option in Spark scripts or specifying `extra_python_lib` in `init_orca_context`.

The FasionMNIST example needs to import the modules from [`model.py`](https://github.com/intel-analytics/BigDL/blob/main/python/orca/tutorial/pytorch/FashionMNIST/model.py).
* When using [`python` command](#use-python-command), please specify `extra_python_lib` in `init_orca_context`.
    ```python
    init_orca_context(..., extra_python_lib="model.py")
    ```

For more details, please see [BigDL Python Dependencies](https://bigdl.readthedocs.io/en/latest/doc/Orca/Overview/orca-context.html#python-dependencies).

* When using [`bigdl-submit`](#use-bigdl-submit) or [`spark-submit`](#use-spark-submit), please specify `--py-files` option in the submit command.
    ```bash
    bigdl-submit # or spark-submit
        ...
        --py-files model.py
        ...
    ```

For more details, please see [Spark Python Dependencies](https://spark.apache.org/docs/latest/submitting-applications.html). 

* After uploading `model.py` to YARN, you can import this custom module as follows:
    ```python
    from model import model_creator, optimizer_creator
    ```


If your program depends on a nested directory of Python files, you are recommended to follow the steps below to use a zipped package instead.

1. Compress the directory into a zipped package.
    ```bash
    zip -q -r FashionMNIST_zipped.zip FashionMNIST
    ```
2. Upload the zipped package (`FashionMNIST_zipped.zip`) to YARN by setting `--py-files` or specifying `extra_python_lib` as discussed above.

3. You can then import the custom modules from the unzipped file in your program as follows:
    ```python
    from FashionMNIST.model import model_creator, optimizer_creator
    ```

---
## 5. Run Jobs on YARN
In the remaining part of this tutorial, we will illustrate three ways to submit and run BigDL Orca applications on YARN.

* Use `python` command
* Use `bigdl-submit`
* Use `spark-submit`

You can choose one of them based on your preference or cluster settings.

We provide the running command for the [Fashion-MNIST example](https://github.com/intel-analytics/BigDL/blob/main/python/orca/tutorial/pytorch/FashionMNIST/) on the __Client Node__ in this section.

### 5.1 Use `python` Command
This is the easiest and most recommended way to run BigDL Orca on YARN as a normal Python program. Using this way, you only need to prepare the environment on the __Client Node__ and the environment would be automatically packaged and distributed to the YARN cluster. 

See [here](#init-orca-context) for the runtime configurations.


#### 5.1.1 Yarn Client
Run the example with the following command by setting the cluster_mode to "yarn-client":
```bash
python train.py --cluster_mode yarn-client --data_dir hdfs://path/to/remote/data
```


#### 5.1.2 Yarn Cluster
Run the example with the following command by setting the cluster_mode to "yarn-cluster":
```bash
python train.py --cluster_mode yarn-cluster --data_dir hdfs://path/to/remote/data
```


#### 5.1.3 Jupyter Notebook
You can easily run the example in a Jupyter Notebook using __`yarn-client` mode__. Launch the notebook using the following command:
```bash
jupyter notebook --notebook-dir=/path/to/notebook/directory --ip=* --no-browser
```

You can copy the code in [train.py](https://github.com/intel-analytics/BigDL/blob/main/python/orca/tutorial/pytorch/FashionMNIST/train.py) to the notebook and run the cells. Set the cluster_mode to "yarn-client" in `init_orca_context`.
```python
sc = init_orca_context(cluster_mode="yarn-client", cores=4, memory="2g", num_nodes=2, 
                       driver_cores=2, driver_memory="2g",
                       extra_python_lib="model.py")
```
Note that Jupyter Notebook cannot run on `yarn-cluster` mode, as the driver is not running on the __Client Node__ (where you run the notebook).


### 5.2 Use `bigdl-submit`
For users who want to use a script instead of Python command, BigDL provides an easy-to-use `bigdl-submit` script, which could automatically setup BigDL configuration and jars files from the current activate conda environment.

Set the cluster_mode to "bigdl-submit" in `init_orca_context`.
```python
sc = init_orca_context(cluster_mode="bigdl-submit")
```

Pack the current activate conda environment to an archive on the __Client Node__ before submitting the example:
```bash
conda pack -o environment.tar.gz
```

Some runtime configurations for `bigdl-submit` are as follows:

* `--master`: the spark master, set it to "yarn".
* `--num_executors`: the number of executors.
* `--executor-cores`: the number of cores for each executor.
* `--executor-memory`: the memory for each executor.
* `--driver-cores`: the number of cores for the driver.
* `--driver-memory`: the memory for the driver.
* `--py-files`: the extra Python dependency files to be uploaded to YARN.
* `--archives`: the conda archive to be uploaded to YARN.

#### 5.2.1 Yarn Client
Submit and run the example for `yarn-client` mode following the `bigdl-submit` script below:
```bash
bigdl-submit \
    --master yarn \
    --deploy-mode client \
    --num-executors 2 \
    --executor-cores 4 \
    --executor-memory 2g \
    --driver-cores 2 \
    --driver-memory 2g \
    --py-files model.py \
    --archives /path/to/environment.tar.gz#environment \
    --conf spark.pyspark.driver.python=python \
    --conf spark.pyspark.python=environment/bin/python \
    train.py --cluster_mode bigdl-submit --data_dir hdfs://path/to/remote/data
```
In the `bigdl-submit` script:
* `--deploy-mode`: set it to `client` when running programs on yarn-client mode.
* `--conf spark.pyspark.driver.python`: set the activate Python location on __Client Node__ as the driver's Python environment.
* `--conf spark.pyspark.python`: set the Python location in the conda archive as each executor's Python environment.


#### 5.2.2 Yarn Cluster
Submit and run the program for `yarn-cluster` mode following the `bigdl-submit` script below: 
```bash
bigdl-submit \
    --master yarn \
    --deploy-mode cluster \
    --num-executors 2 \
    --executor-cores 4 \
    --executor-memory 2g \
    --driver-cores 2 \
    --driver-memory 2g \
    --py-files model.py \
    --archives /path/to/environment.tar.gz#environment \
    --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=environment/bin/python \
    --conf spark.executorEnv.PYSPARK_PYTHON=environment/bin/python \
    train.py --cluster_mode bigdl-submit --data_dir hdfs://path/to/remote/data
```
In the `bigdl-submit` script:
* `--deploy-mode`: set it to `cluster` when running programs on yarn-cluster mode.
* `--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON`: set the Python location in the conda archive as the Python environment of the Application Master.
* `--conf spark.executorEnv.PYSPARK_PYTHON`: also set the Python location in the conda archive as each executor's Python environment. The Application Master and the executors will all use the archive for the Python environment.


### 5.3 Use `spark-submit`
If you prefer to use `spark-submit` instead of `bigdl-submit`, please follow the steps below to prepare the environment on the __Client Node__. 

1. Download the requirement file(s) from [here](https://github.com/intel-analytics/BigDL/tree/main/python/requirements/orca) and install the required Python libraries of BigDL Orca according to your needs.
    ```bash
    pip install -r /path/to/requirements.txt
    ```
    Note that you are recommended **NOT** to install BigDL Orca with pip install command in the conda environment if you use spark-submit to avoid possible conflicts.

    If you are using `requirements_ray.txt`, you need to additionally install `ray[default]` with version 1.9.2 in your environment.

2. Pack the current activate conda environment to an archive:
    ```bash
    conda pack -o environment.tar.gz
    ```

3. Download the BigDL assembly package from [here](../Overview/install.html#download-bigdl-orca) and unzip it. Then setup the environment variables `${BIGDL_HOME}` and `${BIGDL_VERSION}`.
    ```bash
    export BIGDL_VERSION="downloaded BigDL version"
    export BIGDL_HOME=/path/to/unzipped_BigDL  # the folder path where you extract the BigDL package
    ```

4. Download and extract [Spark](https://archive.apache.org/dist/spark/). BigDL is currently released for [Spark 2.4](https://archive.apache.org/dist/spark/spark-2.4.6/spark-2.4.6-bin-hadoop2.7.tgz) and [Spark 3.1](https://archive.apache.org/dist/spark/spark-3.1.3/spark-3.1.3-bin-hadoop2.7.tgz). Make sure the version of your downloaded Spark matches the one that your downloaded BigDL is released with. Then setup the environment variables `${SPARK_HOME}` and `${SPARK_VERSION}`.
    ```bash
    export SPARK_VERSION="downloaded Spark version"
    export SPARK_HOME=/path/to/uncompressed_spark  # the folder path where you extract the Spark package
    ```

5. Set the cluster_mode to "spark-submit" in `init_orca_context`:
    ```python
    sc = init_orca_context(cluster_mode="spark-submit")
    ```

Some runtime configurations for `spark-submit` are as follows:

* `--master`: the spark master, set it to "yarn".
* `--num_executors`: the number of executors.
* `--executor-cores`: the number of cores for each executor.
* `--executor-memory`: the memory for each executor.
* `--driver-cores`: the number of cores for the driver.
* `--driver-memory`: the memory for the driver.
* `--py-files`: the extra Python dependency files to be uploaded to YARN.
* `--archives`: the conda archive to be uploaded to YARN.
* `--properties-file`: the BigDL configuration properties to be uploaded to YARN.
* `--jars`: upload and register BigDL jars to YARN.

#### 5.3.1 Yarn Client
Submit and run the program for `yarn-client` mode following the `spark-submit` script below: 
```bash
${SPARK_HOME}/bin/spark-submit \
    --master yarn \
    --deploy-mode client \
    --num-executors 2 \
    --executor-cores 4 \
    --executor-memory 2g \
    --driver-cores 2 \
    --driver-memory 2g \
    --archives /path/to/environment.tar.gz#environment \
    --properties-file ${BIGDL_HOME}/conf/spark-bigdl.conf \
    --conf spark.pyspark.driver.python=python \
    --conf spark.pyspark.python=environment/bin/python \
    --py-files ${BIGDL_HOME}/python/bigdl-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,model.py \
    --jars ${BIGDL_HOME}/jars/bigdl-assembly-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar \
    train.py --cluster_mode spark-submit --data_dir hdfs://path/to/remote/data
```
In the `spark-submit` script:
* `--deploy-mode`: set it to `client` when running programs on yarn-client mode.
* `--conf spark.pyspark.driver.python`: set the activate Python location on __Client Node__ as the driver's Python environment.
* `--conf spark.pyspark.python`: set the Python location in the conda archive as each executor's Python environment.

#### 5.3.2 Yarn Cluster
Submit and run the program for `yarn-cluster` mode following the `spark-submit` script below:
```bash
${SPARK_HOME}/bin/spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --num-executors 2 \
    --executor-cores 4 \
    --executor-memory 2g \
    --driver-cores 2 \
    --driver-memory 2g \
    --archives /path/to/environment.tar.gz#environment \
    --properties-file ${BIGDL_HOME}/conf/spark-bigdl.conf \
    --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=environment/bin/python \
    --conf spark.executorEnv.PYSPARK_PYTHON=environment/bin/python \
    --py-files ${BIGDL_HOME}/python/bigdl-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,model.py \
    --jars ${BIGDL_HOME}/jars/bigdl-assembly-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar \
    train.py --cluster_mode spark-submit --data_dir hdfs://path/to/remote/data
```
In the `spark-submit` script:
* `--deploy-mode`: set it to `cluster` when running programs on yarn-cluster mode.
* `--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON`: set the Python location in the conda archive as the Python environment of the Application Master.
* `--conf spark.executorEnv.PYSPARK_PYTHON`: also set the Python location in the conda archive as each executor's Python environment. The Application Master and the executors will all use the archive for the Python environment.
