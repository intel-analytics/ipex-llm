For Python users, Analytics Zoo can be installed either [from pip](#install-from-pip-for-local-usage) or [without pip](#install-without-pip).

**NOTE**: We have tested on __Python 3.6__ and __Python 3.7__. Support for Python 2.7 has been removed due to its end of life.

---
## **Install from pip for local usage**
You can use the following command to install the latest release version of __analytics-zoo__ via pip easily:

```bash
pip install analytics-zoo
```

* You are strongly recommended to use Python 3.6 or 3.7. You might need to run `pip3 install analytics-zoo` instead.
* You might need to add `sudo` if you don't have the permission for installation.

**Important:**

1. Installing analytics-zoo from pip will automatically install `pyspark`. To avoid possible conflicts, you are highly recommended to __unset `SPARK_HOME`__ if it exists in your environment.

2. Please always first call `init_nncontext()` at the very beginning of your code after pip install. This will create a SparkContext with optimized performance configuration and initialize the BigDL engine.
```python
from zoo.common.nncontext import *
sc = init_nncontext()
```

**Remarks:**

1. We've tested this package with pip 9.0.1. `pip install --upgrade pip` if necessary.
2. Pip install supports __Mac__ and __Linux__ platforms.
3. You need to install Java __>= JDK8__ before running Analytics Zoo, which is required by `pyspark`.
4. `bigdl==0.10.0`, `pyspark==2.4.3`, `conda-pack==0.3.1` and their dependencies will be automatically installed if they haven't been detected in the current Python environment.

---
## **Install from pip for Yarn cluster**

You only need to following these steps on your driver node and we only support __yarn-client mode__ for now.

1) Install [Conda](https://docs.conda.io/projects/conda/en/latest/commands/install.html) in your environment.

2) Create a new conda environment (with name "zoo" for example):
```
conda create -n zoo python=3.6
source activate zoo
```

3) Install Analytics-Zoo into the created conda-env.

```
pip install analytics-zoo
```

4) Download JDK8 and set the environment variable: JAVA_HOME (recommended).

You can also install JDK via conda without setting the JAVA_HOME manually:

`conda install -c anaconda openjdk=8.0.152`

5) Start `python` and then execute the following code to create a SparkContext on Yarn for verification:

``` python
from zoo import init_spark_on_yarn

sc = init_spark_on_yarn(
    hadoop_conf="path to the yarn configuration folder",
    conda_name="zoo", # The name of the created conda-env
    num_executor=2,
    executor_cores=4,
    executor_memory="8g",
    driver_memory="2g",
    driver_cores=4,
    extra_executor_memory_for_ray="10g")
```

---
## **Install without pip**

If you choose to install Analytics Zoo without pip, you need to prepare Spark and install necessary Python dependencies.

**Steps:**

1. [Download Spark](https://spark.apache.org/downloads.html)

    - Note that __Python 3.6__ is only compatible with Spark 1.6.4, 2.0.3, 2.1.1 and >=2.2.0. See [this issue](https://issues.apache.org/jira/browse/SPARK-19019) for more discussion.


2. You are recommended to download Analytics Zoo prebuilt release package from the [Release Page](../release-download/) and extract it.
Alternatively, you can also build the Analytics Zoo from [source](../ScalaUserGuide/install/#build-with-script-recommended).

3. Install Python dependencies. Analytics Zoo only depends on `numpy` and `six` for now.

#### ***For Spark standalone cluster***

* __Remark__: If you're running in cluster mode, you need to install Python dependencies on both client and each worker node.
* Install numpy: 
```sudo apt-get install python-numpy``` (Ubuntu)
* Install six: 
```sudo apt-get install python-six``` (Ubuntu)

#### ***For Yarn cluster***

You can run Analytics Zoo Python programs on Yarn clusters without changes to the cluster (i.e., no need to pre-install any Python dependency).

You can first package all the required dependencies into a conda environment on the local node (where you will run the spark-submit command),
and then directly use spark-submit to run the Analytics Zoo Python program on the Yarn cluster using that conda environment.

Follow the steps below to create the conda environment:
   
1) Install [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) in your environment.

2) Create a new conda environment (with name "environment" for example):
```
conda create -n environment python=3.6 pip=20.0.2 setuptools=45.2.0 conda-pack==0.3.1
conda activate environment
```

3) Install python dependencies into the created conda environment.
Install the dependencies according to the dependencies listed in `requirements.txt`. You can add your own dependencies into this file if you wish. The current requirements only contain those needed for running Analytics Zoo Python examples and models.
```
pip install -r ${ANALYTICS_ZOO_HOME}/bin/requirements.txt
```

4) Create conda package for the created conda environment

```
conda pack -o environment.tar.gz
```
Then you can find the conda package file under the current directory.
