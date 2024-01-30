# Installation

---
## Prepare the environment
You can follow the commands in this section to install Java and conda before installing BigDL Orca.

### Install Java
You need to download and install JDK in the environment, and properly set the environment variable `JAVA_HOME`. JDK8 is highly recommended.

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

### Install Anaconda
We recommend using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to prepare the Python environment.

You can follow the steps below to install conda:
```bash
# Download Anaconda installation script 
wget -P /tmp https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh

# Execute the script to install conda
bash /tmp/Anaconda3-2020.02-Linux-x86_64.sh

# Run this command in your terminal to activate conda
source ~/.bashrc
``` 

Then create a Python environment for BigDL Orca:
```bash
conda create -n py37 python=3.7  # "py37" is conda environment name, you can use any name you like.
conda activate py37
```

---
## Install BigDL Orca

This section demonstrates how to install BigDL Orca via `pip`, which is the most recommended way.

__Notes:__
* Installing BigDL Orca from pip will automatically install `pyspark`. To avoid possible conflicts, you are highly recommended to  **unset the environment variable `SPARK_HOME`**  if it exists in your environment.

* If you are using a custom URL of Python Package Index to install the latest version, you may need to check whether the latest packages have been sync'ed with pypi. Or you can add the option `-i https://pypi.python.org/simple` when pip install to use pypi as the index-url.


### To use basic Orca features
You can install Orca in your created conda environment for distributed data processing, training and inference with the following command:
```bash
pip install bigdl-orca  # For the official release version
```

or for the nightly build version, use:
```bash
pip install --pre --upgrade bigdl-orca  # For the latest nightly build version
```

Note that installing Orca will automatically install the dependencies including `bigdl-dllib`, `bigdl-tf`, `bigdl-math`, `packaging`, `filelock`, `pyzmq` and their dependencies if they haven't been detected in your conda environment._

### To additionally use RayOnSpark

If you wish to run [RayOnSpark](ray.md) or [sklearn-style Estimator APIs in Orca](distributed-training-inference.md) with **"ray" backend**, use the extra key `[ray]` during the installation above:

```bash
pip install bigdl-orca[ray]  # For the official release version
```

or for the nightly build version, use:
```bash
pip install --pre --upgrade bigdl-orca[ray]  # For the latest nightly build version
```

Note that with the extra key of [ray], `pip` will automatically install the additional dependencies for RayOnSpark,
including `ray[default]==1.9.2`, `aiohttp==3.9.2`, `async-timeout==4.0.1`, `aioredis==1.3.1`, `hiredis==2.0.0`, `prometheus-client==0.11.0`, `psutil`,  `setproctitle`.

### To additionally use AutoML

If you wish to run AutoML, use the extra key `[automl]` during the installation above:

```bash
pip install bigdl-orca[automl]  # For the official release version
````

or for the nightly build version, use:
```bash
pip install --pre --upgrade bigdl-orca[automl]  # For the latest nightly build version
```

Note that with the extra key of [automl], `pip` will automatically install the additional dependencies for distributed hyper-parameter tuning,
including `ray[tune]==1.9.2`, `scikit-learn`, `tensorboard`, `xgboost` together with the dependencies given by the extra key [ray].

- To use [Pytorch AutoEstimator](distributed-tuning.md#pytorch-autoestimator), you need to install Pytorch with `pip install torch==1.8.1`.

- To use [TensorFlow/Keras AutoEstimator](distributed-tuning.md#tensorflow-keras-autoestimator), you need to install TensorFlow with `pip install tensorflow==1.15.0`.

### To install Orca for Spark3

By default, Orca is built on top of Spark 2.4.6 (with pyspark==2.4.6 as a dependency). If you want to install Orca built on top of Spark 3.1.3 (with pyspark==3.1.3 as a dependency), you can use the following command instead:

```bash
# For the official release version
pip install bigdl-orca-spark3
pip install bigdl-orca-spark3[ray]
pip install bigdl-orca-spark3[automl]

# For the latest nightly build version
pip install --pre --upgrade bigdl-orca-spark3
pip install --pre --upgrade bigdl-orca-spark3[ray]
pip install --pre --upgrade bigdl-orca-spark3[automl]
```

__Note__: You should only install Orca built on top of __ONE__ Spark version, but not both. If you want to switch the Spark version, please [**uninstall**](#to-uninstall-orca) Orca cleanly before reinstall.

### To uninstall Orca
```bash
# For default Orca built on top of Spark 2.4.6
pip uninstall bigdl-orca bigdl-dllib bigdl-tf bigdl-math bigdl-core

# For Orca built on top of Spark 3.1.3
pip uninstall bigdl-orca-spark3 bigdl-dllib-spark3 bigdl-tf bigdl-math bigdl-core
```

__Note__: If necessary, you need to manually uninstall `pyspark` and other [dependencies](https://github.com/intel-analytics/BigDL/tree/main/python/requirements/orca) introduced by Orca.

---
## Download BigDL Orca

You can also download the BigDL package via the download links below.

|           | <center>2.2.0</center> | 2.3.0-SNAPSHOT | <center>2.1.0</center> |
| :-: | :-: | :-: | :-: |
| Spark 2.4 | [download](https://repo1.maven.org/maven2/com/intel/analytics/bigdl/bigdl-assembly-spark_2.4.6/2.2.0/bigdl-assembly-spark_2.4.6-2.2.0-fat-jars.zip) | [download](https://oss.sonatype.org/content/repositories/snapshots/com/intel/analytics/bigdl/bigdl-assembly-spark_2.4.6/2.3.0-SNAPSHOT/bigdl-assembly-spark_2.4.6-2.3.0-20230214.114049-32-fat-jars.zip) | [download](https://repo1.maven.org/maven2/com/intel/analytics/bigdl/bigdl-assembly-spark_2.4.6/2.1.0/bigdl-assembly-spark_2.4.6-2.1.0-fat-jars.zip) |
| Spark 3.1 | [download](https://repo1.maven.org/maven2/com/intel/analytics/bigdl/bigdl-assembly-spark_3.1.3/2.2.0/bigdl-assembly-spark_3.1.3-2.2.0-fat-jars.zip) | [download](https://oss.sonatype.org/content/repositories/snapshots/com/intel/analytics/bigdl/bigdl-assembly-spark_3.1.3/2.3.0-SNAPSHOT/bigdl-assembly-spark_3.1.3-2.3.0-20230214.111537-33-fat-jars.zip) | [download](https://repo1.maven.org/maven2/com/intel/analytics/bigdl/bigdl-assembly-spark_3.1.2/2.1.0/bigdl-assembly-spark_3.1.2-2.1.0-fat-jars.zip) |

Note that *SNAPSHOT* indicates the latest nightly build version of BigDL.

If you wish to download the BigDL package in the command line, you can run this [script](https://github.com/intel-analytics/BigDL/blob/main/scripts/download-bigdl.sh) instead.