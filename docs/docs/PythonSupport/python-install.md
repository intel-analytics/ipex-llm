
## Use Python without pip Install

You can use Python without pip install. Pip install only supports certain Spark versions and platforms now. You can always use Python this way if pip support is missing.   

First of all, you need to obtain the BigDL libs. You can either download pre-built libs or build from source code. Refer to [Use pre-built libs](../UserGuide/install-pre-built.md) and [Build from Source](../UserGuide/install-build-src.md) for details. 

Then you can either use an interactive shell, run a python program on commandline, or use jupyter notebook. Refer to below sections for details. 

* [Use an interactive shell (without pip install)](python-run.md#user-an-interactive-shell-without-pip-install)
* [Run Python program on Commandline (without pip install)](python-run.md#run-python-program-in-command-line-without-pip-install)
* [Use Jupyter notebook (without pip install)](python-run.md#use-jupyter-notebook-without-pip-install)

---

You can run the following commands to install and play with BigDL locally and this package support both linux and mac platform.

## Installation ##
  * [BigDL-0.1.2](#BigDL-0.1.2)
  * [BigDL-0.2.0-snapshot](#BigDL-0.2.0-snapshot)
  
## Usage ##

* Launch with Python REPL
```
>>> from bigdl.util.common import *
>>> init_engine()
>>> import bigdl.version
>>> bigdl.version.__version__
'0.1.1rc0'
>>> from bigdl.nn.layer import *
>>> linear = Linear(2, 3)
creating: createLinear
>>> continue your experiment....
```
* Launch with jupyter:
   * jupyter notebook --notebook-dir=./ --ip=* --no-browser
   * You need to create SparkContext in this way as we start jupyter without pyspark scripts:
```
      from bigdl.util.common import *
      sc = get_spark_context()
```

<a name="BigDL-0.1.2"></a>
### Install BigDL-0.1.2 ###

1. Download Spark1.6.3:  
```wget https://d3kbcqa49mib13.cloudfront.net/spark-1.6.3-bin-hadoop2.6.tgz ```
2. Extract the tar ball and set SPARK_HOME
```
tar -zxvf spark-1.6.3-bin-hadoop2.6.tgz
export SPARK_HOME=path to spark-1.6.3-bin-hadoop2.6.tgz
```
3. Install BigDL release via pip (we tested this on pip 9.0.1)
```
pip install --upgrade pip
pip install BigDL==0.1.2     # for Python 2.7
pip3 install BigDL==0.1.2  # for Python 3.n
```

<a name="BigDL-0.2.0-snapshot"></a>
### Install BigDL-0.2.0-snapshot ###

1. Download Spark1.6.3:  
```wget https://d3kbcqa49mib13.cloudfront.net/spark-1.6.3-bin-hadoop2.6.tgz ```
2. Extract the tar ball and set SPARK_HOME
```
tar -zxvf spark-1.6.3-bin-hadoop2.6.tgz
export SPARK_HOME=path to spark-1.6.3-bin-hadoop2.6
```
3. Install BigDL release via pip (we tested this on pip 9.0.1)
```
pip install --upgrade pip
pip install BigDL==0.2.0.dev3     # for Python 2.7
pip3 install BigDL==0.2.0.dev3  # for Python 3.n
```



