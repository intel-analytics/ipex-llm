You can run the following commands to install and play with BigDL locally and this package support both linux and mac platform.

## Installation ##
  * [BigDL-0.1.1](#BigDL-0.1.1)
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

<a name="BigDL-0.1.1"></a>
### Install BigDL-0.1.1 ###

1. Download Spark2.x:  
```wget https://d3kbcqa49mib13.cloudfront.net/spark-2.1.0-bin-hadoop2.7.tgz ```
2. Extract the tar ball and set SPARK_HOME
```
tar -zxvf spark-2.1.0-bin-hadoop2.7.tgz
export SPARK_HOME=path to spark-2.1.0-bin-hadoop2.7
```
3. Install BigDL release via pip (we tested this on pip 9.0.1)
```
pip install --upgrade pip
pip install BigDL==0.1.1rc0     # for Python 2.7
pip3 install BigDL==0.1.1rc0  # for Python 3.n
```

<a name="BigDL-0.2.0-snapshot"></a>
### Install BigDL-0.2.0-snapshot ###

1. Download Spark2.x:  
```wget https://d3kbcqa49mib13.cloudfront.net/spark-2.1.0-bin-hadoop2.7.tgz ```
2. Extract the tar ball and set SPARK_HOME
```
tar -zxvf spark-2.1.0-bin-hadoop2.7.tgz
export SPARK_HOME=path to spark-2.1.0-bin-hadoop2.7
```
3. Install BigDL release via pip (we tested this on pip 9.0.1)
```
pip install --upgrade pip
pip install BigDL==0.2.0.dev3     # for Python 2.7
pip3 install BigDL==0.2.0.dev3  # for Python 3.n
```


