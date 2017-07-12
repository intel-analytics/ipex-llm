---

## **Use an Interactive Shell (W/ pip install)**
 * type `python` in commandline to start a REPL
 * initialize bigdl engine as below, and you'll be able to play with BigDL

```python
 >>> from bigdl.util.common import *
 >>> init_engine()
 >>> import bigdl.version
 >>> bigdl.version.__version__
 '0.1.1rc0'
 >>> from bigdl.nn.layer import *
 >>> linear = Linear(2, 3)
 creating: createLinear
 >>> ... 
```
---

## **User an Interactive Shell (W/O pip install)**

1.Set some environmental variables. Replace SPARK_HOME, BigDL_HOME and BigDL_version, and etc. according to your actual configurations. 

```bash
BigDL_HOME=...
export BigDL_version=bigdl-0.2.0-SNAPSHOT
SPARK_HOME=...
spark.master=local[2]

export PYTHONPATH=SPARK_HOME/python/lib/pyspark.zip:SPARK_HOME/python/lib/py4j-0.9-src.zip:BigDL_HOME/spark/dl/src/main/resources/spark-bigdl.conf:${BigDL_HOME}/dist/lib/${BigDL_version}-python-api.zip
SPARK_CLASSPATH=BigDL_HOME/spark/dl/target/${BigDL_version}-jar-with-dependencies.jar
```

2.source bigdl.sh and set some environmental variables for BigDL. 
```bash
source ${BigDL_HOME}/dist/bin/bigdl.sh  
```

3.start python shell

4.Initialize the engine as below and you'll be able to play with BigDL. 

```python
  >>> from bigdl.util.common import *
  >>> init_engine()
  >>> import bigdl.version
  >>> from bigdl.nn.layer import *
  >>> linear = Linear(2, 3)
  creating: createLinear
  >>> ...
```

---


## **Run Python Program in Command Line (W/O pip install)**
A BigDL Python program runs as a standard PySPark program, which requires all Python dependency (e.g., NumPy) used by the program be installed on each node in the Spark cluster. You can try run the BigDL [lenet Python example](https://github.com/intel-analytics/BigDL/tree/master/pyspark/dl/models/lenet) using [spark-submit](http://spark.apache.org/docs/latest/submitting-applications.html) as follows:

```bash
PYTHON_API_PATH=${BigDL_HOME}/dist/lib/bigdl-VERSION-python-api.zip
BigDL_JAR_PATH=${BigDL_HOME}/dist/lib/bigdl-VERSION-jar-with-dependencies.jar
PYTHONPATH=${PYTHON_API_ZIP_PATH}:$PYTHONPATH
   
${SPARK_HOME}/bin/spark-submit \
    --master ${MASTER} \
    --driver-memory 10g  \
    --driver-cores 4  \
    --executor-memory 20g \
    --total-executor-cores ${TOTAL_CORES}\
    --executor-cores 10 ${EXECUTOR_CORES} \
    --py-files ${PYTHON_API_PATH},${BigDL_HOME}/pyspark/dl/models/lenet/lenet5.py  \
    --properties-file ${BigDL_HOME}/dist/conf/spark-bigdl.conf \
    --jars ${BigDL_JAR_PATH} \
    --conf spark.driver.extraClassPath=${BigDL_JAR_PATH} \
    --conf spark.executor.extraClassPath=bigdl-VERSION-jar-with-dependencies.jar \
    ${BigDL_HOME}/pyspark/dl/models/lenet/lenet5.py
```

If you are writing your own program, remember to create Spark context and initialize engine before calling any BigDL API's, as below.

```python
from bigdl.util.common import *
# Python code example
conf=create_spark_conf()
sc = SparkContext(conf)
init_engine()
```


---

## Use Jupyter Notebook (W/ pip install)

* Start jupyter notebook as you normally did, e.g.
```bash
jupyter notebook --notebook-dir=./ --ip=* --no-browser
```
* Create SparkContext and initialize BigDL engine as below
```python
from bigdl.util.common import *
init_engine()
  
from bigdl.nn.layer import *
linear = Linear(2, 3)
...
```

Sometimes you may need to use SparkContext's `parallelize` method to convert local data into an RDD. You can get SparkContext in below way.
 
```
from bigdl.util.common import *
sc = get_spark_context()
# sc.parallelize(...)
```

---

## Use Jupyter Notebook (W/O pip install)

With the full Python API support in BigDL, users can now use BigDL together with powerful notebooks (such as Jupyter notebook) in a distributed fashion across the cluster, combining Python libraries, Spark SQL / dataframes and MLlib, deep learning models in BigDL, as well as interactive visualization tools.

First, install all the necessary libraries on the local node where you will run Jupyter, e.g., 
```bash
sudo apt install python
sudo apt install python-pip
sudo pip install numpy scipy pandas scikit-learn matplotlib seaborn wordcloud
```

Then, you can launch the Jupyter notebook as follows:
```bash
PYTHON_API_PATH=${BigDL_HOME}/dist/lib/bigdl-0.1.0-python-api.zip
BigDL_JAR_PATH=${BigDL_HOME}/dist/lib/bigdl-0.1.0-jar-with-dependencies.jar

export PYTHONPATH=${PYTHON_API_PATH}:$PYTHONPATH
export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS="notebook --notebook-dir=./ --ip=* --no-browser"

${SPARK_HOME}/bin/pyspark \
  --master ${MASTER} \
  --properties-file ${BigDL_HOME}/dist/conf/spark-bigdl.conf \
  --driver-memory 10g  \
  --driver-cores 4  \
  --executor-memory 20g \
  --total-executor-cores {TOTAL_CORES} \
  --executor-cores {EXECUTOR_CORES} \
  --py-files ${PYTHON_API_PATH} \
  --jars ${BigDL_JAR_PATH} \
  --conf spark.driver.extraClassPath=${BigDL_JAR_PATH} \
  --conf spark.executor.extraClassPath=bigdl-0.1.0-jar-with-dependencies.jar
```

After successfully launching Jupyter, you will be able to navigate to the notebook dashboard using your browser. You can find the exact URL in the console output when you started Jupyter; by default, the dashboard URL is http://your_node:8888/


Now you can create a blank notebook and start some coding. 

Remember to initialize BigDL engine before calling BigDL API's, as shown below.

```python
from bigdl.util.common import *
init_engine()

from bigdl.nn.layer import *
linear = Linear(2, 3)
...
```

Sometimes you may need to use SparkContext's `parallelize` method to convert local    data into an RDD. You can get SparkContext in below way.

```python
from bigdl.util.common import *
sc = get_spark_context()
# sc.parallelize(...)
```

---
## **Use Python on YARN cluster**
You can run BigDL Python programs on YARN clusters without changes to the cluster (e.g., no need to pre-install the      Python dependency). You  can first package all the required Python dependency into a virtual environment on the local    node (where you will run the spark-submit command), and then directly use spark-submit to run the BigDL Python program   on the YARN cluster (using that virtual environment). Please refer to this [patch](https://github.com/intel-analytics/BigDL/pull/706) for more details.
