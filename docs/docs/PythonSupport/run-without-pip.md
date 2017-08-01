First of all, you need to obtain the BigDL libs. Refer to [Install from pre built](../UserGuide/install-pre-built.md) or [Install from source code](../UserGuide/install-build-src.md) for more details


## **A quick launch for local mode**

```bash
cd $BIGDL_HOME/dist/lib 
BIGDL_VERSION=...
${SPARK_HOME}/bin/pyspark --master local[4] \
--conf spark.driver.extraClassPath=bigdl-${BIGDL_VERSION}-jar-with-dependencies.jar \
--py-files bigdl-${BIGDL_VERSION}-python-api.zip \
--properties-file ../conf/spark-bigdl.conf 
```

 [Example code to verify if run successfully](run-from-pip.md#code.verification)


## **Run from spark-submit**

- A BigDL Python program runs as a standard PySPark program, which requires all Python dependency (e.g., NumPy) used by the program be installed on each node in the Spark cluster. You can try run the BigDL [lenet Python example](https://github.com/intel-analytics/BigDL/tree/master/pyspark/bigdl/models/lenet) using [spark-submit](http://spark.apache.org/docs/latest/submitting-applications.html) as follows:
- __Ensure every path is valid__ 

```bash
   BigDL_HOME=...
   SPARK_HOME=...
   BIGDL_VERSION=...
   MASTER=...
   PYTHON_API_ZIP_PATH=${BigDL_HOME}/dist/lib/bigdl-${BIGDL_VERSION}-python-api.zip
   BigDL_JAR_PATH=${BigDL_HOME}/dist/lib/bigdl-${BIGDL_VERSION}-jar-with-dependencies.jar
   PYTHONPATH=${PYTHON_API_ZIP_PATH}:$PYTHONPATH
   
   ${SPARK_HOME}/bin/spark-submit \
       --master ${MASTER} \
       --driver-cores 5  \
      --driver-memory 10g  \
      --total-executor-cores 80  \
      --executor-cores 10  \
      --executor-memory 20g \
       --py-files ${PYTHON_API_ZIP_PATH},${BigDL_HOME}/pyspark/bigdl/models/lenet/lenet5.py  \
       --properties-file ${BigDL_HOME}/dist/conf/spark-bigdl.conf \
       --jars ${BigDL_JAR_PATH} \
       --conf spark.driver.extraClassPath=${BigDL_JAR_PATH} \
       --conf spark.executor.extraClassPath=bigdl-${BIGDL_VERSION}-jar-with-dependencies.jar \
       ${BigDL_HOME}/pyspark/bigdl/models/lenet/lenet5.py
```




## **Run from pyspark + Jupyter**

- With the full Python API support in BigDL, users can now use BigDL together with powerful notebooks (such as Jupyter notebook) in a distributed fashion across the cluster, combining Python libraries, Spark SQL / dataframes and MLlib, deep learning models in BigDL, as well as interactive visualization tools.

- First, install all the necessary libraries on the local node where you will run Jupyter, e.g., 
```bash
sudo apt install python
sudo apt install python-pip
sudo pip install numpy scipy pandas scikit-learn matplotlib seaborn wordcloud
```

- Then, you can launch the Jupyter notebook as follows:
- __Ensure every path is valid__ 

```bash
   BigDL_HOME=...                                                                                         
   BIGDL_VERSION=...
   SPARK_HOME=...
   MASTER=...
   PYTHON_API_ZIP_PATH=${BigDL_HOME}/dist/lib/bigdl-${BIGDL_VERSION}-python-api.zip
   BigDL_JAR_PATH=${BigDL_HOME}/dist/lib/bigdl-${BIGDL_VERSION}-jar-with-dependencies.jar

   export PYTHONPATH=${PYTHON_API_ZIP_PATH}:$PYTHONPATH
   export PYSPARK_DRIVER_PYTHON=jupyter
   export PYSPARK_DRIVER_PYTHON_OPTS="notebook --notebook-dir=./  --ip=* --no-browser"
   
   ${SPARK_HOME}/bin/pyspark \
       --master ${MASTER} \
       --properties-file ${BigDL_HOME}/dist/conf/spark-bigdl.conf \
       --driver-cores 5  \
      --driver-memory 10g  \
      --total-executor-cores 8  \
      --executor-cores 1  \
      --executor-memory 20g \
       --py-files ${PYTHON_API_ZIP_PATH} \
       --jars ${BigDL_JAR_PATH} \
       --conf spark.driver.extraClassPath=${BigDL_JAR_PATH} \
       --conf spark.executor.extraClassPath=bigdl-${BIGDL_VERSION}-jar-with-dependencies.jar
```

After successfully launching Jupyter, you will be able to navigate to the notebook dashboard using your browser. You can find the exact URL in the console output when you started Jupyter; by default, the dashboard URL is http://your_node:8888/

[Example code to verify if run successfully](run-from-pip.md#code.verification)

## BigDL Configuration
Please check [this page](../UserGuide/configuration.md)

## **FAQ**
- TypeError: 'JavaPackage' object is not callable
  - `Check if every path within the launch script is valid expecially the path end with jar `

- java.lang.NoSuchMethodError:XXX
  - `Check if the spark version is match i.e you are using Spark2.x but the underneath BigDL compiled with Spark1.6`
