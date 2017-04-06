The Python API is almost identical to the scala version, and it would map ndarray to tensor for the training samples, so basically user only need to care about how to manipulate ndarray.

This Python binding has been tested with Python 2.7 and Spark 1.6.0 / Spark 2.0.0.


## Run python test
* Package Scala code by: ```$BigDL_HOME/make-dist.sh```

* Set SPARK_HOME and then run: ```$BigDL_HOME/pyspark/test/dev/run-all.sh``` 

## Installing on Ubuntu
1. Build BigDL
[Build Page](https://github.com/intel-analytics/BigDL/wiki/Build-Page)
    * With Spark1.6: ```  $BIGDL_HOME/make-dist.sh ``` 
    * With Spark2.0: ``` $BIGDL_HOME/make-dist.sh -P spark_2.0 ```

2. Install python dependensies(You might want to install them for each worker node):
  * Installing Numpy: 
    ```sudo apt-get install python-numpy```

  * Installing Python setuptools: 
    ```sudo apt-get install -y python-setuptools python-pip```
    
  * Install Jupyter:
    ```sudo pip install jupyter```
    
## Run a Lenet example on standalone cluster
    
 ```
    BigDL_HOME=...
    SPARK_HOME=...
    MASTER=...
    PYTHON_API_ZIP_PATH=${BigDL_HOME}/dist/lib/bigdl-VERSION-python-api.zip
    BigDL_JAR_PATH=${BigDL_HOME}/dist/lib/bigdl-VERSION-jar-with-dependencies.jar
    PYTHONPATH=${PYTHON_API_ZIP_PATH}:$PYTHONPATH
    ${SPARK_HOME}/bin/spark-submit \
        --master ${MASTER} \
        --driver-cores 5  \
       --driver-memory 10g  \
       --total-executor-cores 80  \
       --executor-cores 10  \
       --executor-memory 20g \
       --conf spark.akka.frameSize=64 \
        --py-files ${PYTHON_API_ZIP_PATH},${BigDL_HOME}/pyspark/dl/models/lenet/lenet5.py  \
        --properties-file ${BigDL_HOME}/dist/conf/spark-bigdl.conf \
        --jars ${BigDL_JAR_PATH} \
        --conf spark.driver.extraClassPath=${BigDL_JAR_PATH} \
        --conf spark.executor.extraClassPath=bigdl-VERSION-jar-with-dependencies.jar \
        ${BigDL_HOME}/pyspark/dl/models/lenet/lenet5.py
 ```
details can be found at: [LeNet5](https://github.com/intel-analytics/BigDL/tree/master/pyspark/dl/models/lenet/README.md).

## Launch Jupyter on standalone cluster

 ```
    BigDL_HOME=...                                                                                         
    SPARK_HOME=...
    MASTER=...
    PYTHON_API_ZIP_PATH=${BigDL_HOME}/dist/lib/bigdl-VERSION-python-api.zip
    BigDL_JAR_PATH=${BigDL_HOME}/dist/lib/bigdl-VERSION-jar-with-dependencies.jar

    export PYTHONPATH=${PYTHON_API_ZIP_PATH}:$PYTHONPATH
    export PYSPARK_DRIVER_PYTHON=jupyter
    export PYSPARK_DRIVER_PYTHON_OPTS="notebook --notebook-dir=./  --ip=* --no-browser"

    source ${BigDL_HOME}/dist/bin/bigdl.sh

    ${SPARK_HOME}/bin/pyspark \
        --master ${MASTER} \
        --properties-file ${BigDL_HOME}/dist/conf/spark-bigdl.conf \
        --driver-cores 5  \
       --driver-memory 10g  \
       --total-executor-cores 8  \
       --executor-cores 1  \
       --executor-memory 20g \
       --conf spark.akka.frameSize=64 \
        --py-files ${PYTHON_API_ZIP_PATH} \
        --jars ${BigDL_JAR_PATH} \
        --conf spark.driver.extraClassPath=${BigDL_JAR_PATH} \
        --conf spark.executor.extraClassPath=bigdl-VERSION-jar-with-dependencies.jar
 ```

## Run a CNN/LSTM/GRU Text Classifier example on standalone cluster
Please refer to the page
[python text classifier](https://github.com/intel-analytics/BigDL/tree/master/pyspark/dl/models/textclassifier/README.md).
