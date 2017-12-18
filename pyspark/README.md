The Python API is almost identical to the scala version, and it would map ndarray to tensor for the training samples, so basically user only need to care about how to manipulate ndarray.

This Python binding has been tested with Python 2.7 and Spark 1.6.0 / Spark 2.0.0.


## Run python test
* Package Scala code by: ```$BigDL_HOME/make-dist.sh```

* Set SPARK_HOME and then run: ```$BigDL_HOME/pyspark/test/dev/run-all.sh``` 

<a name="install.dependencies"></a>
## Installing on Ubuntu
1. Build BigDL
[Build Page](https://bigdl-project.github.io/master/#ScalaUserGuide/install-build-src/)
    * With Spark 1.6: ```  $BIGDL_HOME/make-dist.sh -P spark_1.6``` 
    * With Spark 2.0 or later: ``` $BIGDL_HOME/make-dist.sh -P spark_2.x ```

2. Install python dependensies(if you're running cluster mode, you need to install them on client and each worker node):
  * Installing Numpy: 
    ```sudo apt-get install python-numpy```

  * Installing Python setuptools: 
    ```sudo apt-get install -y python-setuptools python-pip```
    
  * Install Jupyter on client node (only if you need to use BigDL within Jupyter notebook):
    ```sudo pip install jupyter```
   
  * Install other python dependency libs if you need to use them in your python application
  
## Run a Lenet example on standalone cluster
    
 ```bash
    BigDL_HOME=...
    SPARK_HOME=...
    MASTER=...
    PYTHON_API_ZIP_PATH=${BigDL_HOME}/dist/lib/bigdl-VERSION-python-api.zip
    BigDL_JAR_PATH=${BigDL_HOME}/dist/lib/bigdl-VERSION-jar-with-dependencies.jar
    export PYTHONPATH=${PYTHON_API_ZIP_PATH}:$PYTHONPATH
    
    ${SPARK_HOME}/bin/spark-submit \
        --master ${MASTER} \
        --driver-cores 5  \
       --driver-memory 10g  \
       --total-executor-cores 80  \
       --executor-cores 10  \
       --executor-memory 20g \
        --py-files ${PYTHON_API_ZIP_PATH},${BigDL_HOME}/pyspark/dl/models/lenet/lenet5.py  \
        --properties-file ${BigDL_HOME}/dist/conf/spark-bigdl.conf \
        --jars ${BigDL_JAR_PATH} \
        --conf spark.driver.extraClassPath=${BigDL_JAR_PATH} \
        --conf spark.executor.extraClassPath=bigdl-VERSION-jar-with-dependencies.jar \
        ${BigDL_HOME}/pyspark/dl/models/lenet/lenet5.py
 ```
details can be found at: [LeNet5](https://github.com/intel-analytics/BigDL/tree/master/pyspark/bigdl/models/lenet/README.md).

## Launch Jupyter on standalone cluster

 ```bash
    BigDL_HOME=...                                                                                         
    SPARK_HOME=...
    MASTER=...
    PYTHON_API_ZIP_PATH=${BigDL_HOME}/dist/lib/bigdl-VERSION-python-api.zip
    BigDL_JAR_PATH=${BigDL_HOME}/dist/lib/bigdl-VERSION-jar-with-dependencies.jar

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
       --conf spark.akka.frameSize=64 \
        --py-files ${PYTHON_API_ZIP_PATH} \
        --jars ${BigDL_JAR_PATH} \
        --conf spark.driver.extraClassPath=${BigDL_JAR_PATH} \
        --conf spark.executor.extraClassPath=bigdl-VERSION-jar-with-dependencies.jar
 ```

## Run a CNN/LSTM/GRU Text Classifier example on standalone cluster
Please refer to the page
[python text classifier](https://github.com/intel-analytics/BigDL/tree/master/pyspark/bigdl/models/textclassifier/README.md).
