
When writing python applications with BigDL python API, you may import third-party libraries. Sometimes you may not want or have permission to install these dependencies on executors. This guide is show how to ship your dependencies environment to executors when you submit your application to Spark/Hadoop clusters.

### Install necessary libraries for creating virtual env on your client
* Make sure you already install such libraries(python-setuptools, python-dev, gcc, make, zip, pip) for creating virtual environment. If not, please install them first. For example, on Ubuntu, run these commands to install:
  ```
    apt-get update
    apt-get install -y python-setuptools python-dev
    apt-get install -y gcc make
    apt-get install -y zip
    easy_install pip
  ```	


### Create dependency virtualenv package

In this directory, run this command to create dependency environment package according to dependency descriptions in requirements.txt. You can add your own dependencies in requirements.txt. The current requirements.txt only contains dependencies for BigDL python examples and models.

```
    ./python_package.sh
```    
    
After running this script, there will be venv.zip and venv directory generated in current directory. Use them to submit your python jobs. 
    
### Run Lenet example on YARN Cluster mode

Refer to python_submit_yarn_cluster.sh.example to run lenet example on yarn cluster mode. 
    
 ```
    BigDL_HOME=...
    SPARK_HOME=...
    PYTHON_API_PATH=${BigDL_HOME}/dist/lib/bigdl-VERSION-python-api.zip
    BigDL_JAR_PATH=${BigDL_HOME}/dist/lib/bigdl-VERSION-jar-with-dependencies.jar
    PYTHONPATH=${PYTHON_API_PATH}:$PYTHONPATH
    
    PYSPARK_PYTHON=./venv.zip/venv/bin/python ${SPARK_HOME}/bin/spark-submit \
    --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./venv.zip/venv/bin/python \
    #--conf spark.yarn.appMasterEnv.http_proxy=http://... \
    #--conf spark.executorEnv.http_proxy=http://... \
    --master yarn-cluster \
    --executor-memory 10g \
    --driver-memory 10g \
    --executor-cores 8 \
    --num-executors 2 \
    --properties-file ${BigDL_HOME}/dist/conf/spark-bigdl.conf \
    --jars ${BigDL_JAR_PATH} \
    --py-files ${PYTHON_API_PATH} \
    --archives venv.zip \
    --conf spark.driver.extraClassPath=bigdl-VERSION-jar-with-dependencies.jar \
    --conf spark.executor.extraClassPath=bigdl-VERSION-jar-with-dependencies.jar \
    ${BigDL_HOME}/pyspark/dl/models/lenet/lenet5.py
 ```
details can be found at: [LeNet5](https://github.com/intel-analytics/BigDL/tree/master/pyspark/dl/models/lenet/README.md).

### Run Lenet example on YARN Client mode

Refer to python_submit_yarn_client.sh.example to run lenet example on yarn client mode.

 ```
    BigDL_HOME=...
    SPARK_HOME=...
    PYTHON_API_PATH=${BigDL_HOME}/dist/lib/bigdl-VERSION-python-api.zip
    BigDL_JAR_PATH=${BigDL_HOME}/dist/lib/bigdl-VERSION-jar-with-dependencies.jar
    PYTHONPATH=${PYTHON_API_PATH}:$PYTHONPATH
    # http_proxy=http://...
    PYSPARK_DRIVER_PYTHON=./venv/bin/python PYSPARK_PYTHON=./venv.zip/venv/bin/python ${SPARK_HOME}/bin/spark-submit \
    --master yarn \
    --deploy-mode client \
    #--conf spark.executorEnv.http_proxy=http://... \
    --executor-memory 10g \
    --driver-memory 10g \
    --executor-cores 16 \
    --num-executors 2 \
    --properties-file ${BigDL_HOME}/dist/conf/spark-bigdl.conf \
    --jars ${BigDL_JAR_PATH} \
    --py-files ${PYTHON_API_PATH} \
    --archives venv.zip \
    --conf spark.driver.extraClassPath=${BigDL_JAR_PATH} \
    --conf spark.executor.extraClassPath=bigdl-VERSION-jar-with-dependencies.jar \
    ${BigDL_HOME}/pyspark/dl/models/lenet/lenet5.py
 ```
