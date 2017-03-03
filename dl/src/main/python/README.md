
BigDL comes with scala API for now. However Python is a powerful programming language for data analysis and with large amount of useful libraries, we are developing a lightweight python binding on top of PySpark which can enable us use Python naively with BigDL. 

The Python API is almost identical to the scala version, and it would map ndarray to tensor for the training samples, so basically user only need to care about how to manipulate ndarray.

This Python binding tested with Python 2.7 and Spark 1.6.0.

Here are the steps for training a simple LeNet model:

1). Create a RDD[Sample]:
```
RDD[..] --transform-->RDD[ndarray, ndarray].map(Sample.from_ndarray(features, label)) --> RDD[Sample]
```
    
2). Define a model:
```
    def build_model(class_num):
        model = Sequential()
        model.add(Reshape([1, 28, 28]))
        model.add(SpatialConvolution(1, 6, 5, 5))
        model.add(Tanh())
        model.add(SpatialMaxPooling(2, 2, 2, 2))
        model.add(Tanh())
        model.add(SpatialConvolution(6, 12, 5, 5))
        model.add(SpatialMaxPooling(2, 2, 2, 2))
        model.add(Reshape([12 * 4 * 4]))
        model.add(Linear(12 * 4 * 4, 100))
        model.add(Tanh())
        model.add(Linear(100, class_num))
        model.add(LogSoftMax())
        return model
 ```
    
3). Create Optimizer and train:
```
    optimizer = Optimizer(
        model=build_model(10),
        training_rdd=train_data,
        criterion=ClassNLLCriterion(),
        optim_method="SGD",
        state=state,
        end_trigger=MaxEpoch(100),
        batch_size=int(options.batchSize))
    optimizer.setvalidation(
        batch_size=32,
        val_rdd=test_data,
        trigger=EveryEpoch(),
        val_method=["top1"]
    )
    optimizer.setcheckpoint(EveryEpoch(), "/tmp/lenet5/")
    trained_model = optimizer.optimize()
```

4) LeNet example can be found from: models/lenet5.py

## Installing on Ubuntu
1. Build BigDL
[Build Page](https://github.com/intel-analytics/BigDL/wiki/Build-Page)
2. Install python dependensies:
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
    PYTHON_API_ZIP_PATH=${BigDL_HOME}/dist/lib/bigdl-0.1.0-SNAPSHOT-python-api.zip
    BigDL_JAR_PATH=${BigDL_HOME}/dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar
    PYTHONPATH=${PYTHON_API_ZIP_PATH}:$PYTHONPATH
    ${SPARK_HOME}/bin/spark-submit \
        --master ${MASTER} \
        --driver-cores 5  \
       --driver-memory 10g  \
       --total-executor-cores 80  \
       --executor-cores 10  \
       --executor-memory 20g \
       --conf spark.akka.frameSize=64 \
        --py-files ${PYTHON_API_ZIP_PATH},${BigDL_HOME}/dl/src/main/python/models/lenet/lenet5.py  \
        --jars ${BigDL_JAR_PATH} \
        --conf spark.driver.extraClassPath=${BigDL_JAR_PATH} \
        --conf spark.executor.extraClassPath=bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
        ${BigDL_HOME}/dl/src/main/python/models/lenet/lenet5.py \
        --coreNum 10 --nodeNum 8
 ```


## Launch Jupyter on standalone cluster

 ```
    export IPYTHON_OPTS="jupyter notebook"

    BigDL_HOME=...                                                                                         
    SPARK_HOME=...
    MASTER=...
    PYTHON_API_ZIP_PATH=${BigDL_HOME}/dist/lib/bigdl-0.1.0-SNAPSHOT-python-api.zip
    BigDL_JAR_PATH=${BigDL_HOME}/dist/lib/bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar

    export PYTHONPATH=${PYTHON_API_ZIP_PATH}:$PYTHONPATH
    export IPYTHON_OPTS="notebook --notebook-dir=./  --ip=* --no-browser"

    ${SPARK_HOME}/bin/pyspark \
        --master ${MASTER} \
        --driver-cores 5  \
       --driver-memory 10g  \
       --total-executor-cores 8  \
       --executor-cores 1  \
       --executor-memory 20g \
       --conf spark.akka.frameSize=64 \
        --py-files ${PYTHON_API_ZIP_PATH} \
        --jars ${BigDL_JAR_PATH} \
        --conf spark.driver.extraClassPath=${BigDL_JAR_PATH} \
        --conf spark.executor.extraClassPath=bigdl-0.1.0-SNAPSHOT-jar-with-dependencies.jar
 ```
