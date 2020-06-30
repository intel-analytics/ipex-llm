# **Examples for Keras Model Loading**

We provide several simple examples here to show how to load a Keras model into BigDL and running the model in a distributed fashion.

You may need to see the page [Keras Support](../../../../docs/docs/ProgrammingGuide/keras-support.md) first before going into these examples.

Note that the Keras version we support and test is [__Keras 1.2.2__](https://faroit.github.io/keras-docs/1.2.2/) with TensorFlow backend.

For the sake of illustration, in these examples, we first define a model in Keras 1.2.2, then save it as a JSON/HDF5 file and load it into BigDL.

You can directly run these examples if you [install BigDL from pip](../../../../docs/docs/PythonUserGuide/install-from-pip.md). After the training, good accuracy can be achieved.

In the future, we are going to provide more examples for users to try out.

# **How to run these examples**

Here we take `minst_cnn.py` to show how to run the examples under this directory. You can run other examples in a similar way as the following:

* Run in the command line directly using `python`:

```
source BigDL/pyspark/test/dev/prepare_env.sh
python BigDL/pyspark/bigdl/examples/keras/mnist_cnn.py
```

* Run using `spark-submit` in local mode:

```
BigDL_HOME=...
SPARK_HOME=...
BIGDL_VERSION=...
MASTER=local[*]
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
    --py-files ${PYTHON_API_ZIP_PATH},${BigDL_HOME}/pyspark/bigdl/examples/keras/mnist_cnn.py  \
    --properties-file ${BigDL_HOME}/dist/conf/spark-bigdl.conf \
    --jars ${BigDL_JAR_PATH} \
    --conf spark.driver.extraClassPath=${BigDL_JAR_PATH} \
    --conf spark.executor.extraClassPath=bigdl-${BIGDL_VERSION}-jar-with-dependencies.jar \
    ${BigDL_HOME}/pyspark/bigdl/examples/keras/mnist_cnn.py
```
* ```--batchSize``` an option that can be used to set batch size.
* ```--max_epoch``` an option that can be used to set how many epochs for which the model is to be trained.
* ```--optimizerVersion``` an option that can be used to set DistriOptimizer version, the default value is "optimizerV1".