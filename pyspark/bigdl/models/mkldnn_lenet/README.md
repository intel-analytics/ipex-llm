# LeNet5 Model on MNIST (training and inference with MKLDNN)

LeNet5 is a classical CNN model used in digital number classification. For detail information,
please refer to <http://yann.lecun.com/exdb/lenet/>.

## Install dependencies
 * [Install dependencies](../../../README.md#install.dependencies)

## How to run this example:
Please note that due to some permission issue, this example **cannot** be run on Windows.

This example would demonstrate how to do training and inference with a LeNet model using MKLDNN

Program would download the mnist data into ```/tmp/mnist``` automatically by default.

```
/tmp/mnist$ tree .
.
├── t10k-images-idx3-ubyte.gz
├── t10k-labels-idx1-ubyte.gz
├── train-images-idx3-ubyte.gz
└── train-labels-idx1-ubyte.gz

```

We would train a LeNet model with MKLDNN in spark local mode with the following commands and you can distribute it across cluster by modifying the spark master and the executor cores.

```
    BigDL_HOME=...
    SPARK_HOME=...
    MASTER=local[*]
    PYTHON_API_ZIP_PATH=${BigDL_HOME}/dist/lib/bigdl-VERSION-python-api.zip
    BigDL_JAR_PATH=${BigDL_HOME}/dist/lib/bigdl-VERSION-jar-with-dependencies.jar
    PYTHONPATH=${PYTHON_API_ZIP_PATH}:$PYTHONPATH
    ${SPARK_HOME}/bin/spark-submit \
        --master ${MASTER} \
        --conf "spark.driver.extraJavaOptions=-Dbigdl.engineType=mkldnn"
        --conf "spark.executor.extraJavaOptions=-Dbigdl.engineType=mkldnn"
        --driver-cores 2  \
        --driver-memory 2g  \
        --total-executor-cores 2  \
        --executor-cores 2  \
        --executor-memory 4g \
        --py-files ${PYTHON_API_ZIP_PATH},${BigDL_HOME}/pyspark/bigdl/models/mkldnn_lenet/mkldnn_lenet5.py  \
        --properties-file ${BigDL_HOME}/dist/conf/spark-bigdl.conf \
        --jars ${BigDL_JAR_PATH} \
        --conf spark.driver.extraClassPath=${BigDL_JAR_PATH} \
        --conf spark.executor.extraClassPath=bigdl-VERSION-jar-with-dependencies.jar \
        ${BigDL_HOME}/pyspark/bigdl/models/mkldnn_lenet/mkldnn_lenet5.py \
        --action train \
        --dataPath /tmp/mnist
 ```
 
 Make sure the engine type MKLDNN is set:
 ```
 --conf "spark.driver.extraJavaOptions=-Dbigdl.engineType=mkldnn"
 --conf "spark.executor.extraJavaOptions=-Dbigdl.engineType=mkldnn"
 ```


* ```--action``` it can be train or test.

* ```--dataPath``` option can be used to set the path for downloading mnist data, the default value is /tmp/mnist. Make sure that you have write permission to the specified path.

* ```--batchSize``` option can be used to set batch size, the default value is 128.

* ```--endTriggerType``` option can be used to control how to end the training process, the value can be "epoch" or "iteration" and default value is "epoch".

* ```--endTriggerNum``` use together with ```endTriggerType```, the default value is 20.

* ```--modelPath``` option can be used to set model path for testing, the default value is /tmp/mkldnn_lenet5/model.470.

* ```--checkpointPath``` option can be used to set checkpoint path for saving model, the default value is /tmp/mkldnn_lenet5/.