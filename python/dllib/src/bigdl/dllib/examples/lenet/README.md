# LeNet5 Model on MNIST using Keras-Style API

LeNet5 is a classical CNN model used in digital number classification. For detailed information with regard to LeNet, please refer to <http://yann.lecun.com/exdb/lenet/>.

This example is the same as [../../model/lenet/lenet5.py](../../models/lenet/lenet5.py), except that here we use the new set of Keras-Style API in BigDL for model definition and training, which is more user-friendly.

## Install dependencies
 * [Install dependencies](../../../README.md#install.dependencies)

## How to run this example:
Please note that due to some permission issue, this example **cannot** be run on Windows.


Program would download the mnist data into ```/tmp/mnist``` automatically by default.

```
/tmp/mnist$ tree .
.
├── t10k-images-idx3-ubyte.gz
├── t10k-labels-idx1-ubyte.gz
├── train-images-idx3-ubyte.gz
└── train-labels-idx1-ubyte.gz

```

We would train a LeNet model in spark local mode with the following commands and you can distribute it across cluster by modifying the spark master and the executor cores.

```
    BigDL_HOME=...
    SPARK_HOME=...
    MASTER=local[*]
    PYTHON_API_ZIP_PATH=${BigDL_HOME}/dist/lib/bigdl-VERSION-python-api.zip
    BigDL_JAR_PATH=${BigDL_HOME}/dist/lib/bigdl-VERSION-jar-with-dependencies.jar
    PYTHONPATH=${PYTHON_API_ZIP_PATH}:$PYTHONPATH
    ${SPARK_HOME}/bin/spark-submit \
        --master ${MASTER} \
        --driver-cores 2  \
        --driver-memory 2g  \
        --total-executor-cores 2  \
        --executor-cores 2  \
        --executor-memory 4g \
        --py-files ${PYTHON_API_ZIP_PATH},${BigDL_HOME}/pyspark/bigdl/models/lenet/lenet5.py  \
        --properties-file ${BigDL_HOME}/dist/conf/spark-bigdl.conf \
        --jars ${BigDL_JAR_PATH} \
        --conf spark.driver.extraClassPath=${BigDL_JAR_PATH} \
        --conf spark.executor.extraClassPath=bigdl-VERSION-jar-with-dependencies.jar \
        ${BigDL_HOME}/pyspark/bigdl/models/lenet/lenet5.py
 ```

* ```--dataPath``` an option to set the path for downloading mnist data, the default value is /tmp/mnist. Make sure that you have write permission to the specified path.

* ```--batchSize``` an option to set the batch size, the default value is 128.

* ```--maxEpoch``` an option to set the number of epochs to train the model, the default value is 20.

To verify the accuracy, search "accuracy" from log:

```
INFO  DistriOptimizer$:247 - [Epoch 1 0/60000][Iteration 1][Wall Clock 0.0s] Train 128 in xx seconds. Throughput is xx records/second.

INFO  DistriOptimizer$:522 - Top1Accuracy is Accuracy(correct: 9572, count: 10000, accuracy: 0.9572)

```

Or you can train a LeNet model directly in shell after installing BigDL from pip:
```
python ${BigDL_HOME}/pyspark/bigdl/examples/lenet/lenet.py
```