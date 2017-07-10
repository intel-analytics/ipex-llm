# LeNet5 Model on MNIST

LeNet5 is a classical CNN model used in digital number classification. For detail information,
please refer to <http://yann.lecun.com/exdb/lenet/>.

## How to run this example:

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
        --conf spark.akka.frameSize=64 \
        --py-files ${PYTHON_API_ZIP_PATH},${BigDL_HOME}/pyspark/bigdl/models/lenet/lenet5.py  \
        --properties-file ${BigDL_HOME}/dist/conf/spark-bigdl.conf \
        --jars ${BigDL_JAR_PATH} \
        --conf spark.driver.extraClassPath=${BigDL_JAR_PATH} \
        --conf spark.executor.extraClassPath=bigdl-VERSION-jar-with-dependencies.jar \
        ${BigDL_HOME}/pyspark/bigdl/models/lenet/lenet5.py \
        --action train
 ```


* ```--action``` it can be train or test.

* ```--batchSize``` option can be used to set batch size, the default value is 128.

* ```--endTriggerType``` option can be used to control how to end the training process, the value can be "epoch" or "iteration" and default value is "epoch".

* ```--endTriggerNum``` use together with ```endTriggerType```, the default value is 20.

* ```--modelPath``` option can be used to set model path for testing, the default value is /tmp/lenet5/model.470.

* ```--checkpointPath``` option can be used to set checkpoint path for saving model, the default value is /tmp/lenet5/.

To verify the accuracy, search "accuracy" from log:

```
INFO  DistriOptimizer$:247 - [Epoch 1 0/60000][Iteration 1][Wall Clock 0.0s] Train 128 in xx seconds. Throughput is xx records/second.

INFO  DistriOptimizer$:522 - Top1Accuracy is Accuracy(correct: 9572, count: 10000, accuracy: 0.9572)


```
