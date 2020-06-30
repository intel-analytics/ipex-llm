# LeNet5 Model on MNIST

LeNet5 is a classical CNN model used in digital number classification. For detail information,
please refer to <http://yann.lecun.com/exdb/lenet/>.

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
        ${BigDL_HOME}/pyspark/bigdl/models/lenet/lenet5.py \
        --action train \
        --dataPath /tmp/mnist
 ```

* ```--action``` it can be train or test.

* ```--dataPath``` option can be used to set the path for downloading mnist data, the default value is /tmp/mnist. Make sure that you have write permission to the specified path.

* ```--batchSize``` option can be used to set batch size, the default value is 128.

* ```--endTriggerType``` option can be used to control how to end the training process, the value can be "epoch" or "iteration" and default value is "epoch".

* ```--endTriggerNum``` use together with ```endTriggerType```, the default value is 20.

* ```--modelPath``` option can be used to set model path for testing, the default value is /tmp/lenet5/model.470.

* ```--checkpointPath``` option can be used to set checkpoint path for saving model, the default value is /tmp/lenet5/.

* ```--optimizerVersion``` option can be used to set DistriOptimizer version, the value can be "optimizerV1" or "optimizerV2".

##### In order to use MKL-DNN as the backend, you should:
1. Define a graph model with Model or convert a sequential model to a graph model using:
   ```
   convertedModel = sequentialModel.to_graph()
   ```
2. Specify the input and output formats of it.
   For example:
   ```
   theDefinedModel.set_input_formats([theInputFormatIndex])
   theDefinedModel.set_output_formats([theOutputFormatIndex])
   ```
   BigDL needs these format information to build a graph running with MKL-DNN backend.
   
   The format index of input or output format can be checked
   in: 
   ```
   ${BigDL-core}/native-dnn/src/main/java/com/intel/analytics/bigdl/mkl/Memory.java
   
   For instance:
   public static final int nchw = 7;
   means the index of format nchw is 7.
   
   public static final int nc = 4;
   means the index of format nc is 4.
   
   ```
3. Run spark-submit command with correct configurations
   ```
   --conf "spark.driver.extraJavaOptions=-Dbigdl.engineType=mkldnn"
   --conf "spark.executor.extraJavaOptions=-Dbigdl.engineType=mkldnn"
   ```

To verify the accuracy, search "accuracy" from log:

```
INFO  DistriOptimizer$:247 - [Epoch 1 0/60000][Iteration 1][Wall Clock 0.0s] Train 128 in xx seconds. Throughput is xx records/second.

INFO  DistriOptimizer$:522 - Top1Accuracy is Accuracy(correct: 9572, count: 10000, accuracy: 0.9572)

```

Or you can train a LeNet model directly in shell after installing BigDL from pip:
```
python ${BigDL_HOME}/pyspark/bigdl/models/lenet/lenet5.py
```
