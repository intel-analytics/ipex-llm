# Recurrent Neural Network

Model that supports sequence to sequence processing

This is an implementation of Simple Recurrent Neural Networks for Language Modeling. Please refer to the [original paper](http://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf) by Tomas Mikolov.

The implementation of RNNs in this code is referred to in the [Keras Recurrent](https://keras.io/layers/recurrent/) documentation.


## Get the BigDL files

Please build BigDL referring to [Build Page](https://bigdl-project.github.io/master/#ScalaUserGuide/install-build-src/).


## Prepare the Input Data
You can download the Tiny Shakespeare Texts corpus from [here](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt).

After downloading the text, please place it into an appropriate directory (e.g `/opt/text/input.txt`). Please separate it into `train.txt` and `val.txt`. In our case, we just select 80 percentage of the input to be train and remaining 20 percentage to be val. The program will later read in the original text file from this directory.
```shell
export LANG=en_US.UTF-8
head -n 8000 input.txt > val.txt
tail -n +8000 input.txt > train.txt
```

If you run on spark local mode, you can skip this step, we will download the file for you.

### Sample Text

The input text may look as follows:

```
      First Citizen:
      Before we proceed any further, hear me speak.

      All:
      Speak, speak.

      First Citizen:
      You are all resolved rather to die than to famish?
```
## Preprocessing

The <code>sentences_split</code>, <code>sentence_tokenizer</code> use [NLTK Toolkit](http://www.nltk.org/). NLTK is included
in created virtual env(see ${BigDL_HOME}/pyspark/python_package/README.md to see how to create virtual env). As NLTK expects english.pickle to tokenizer, we need download punkt and navigate into the folder for packaging
```
$python
>>> import nltk
>>> nltk.download('punkt')
```
then you can find nltk_data folder, zip file with tokenizer.zip#tokenizer
``` 
$ cd nltk_data/tokenizers/
$ zip -r ../../tokenizers.zip *
$ cd ../../
```

### Sample Sequence of Processed Data
```
      3998,3875,3690,3999
      3998,3171,3958,2390,3832,3202,3855,3983,3883,3999
      3998,3667,3999
      3998,3151,3883,3999
      3998,3875,3690,3999
```

## Train the Model
Example command in yarn:
```bash

BigDL_HOME=...
SPARK_HOME=...
MASTER=yarn
PYTHON_API_PATH=${BigDL_HOME}/dist/lib/bigdl-VERSION-python-api.zip
BigDL_JAR_PATH=${BigDL_HOME}/dist/lib/bigdl-VERSION-jar-with-dependencies.jar
PYTHONPATH=${PYTHON_API_PATH}:$PYTHONPATH
# set http_proxy if you need proxy to access internet
http_proxy=...
PYSPARK_DRIVER_PYTHON=./venv/bin/python PYSPARK_PYTHON=./venv.zip/venv/bin/python ${SPARK_HOME}/bin/spark-submit \
       --master ${MASTER} \
       --deploy-mode client \
       --conf spark.executorEnv.http_proxy=${http_proxy} \
       --driver-memory 10g  \
       --executor-cores 1  \
       --executor-memory 60g \
       --py-files ${PYTHON_API_PATH} \
       --properties-file ${BigDL_HOME}/dist/conf/spark-bigdl.conf \
       --jars ${BigDL_JAR_PATH} \
       --archives venv.zip,tokenizers.zip#tokenizers \
       --conf spark.driver.extraClassPath=${BigDL_JAR_PATH} \
       --conf spark.executor.extraClassPath=bigdl-version-jar-with-dependencies.jar \
       --conf spark.yarn.appMasterEnv.NLTK_DATA=./ \
       --num-executors 1 \
       ${BigDL_HOME}/pyspark/bigdl/models/rnn/rnnexample.py --folder hdfs://xxx:9000/rnn/ --batchSize 12
```

* `--folder` hdfs directory where `train.txt` and `val.txt` are located. the default value is /tmp/rnn.
* `--batchSize` option can be used to set batch size, the default value is 12.
* `--hiddenSize` hidden unit size in the rnn cell, the default value is 40.
* `--vocabSize` vocabulary size, the default value is 4000.
* `--learningRate` inital learning rate, the default value is 0.1.
* `--weightDecay` weight decay, the default value is 0.
* `--momentum` momentum, the default value is 0.
* `--dampening` dampening for momentum, the default value is 0.
* `--maxEpoch` max number of epochs to train, the default value is 30.

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
   public static final int ntc = 27;
   means the index of format ntc is 27.
   ```
3. Run spark-submit command with correct configurations
   ```
   --conf "spark.driver.extraJavaOptions=-Dbigdl.engineType=mkldnn"
   --conf "spark.executor.extraJavaOptions=-Dbigdl.engineType=mkldnn"
   ```

## Expected Training Output
Users can see the Loss of the model printed by the program. The Loss, in this case, is the perplexity of the language model. The lower, the better.
```
INFO  DistriOptimizer$:247 - [Epoch 1 0/6879][Iteration 1][Wall Clock 0.0s] Train 12 in 4.926679827seconds. Throughput is 2.4357176 records/second. Loss is 8.277311. Current learning rate is 0.1.
INFO  DistriOptimizer$:247 - [Epoch 1 12/6879][Iteration 2][Wall Clock 4.926679827s] Train 12 in 2.622718594seconds. Throughput is 4.575405 records/second. Loss is 8.07377. Current learning rate is 0.1.
INFO  DistriOptimizer$:247 - [Epoch 1 24/6879][Iteration 3][Wall Clock 7.549398421s] Train 12 in 2.478575083seconds. Throughput is 4.8414917 records/second. Loss is 7.8527904. Current learning rate is 0.1.
INFO  DistriOptimizer$:247 - [Epoch 1 36/6879][Iteration 4][Wall Clock 10.027973504s] Train 12 in 2.475138056seconds. Throughput is 4.8482146 records/second. Loss is 7.581617. Current learning rate is 0.1.
...
```