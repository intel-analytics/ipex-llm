## Summary
 This example use a (pre-trained GloVe embedding) to convert word to vector,
 and uses it to train a CNN, LSTM or GRU text classification model on a 20 Newsgroup dataset
 with 20 different categories. CNN model can achieve around 85% accuracy after 20 epochs training.
 LSTM and GRU are a little difficult to train, which need more epochs to achieve the equivalent result.
(It was first described in: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html)
## Data
* Embedding: 200-dimensional pre-trained GloVe embeddings of 400k words which trained on a 2014 dump of English Wikipedia.
* Training data: "20 Newsgroup dataset" which containing 20 categories and with totally 19997 texts.


## How to run this example:

- Please note that due to some permission issue, this example **cannot** be run on Windows.

### Data preparation
- You don't need to download the data by yourself.
- If there is no [Pre-train GloVe word embeddings](http://nlp.stanford.edu/data/glove.6B.zip)
or [20 Newsgroup dataset](http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html) in
`./data/news20` directory with the following structure looks like:

```{r, engine='sh'}
$ [/tmp/news20]$ tree . -L 1
  .
  ├── 20news-18828.tar.gz
  └── glove.6B.zip
```
- The example code would automatically download the data during the first run.

### Run via pip install
- [Install from pip](https://bigdl-project.github.io/0.13.0-SNAPSHOT/#PythonUserGuide/install-from-pip/)
- Optional: check [Run after pip install](https://bigdl-project.github.io/0.13.0-SNAPSHOT/#PythonUserGuide/run-from-pip/)
- Run the following command locally
```
python ${BigDL_HOME}/pyspark/bigdl/models/textclassifier/textclassifier.py --max_epoch 3 --model cnn
      
```

### Run via spark-submit
- [Install without pip](https://bigdl-project.github.io/0.13.0-SNAPSHOT/#PythonUserGuide/install-without-pip/)
- Optional: check [Run without pip](https://bigdl-project.github.io/0.13.0-SNAPSHOT/#PythonUserGuide/run-without-pip/)
- Run the following command
```{r, engine='sh'}
        PYTHONHASHSEED=0
        BigDL_HOME=...
        SPARK_HOME=...
        MASTER=...
        PYTHON_API_ZIP_PATH=${BigDL_HOME}/dist/lib/bigdl-VERSION-python-api.zip
        BigDL_JAR_PATH=${BigDL_HOME}/dist/lib/bigdl-VERSION-jar-with-dependencies.jar
        PYTHONPATH=${PYTHON_API_ZIP_PATH}:$PYTHONPATH

        ${SPARK_HOME}/bin/spark-submit \
            --master ${MASTER} \
            --driver-cores 4  \
            --driver-memory 25g  \
            --total-executor-cores 4  \
            --executor-cores 4  \
            --executor-memory 20g \
            --py-files ${PYTHON_API_ZIP_PATH},${BigDL_HOME}/pyspark/bigdl/models/textclassifier/textclassifier.py  \
            --jars ${BigDL_JAR_PATH} \
            --conf spark.driver.extraClassPath=${BigDL_JAR_PATH} \
            --conf spark.executor.extraClassPath=bigdl-VERSION-jar-with-dependencies.jar \
            --conf spark.executorEnv.PYTHONHASHSEED=${PYTHONHASHSEED} \
            ${BigDL_HOME}/pyspark/bigdl/models/textclassifier/textclassifier.py \
             --model cnn
```
* `--data_path` option can be used to set the path for downloading news20 data, the default value is /tmp/news20. Make sure that you have write permission to the specified path.

* `--max_epoch` option can be used to set how many epochs the model to be trained

* `--model` option can be used to choose a model to be trained, three models are supported in this example,
which are `cnn`, `lstm` and `gru`, default is `cnn`

* `--batchSize` option can be used to set batch size, the default value is 128.

* `--embedding_dim` option can be used to set the embedding size of word vector, the default value is 200.

* `--learning_rate` option can be used to set learning rate, default is 0.05.

* `--optimizerVersion` option can be used to set DistriOptimizer version, the value can be "optimizerV1" or "optimizerV2".

To verify the accuracy, search "accuracy" from log:

```{r, engine='sh'}
   [Epoch 20 15104/15005][Iteration 2360]

   top1 accuracy is Accuracy(correct: 3239, count: 3823, accuracy: 0.84724)
```
