# **Python Support**

* [BigDL Python API](#bigdl-python-api)
    * [Tutorial: Text Classification using BigDL Python API](#tutorial-text-classification-using-bigdl-python-api)
* [Running BigDL Python Programs](#running-bigdl-python-programs)
    * [Automatically Packaging Python Dependency](#automatically-packaging-python-dependency)
* [Running BigDL Python Code in Notebooks](#running-bigdl-python-code-in-notebooks)

## **BigDL Python API**
Python is one of the most widely used language in the big data and data science community, and BigDL provides full support for Python APIs (built on top of PySpark), which are similar to the [Scala interface](https://www.javadoc.io/doc/com.intel.analytics.bigdl/bigdl/0.1.0-doc). (Please note currently the Python support has been tested with Python 2.7 and Spark 1.6 / Spark 2.0). 

With the full Python API support in BigDL, users can use deep learning models in BigDL together with existing Python libraries (e.g., Numpy and Pandas), which automatically runs in a distributed fashion to process large volumes of data on Spark.  

### **Tutorial: Text Classification using BigDL Python API**  

This tutorial describes the [textclassifier](https://github.com/intel-analytics/BigDL/tree/master/pyspark/dl/models/textclassifier) example written using BigDL Python API, which builds a text classifier using a CNN (convolutional neural network) or LSTM or GRU model (as specified by the user). (It was first described by [this Keras tutorial](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html))

The example first creates the `SparkContext` using the SparkConf` return by the `create_spark_conf()` method, and then initialize the engine:
```python
  sc = SparkContext(appName="text_classifier",
                    conf=create_spark_conf())
  init_engine()
```

It then loads the [20 Newsgroup dataset](http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html) into RDD, and transforms the input data into an RDD of `Sample`. (Each `Sample` in essence contains a tuple of two NumPy ndarray representing the feature and label).

```python
  texts = news20.get_news20()
  data_rdd = sc.parallelize(texts, 2)
  ...
  sample_rdd = vector_rdd.map(
      lambda (vectors, label): to_sample(vectors, label, embedding_dim))
  train_rdd, val_rdd = sample_rdd.randomSplit(
      [training_split, 1-training_split])   
```

After that, the example creates the neural network model as follows:
```python
def build_model(class_num):
    model = Sequential()

    if model_type.lower() == "cnn":
        model.add(Reshape([embedding_dim, 1, sequence_len]))
        model.add(SpatialConvolution(embedding_dim, 128, 5, 1))
        model.add(ReLU())
        model.add(SpatialMaxPooling(5, 1, 5, 1))
        model.add(SpatialConvolution(128, 128, 5, 1))
        model.add(ReLU())
        model.add(SpatialMaxPooling(5, 1, 5, 1))
        model.add(Reshape([128]))
    elif model_type.lower() == "lstm":
        model.add(Recurrent()
                  .add(LSTM(embedding_dim, 128)))
        model.add(Select(2, -1))
    elif model_type.lower() == "gru":
        model.add(Recurrent()
                  .add(GRU(embedding_dim, 128)))
        model.add(Select(2, -1))
    else:
        raise ValueError('model can only be cnn, lstm, or gru')

    model.add(Linear(128, 100))
    model.add(Linear(100, class_num))
    model.add(LogSoftMax())
    return model
```
Finally the example creates the `Optimizer` (which accepts both the model and the training Sample RDD) and trains the model by calling `Optimizer.optimize()`:

```python
optimizer = Optimizer(
    model=build_model(news20.CLASS_NUM),
    training_rdd=train_rdd,
    criterion=ClassNLLCriterion(),
    end_trigger=MaxEpoch(max_epoch),
    batch_size=batch_size,
    optim_method="Adagrad",
    state=state)
...
train_model = optimizer.optimize()
```

## **Running BigDL Python Programs**
A BigDL Python program runs as a standard PySPark program, which requires all Python dependency (e.g., NumPy) used by the program be installed on each node in the Spark cluster. One can run the BigDL [lenet Python example](https://github.com/intel-analytics/BigDL/tree/master/pyspark/dl/models/lenet) using [spark-submit](http://spark.apache.org/docs/latest/submitting-applications.html) as follows:
```bash
PYTHON_API_PATH=${BigDL_HOME}/dist/lib/bigdl-VERSION-python-api.zip
BigDL_JAR_PATH=${BigDL_HOME}/dist/lib/bigdl-VERSION-jar-with-dependencies.jar
PYTHONPATH=${PYTHON_API_ZIP_PATH}:$PYTHONPATH
   
source ${BigDL_HOME}/dist/bin/bigdl.sh
   
${SPARK_HOME}/bin/spark-submit \
    --master ${MASTER} \
    --driver-memory 10g  \
    --driver-cores 4  \
    --executor-memory 20g \
    --total-executor-cores ${TOTAL_CORES}\
    --executor-cores 10 ${EXECUTOR_CORES} \
    --py-files ${PYTHON_API_PATH},${BigDL_HOME}/pyspark/dl/models/lenet/lenet5.py  \
    --properties-file ${BigDL_HOME}/dist/conf/spark-bigdl.conf \
    --jars ${BigDL_JAR_PATH} \
    --conf spark.driver.extraClassPath=${BigDL_JAR_PATH} \
    --conf spark.executor.extraClassPath=bigdl-VERSION-jar-with-dependencies.jar \
    ${BigDL_HOME}/pyspark/dl/models/lenet/lenet5.py
```

### **Automatically Packaging Python Dependency**
You can run BigDL Python programs on YARN clusters without changes to the cluster (e.g., no need to pre-install the Python dependency). You  can first package all the required Python dependency into a virtual environment on the local node (where you will run the spark-submit command), and then directly use spark-submit to run the BigDL Python program on the YARN cluster (using that virtual environment). Please refer to this [patch](https://github.com/intel-analytics/BigDL/pull/706) for more details.

## **Running BigDL Python Code in Notebooks**
With the full Python API support in BigDL, users can now use BigDL together with powerful notebooks (such as Jupyter notebook) in a distributed fashion across the cluster, combining Python libraries, Spark SQL / dataframes and MLlib, deep learning models in BigDL, as well as interactive visualization tools.

First, install all the necessary libraries on the local node where you will run Jupyter, e.g., 
```bash
sudo apt install python
sudo apt install python-pip
sudo pip install numpy scipy pandas scikit-learn matplotlib seaborn wordcloud
```

Then, you can launch the Jupyter notebook as follows:
```bash
PYTHON_API_PATH=${BigDL_HOME}/dist/lib/bigdl-0.1.0-python-api.zip
BigDL_JAR_PATH=${BigDL_HOME}/dist/lib/bigdl-0.1.0-jar-with-dependencies.jar

export PYTHONPATH=${PYTHON_API_PATH}:$PYTHONPATH
export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS="notebook --notebook-dir=./ --ip=* --no-browser"

source ${BigDL_HOME}/dist/bin/bigdl.sh

${SPARK_HOME}/bin/pyspark \
  --master ${MASTER} \
  --properties-file ${BigDL_HOME}/dist/conf/spark-bigdl.conf \
  --driver-memory 10g  \
  --driver-cores 4  \
  --executor-memory 20g \
  --total-executor-cores {TOTAL_CORES} \
  --executor-cores {EXECUTOR_CORES} \
  --py-files ${PYTHON_API_PATH} \
  --jars ${BigDL_JAR_PATH} \
  --conf spark.driver.extraClassPath=${BigDL_JAR_PATH} \
  --conf spark.executor.extraClassPath=bigdl-0.1.0-jar-with-dependencies.jar
```

After successfully launching Jupyter, you will be able to navigate to the notebook dashboard using your browser. You can find the exact URL in the console output when you started Jupyter; by default, the dashboard URL is http://your_node:8888/