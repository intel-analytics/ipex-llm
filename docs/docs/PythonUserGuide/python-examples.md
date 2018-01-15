---

## **Text Classification using BigDL Python API**  

This tutorial describes the [textclassifier]( https://github.com/intel-analytics/BigDL/tree/master/pyspark/bigdl/models/textclassifier) example written using BigDL Python API, which builds a text classifier using a CNN (convolutional neural network) or LSTM or GRU model (as specified by the user). (It was first described by [this Keras tutorial](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html))

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
    optim_method=Adagrad())
...
train_model = optimizer.optimize()
```


