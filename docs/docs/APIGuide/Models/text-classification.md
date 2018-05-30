Analytics Zoo provides pre-defined models having different encoders that can be used for classifying texts.
The model could be fed into NNFrames or BigDL Optimizer directly for training.

---
## Build a TextClassifier Model

**Scala**
```scala
val textClassifier = TextClassifier(classNum, tokenLength, sequenceLength = 500, encoder = "cnn", encoderOutputDim = 256)
```

* `classNum`: The number of text categories to be classified. Positive integer.
* `tokenLength`: The size of each word vector. Positive integer.
* `sequenceLength`: The length of a sequence. Positive integer. Default is 500.
* `encoder`: The encoder for input sequences. String. "cnn" or "lstm" or "gru" are supported. Default is "cnn".
* `encoderOutputDim`: The output dimension for the encoder. Positive integer. Default is 256.

See [here](https://github.com/intel-analytics/analytics-zoo/tree/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/textclassification) for the Scala example that trains the TextClassifier model on 20 Newsgroup dataset and uses the model to do prediction.


**Python**
```python
text_classifier = TextClassifier(class_num, token_length, sequence_length=500, encoder="cnn", encoder_output_dim=256)
```

* `class_num`: The number of text categories to be classified. Positive int.
* `token_length`: The size of each word vector. Positive int.
* `sequence_length`: The length of a sequence. Positive int. Default is 500.
* `encoder`: The encoder for input sequences. String. 'cnn' or 'lstm' or 'gru' are supported. Default is 'cnn'.
* `encoder_output_dim`: The output dimension for the encoder. Positive int. Default is 256.

See [here](https://github.com/intel-analytics/analytics-zoo/tree/master/pyzoo/zoo/examples/textclassification) for the Python example that trains the TextClassifier model on 20 Newsgroup dataset and uses the model to do prediction.

---
## Model Save
After building and training a TextClassifier model, you can save it for future use.

**Scala**
```scala
textClassifier.saveModel(path, weightPath = null, overWrite = false)
```

* `path`: The path to save the model. Local file system, HDFS and Amazon S3 are supported. HDFS path should be like "hdfs://[host]:[port]/xxx". Amazon S3 path should be like "s3a://bucket/xxx".
* `weightPath`: The path to save weights. Default is null.
* `overWrite`: Whether to overwrite the file if it already exists. Default is false.

**Python**
```python
text_classifier.save_model(path, weight_path=None, over_write=False)
```

* `path`: The path to save the model. Local file system, HDFS and Amazon S3 are supported. HDFS path should be like 'hdfs://[host]:[port]/xxx'. Amazon S3 path should be like 's3a://bucket/xxx'.
* `weight_path`: The path to save weights. Default is None.
* `over_write`: Whether to overwrite the file if it already exists. Default is False.

---
## Model Load
To load a TextClassifier model (with weights) saved [above](#model-save):

**Scala**
```scala
TextClassifier.loadModel[Float](path, weightPath = null)
```

* `path`: The path for the pre-defined model. Local file system, HDFS and Amazon S3 are supported. HDFS path should be like "hdfs://[host]:[port]/xxx". Amazon S3 path should be like "s3a://bucket/xxx".
* `weightPath`: The path for pre-trained weights if any. Default is null.

**Python**
```python
TextClassifier.load_model(path, weight_path=None)
```

* `path`: The path for the pre-defined model. Local file system, HDFS and Amazon S3 are supported. HDFS path should be like 'hdfs://[host]:[port]/xxx'. Amazon S3 path should be like 's3a://bucket/xxx'.
* `weight_path`: The path for pre-trained weights if any. Default is None.
