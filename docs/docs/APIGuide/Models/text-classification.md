Analytics Zoo provides pre-defined models having different encoders that can be used for classifying texts.
The model could be fed into NNFrames or BigDL Optimizer directly for training.

---
## **Build a TextClassifier Model**
You can call the following API in Scala and Python respectively to create a `TextClassifier` with *pre-trained GloVe word embeddings as the first layer*.

**Scala**
```scala
val textClassifier = TextClassifier(classNum, embeddingFile, wordIndex = null, sequenceLength = 500, encoder = "cnn", encoderOutputDim = 256)
```

* `classNum`: The number of text categories to be classified. Positive integer.
* `embeddingFile`: The path to the word embedding file. Currently only *glove.6B.50d.txt, glove.6B.100d.txt, glove.6B.200d.txt, glove.6B.300d.txt, glove.42B.300d.txt, glove.840B.300d.txt* are supported. You can download from [here](https://nlp.stanford.edu/projects/glove/).
* `wordIndex`: Map of word (String) and its corresponding index (integer). The index is supposed to __start from 1__ with 0 reserved for unknown words. During the prediction, if you have words that are not in the wordIndex for the training, you can map them to index 0. Default is null. In this case, all the words in the embeddingFile will be taken into account and you can call `WordEmbedding.getWordIndex(embeddingFile)` to retrieve the map.
* `sequenceLength`: The length of a sequence. Positive integer. Default is 500.
* `encoder`: The encoder for input sequences. String. "cnn" or "lstm" or "gru" are supported. Default is "cnn".
* `encoderOutputDim`: The output dimension for the encoder. Positive integer. Default is 256.

See [here](https://github.com/intel-analytics/analytics-zoo/tree/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/textclassification) for the Scala example that trains the TextClassifier model on 20 Newsgroup dataset and uses the model to do prediction.


**Python**
```python
text_classifier = TextClassifier(class_num, embedding_file, word_index=None, sequence_length=500, encoder="cnn", encoder_output_dim=256)
```

* `class_num`: The number of text categories to be classified. Positive int.
* `embedding_file`: The path to the word embedding file. Currently only *glove.6B.50d.txt, glove.6B.100d.txt, glove.6B.200d.txt, glove.6B.300d.txt, glove.42B.300d.txt, glove.840B.300d.txt* are supported. You can download from [here](https://nlp.stanford.edu/projects/glove/).
* `word_index`: Dictionary of word (string) and its corresponding index (int). The index is supposed to __start from 1__ with 0 reserved for unknown words. During the prediction, if you have words that are not in the wordIndex for the training, you can map them to index 0. Default is None. In this case, all the words in the embedding_file will be taken into account and you can call `WordEmbedding.get_word_index(embedding_file)` to retrieve the dictionary.
* `sequence_length`: The length of a sequence. Positive int. Default is 500.
* `encoder`: The encoder for input sequences. String. 'cnn' or 'lstm' or 'gru' are supported. Default is 'cnn'.
* `encoder_output_dim`: The output dimension for the encoder. Positive int. Default is 256.

See [here](https://github.com/intel-analytics/analytics-zoo/tree/master/pyzoo/zoo/examples/textclassification) for the Python example that trains the TextClassifier model on 20 Newsgroup dataset and uses the model to do prediction.

---
## **Save Model**
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
## **Load Model**
To load a TextClassifier model (with weights) saved [above](#save-model):

**Scala**
```scala
TextClassifier.loadModel(path, weightPath = null)
```

* `path`: The path for the pre-defined model. Local file system, HDFS and Amazon S3 are supported. HDFS path should be like "hdfs://[host]:[port]/xxx". Amazon S3 path should be like "s3a://bucket/xxx".
* `weightPath`: The path for pre-trained weights if any. Default is null.

**Python**
```python
TextClassifier.load_model(path, weight_path=None)
```

* `path`: The path for the pre-defined model. Local file system, HDFS and Amazon S3 are supported. HDFS path should be like 'hdfs://[host]:[port]/xxx'. Amazon S3 path should be like 's3a://bucket/xxx'.
* `weight_path`: The path for pre-trained weights if any. Default is None.
