Analytics Zoo provides pre-defined models having different encoders that can be used for classifying texts.

**Highlights**

1. Easy-to-use Keras-Style defined models which provides compile and fit methods for training. Alternatively, they could be fed into NNFrames or BigDL Optimizer.
2. The encoders we support include CNN, LSTM and GRU.

---
## **Build a TextClassifier model**
You can call the following API in Scala and Python respectively to create a `TextClassifier` with *pre-trained GloVe word embeddings as the first layer*.

**Scala**
```scala
val textClassifier = TextClassifier(classNum, embeddingFile, wordIndex = null, sequenceLength = 500, encoder = "cnn", encoderOutputDim = 256)
```

* `classNum`: The number of text categories to be classified. Positive integer.
* `embeddingFile` The path to the word embedding file. Currently only *glove.6B.50d.txt, glove.6B.100d.txt, glove.6B.200d.txt, glove.6B.300d.txt, glove.42B.300d.txt, glove.840B.300d.txt* are supported. You can download from [here](https://nlp.stanford.edu/projects/glove/).
* `wordIndex` Map of word (String) and its corresponding index (integer). The index is supposed to __start from 1__ with 0 reserved for unknown words. During the prediction, if you have words that are not in the wordIndex for the training, you can map them to index 0. Default is null. In this case, all the words in the embeddingFile will be taken into account and you can call `WordEmbedding.getWordIndex(embeddingFile)` to retrieve the map.
* `sequenceLength`: The length of a sequence. Positive integer. Default is 500.
* `encoder`: The encoder for input sequences. String. "cnn" or "lstm" or "gru" are supported. Default is "cnn".
* `encoderOutputDim`: The output dimension for the encoder. Positive integer. Default is 256.

**Python**
```python
text_classifier = TextClassifier(class_num, embedding_file, word_index=None, sequence_length=500, encoder="cnn", encoder_output_dim=256)
```

* `class_num`: The number of text categories to be classified. Positive int.
* `embedding_file` The path to the word embedding file. Currently only *glove.6B.50d.txt, glove.6B.100d.txt, glove.6B.200d.txt, glove.6B.300d.txt, glove.42B.300d.txt, glove.840B.300d.txt* are supported. You can download from [here](https://nlp.stanford.edu/projects/glove/).
* `word_index` Dictionary of word (string) and its corresponding index (int). The index is supposed to __start from 1__ with 0 reserved for unknown words. During the prediction, if you have words that are not in the wordIndex for the training, you can map them to index 0. Default is None. In this case, all the words in the embedding_file will be taken into account and you can call `WordEmbedding.get_word_index(embedding_file)` to retrieve the dictionary.
* `sequence_length`: The length of a sequence. Positive int. Default is 500.
* `encoder`: The encoder for input sequences. String. 'cnn' or 'lstm' or 'gru' are supported. Default is 'cnn'.
* `encoder_output_dim`: The output dimension for the encoder. Positive int. Default is 256.

---
## **Train a TextClassifier model**
After building the model, we can call compile and fit to train it (with validation).

For training and validation data, you can first read files as `TextSet` (see [here](../APIGuide/FeatureEngineering/text/#read-texts-from-a-directory)) and then do preprocessing (see [here](../APIGuide/FeatureEngineering/text/#textset-transformations)).

**Scala**
```scala
model.compile(optimizer = new Adagrad(learningRate), loss = SparseCategoricalCrossEntropy(), metrics = List(new Accuracy()))
model.fit(trainSet, batchSize, nbEpoch, validateSet)
```

**Python**
```python
model.compile(optimizer=Adagrad(learning_rate, loss="sparse_categorical_crossentropy", metrics=['accuracy'])
model.fit(train_set, batch_size, nb_epoch, validate_set)
```

---
## **Do prediction**
After training the model, it can be used to predict probability distributions.

**Scala**
```scala
val predictSet = textClassifier.predict(validateSet)
```

**Python**
```python
predict_set = text_classifier.predict(validate_set)
```

---
## **Examples**
We provide an example to train the TextClassifier model on 20 Newsgroup dataset and uses the model to do prediction.

See [here](https://github.com/intel-analytics/analytics-zoo/tree/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/textclassification) for the Scala example.

See [here](https://github.com/intel-analytics/analytics-zoo/tree/master/pyzoo/zoo/examples/textclassification) for the Python example.
