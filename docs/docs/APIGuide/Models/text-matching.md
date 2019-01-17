Analytics Zoo provides a pre-defined KNRM model that can be used for text matching (e.g. question answering).
For training, you can use Keras-Style API methods or alternatively feed the model into NNFrames and BigDL Optimizer.
More text matching models will be supported in the future.

---
## **Build a KNRM Model**
Kernel-pooling Neural Ranking Model with RBF kernel. See [here](https://arxiv.org/abs/1706.06613) for more details.

You can call the following API in Scala and Python respectively to create a `KNRM` with *pre-trained GloVe word embeddings*.

**Scala**
```scala
val knrm = KNRM(text1Length, text2Length, embeddingFile, wordIndex = null, trainEmbed = true, kernelNum = 21, sigma = 0.1, exactSigma = 0.001, targetMode = "ranking")
```

* `text1Length`: Sequence length of text1 (query).
* `text2Length`: Sequence length of text2 (doc).
* `embeddingFile`: The path to the word embedding file. Currently only *glove.6B.50d.txt, glove.6B.100d.txt, glove.6B.200d.txt, glove.6B.300d.txt, glove.42B.300d.txt, glove.840B.300d.txt* are supported. You can download from [here](https://nlp.stanford.edu/projects/glove/).
* `wordIndex`: Map of word (String) and its corresponding index (integer). The index is supposed to __start from 1__ with 0 reserved for unknown words. During the prediction, if you have words that are not in the wordIndex for the training, you can map them to index 0. Default is null. In this case, all the words in the embeddingFile will be taken into account and you can call `WordEmbedding.getWordIndex(embeddingFile)` to retrieve the map.
* `trainEmbed`: Boolean. Whether to train the embedding layer or not. Default is true.
* `kernelNum`: Integer > 1. The number of kernels to use. Default is 21.
* `sigma`: Double. Defines the kernel width, or the range of its softTF count. Default is 0.1.
* `exactSigma`: Double. The sigma used for the kernel that harvests exact matches in the case where RBF mu=1.0. Default is 0.001.
* `targetMode`: String. The target mode of the model. Either 'ranking' or 'classification'. For ranking, the output will be the relevance score between text1 and text2 and you are recommended to use 'rank_hinge' as loss for pairwise training.
For classification, the last layer will be sigmoid and the output will be the probability between 0 and 1 indicating whether text1 is related to text2 and
you are recommended to use 'binary_crossentropy' as loss for binary classification. Default mode is 'ranking'.

See [here](https://github.com/intel-analytics/analytics-zoo/tree/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/qaranker) for the Scala example that trains a KNRM model on WikiQA dataset.


**Python**
```python
knrm = KNRM(text1_length, text2_length, embedding_file, word_index=None, train_embed=True, kernel_num=21, sigma=0.1, exact_sigma=0.001, target_mode="ranking")
```

* `text1_length`: Sequence length of text1 (query).
* `text2_length`: Sequence length of text2 (doc).
* `embedding_file`: The path to the word embedding file. Currently only *glove.6B.50d.txt, glove.6B.100d.txt, glove.6B.200d.txt, glove.6B.300d.txt, glove.42B.300d.txt, glove.840B.300d.txt* are supported. You can download from [here](https://nlp.stanford.edu/projects/glove/).
* `word_index`: Dictionary of word (string) and its corresponding index (int). The index is supposed to __start from 1__ with 0 reserved for unknown words. During the prediction, if you have words that are not in the wordIndex for the training, you can map them to index 0. Default is None. In this case, all the words in the embedding_file will be taken into account and you can call `WordEmbedding.get_word_index(embedding_file)` to retrieve the dictionary.
* `train_embed`: Boolean. Whether to train the embedding layer or not. Default is True.
* `kernel_num`: Int > 1. The number of kernels to use. Default is 21.
* `sigma`: Float. Defines the kernel width, or the range of its softTF count. Default is 0.1.
* `exact_sigma`: Float. The sigma used for the kernel that harvests exact matches in the case where RBF mu=1.0. Default is 0.001.
* `target_mode`: String. The target mode of the model. Either 'ranking' or 'classification'. For ranking, the output will be the relevance score between text1 and text2 and you are recommended to use 'rank_hinge' as loss for pairwise training.
For classification, the last layer will be sigmoid and the output will be the probability between 0 and 1 indicating whether text1 is related to text2 and
you are recommended to use 'binary_crossentropy' as loss for binary classification. Default mode is 'ranking'.

See [here](https://github.com/intel-analytics/analytics-zoo/tree/master/pyzoo/zoo/examples/qaranker) for the Python example that trains a KNRM model on WikiQA dataset.

---
## **Save Model**
After building and training a KNRM model, you can save it for future use.

**Scala**
```scala
knrm.saveModel(path, weightPath = null, overWrite = false)
```

* `path`: The path to save the model. Local file system, HDFS and Amazon S3 are supported. HDFS path should be like "hdfs://[host]:[port]/xxx". Amazon S3 path should be like "s3a://bucket/xxx".
* `weightPath`: The path to save weights. Default is null.
* `overWrite`: Whether to overwrite the file if it already exists. Default is false.

**Python**
```python
knrm.save_model(path, weight_path=None, over_write=False)
```

* `path`: The path to save the model. Local file system, HDFS and Amazon S3 are supported. HDFS path should be like 'hdfs://[host]:[port]/xxx'. Amazon S3 path should be like 's3a://bucket/xxx'.
* `weight_path`: The path to save weights. Default is None.
* `over_write`: Whether to overwrite the file if it already exists. Default is False.

---
## **Load Model**
To load a KNRM model (with weights) saved [above](#save-model):

**Scala**
```scala
KNRM.loadModel(path, weightPath = null)
```

* `path`: The path for the pre-defined model. Local file system, HDFS and Amazon S3 are supported. HDFS path should be like "hdfs://[host]:[port]/xxx". Amazon S3 path should be like "s3a://bucket/xxx".
* `weightPath`: The path for pre-trained weights if any. Default is null.

**Python**
```python
KNRM.load_model(path, weight_path=None)
```

* `path`: The path for the pre-defined model. Local file system, HDFS and Amazon S3 are supported. HDFS path should be like 'hdfs://[host]:[port]/xxx'. Amazon S3 path should be like 's3a://bucket/xxx'.
* `weight_path`: The path for pre-trained weights if any. Default is None.
