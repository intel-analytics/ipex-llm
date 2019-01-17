Analytics Zoo provides a pre-defined KNRM model that can be used for text matching (e.g. question answering).
More text matching models will be supported in the future.

**Highlights**

1. Easy-to-use Keras-Style defined model which provides compile and fit methods for training. Alternatively, it could be fed into NNFrames or BigDL Optimizer.
2. The model can be used for both ranking and classification tasks.

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

---
## **Pairwise training**
For ranking, the model can be trained pairwisely with the following steps:

1. Read train relations. See [here](../APIGuide/FeatureEngineering/relation/#read-relations) for more details.
2. Read text1 and text2 corpus as TextSet. See [here](../APIGuide/FeatureEngineering/text/#read-texts-from-csv-file) for more details.
3. Preprocess text1 and text2 corpus. See [here](../APIGuide/FeatureEngineering/text/#textset-transformations) for more details.
4. Generate all relation pairs from train relations. Each pair is made up of a positive relation and a negative one of the same id1.
During the training process, we intend to optimize the margin loss within each pair.
We provide the following API to generate a `TextSet` for pairwise training:

**Scala**
```scala
val trainSet = TextSet.fromRelationPairs(relations, corpus1, corpus2)
```

* `relations`: RDD or array of Relation.
* `corpus1`: TextSet that contains all id1 in relations.
* `corpus2`: TextSet that contains all id2 in relations.
* For corpus1 and corpus2 respectively, each text must have been transformed to indices of the same length by
  calling [tokenize](../APIGuide/FeatureEngineering/text/#tokenization), [word2idx](../APIGuide/FeatureEngineering/text/#word-to-index) 
  and [shapeSequence](../APIGuide/FeatureEngineering/text/#sequence-shaping) in order.
* If relations is an RDD, then corpus1 and corpus2 must both be DistributedTextSet.
If relations is an array, then corpus1 and corpus2 must both be LocalTextSet.

**Python**
```python
train_set = TextSet.from_relation_pairs(relations, corpus1, corpus2)
```

* `relations`: RDD or list of Relation.
* `corpus1`: TextSet that contains all id1 in relations.
* `corpus2`: TextSet that contains all id2 in relations.
* For corpus1 and corpus2 respectively, each text must have been transformed to indices of the same length by
  calling [tokenize](../APIGuide/FeatureEngineering/text/#tokenization), [word2idx](../APIGuide/FeatureEngineering/text/#word-to-index) 
  and [shape_sequence](../APIGuide/FeatureEngineering/text/#sequence-shaping) in order.
* If relations is an RDD, then corpus1 and corpus2 must both be DistributedTextSet.
If relations is a list, then corpus1 and corpus2 must both be LocalTextSet.

Call compile and fit to train the model:

**Scala**
```scala
val model = Sequential().add(TimeDistributed(knrm, inputShape = Shape(2, text1Length + text2Length)))
model.compile(optimizer = new SGD(learningRate), loss = RankHinge())
model.fit(trainSet, batchSize, nbEpoch)
```

**Python**
```python
model = Sequential().add(TimeDistributed(knrm, input_shape=(2, text1Length + text2Length)))
model.compile(optimizer=SGD(learning_rate), loss='rank_hinge')
model.fit(train_set, batch_size, nb_epoch)
```


---
## **Listwise evaluation**
Given text1 and a list of text2 candidates, we provide metrics [NDCG](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Discounted_cumulative_gain) and [MAP](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision) to listwisely evaluate a ranking model with the following steps:

1. Read validation relations. See [here](../APIGuide/FeatureEngineering/relation/#read-relations) for more details.
2. Read text1 and text2 corpus as TextSet. See [here](../APIGuide/FeatureEngineering/text/#read-texts-from-csv-file) for more details.
3. Preprocess text1 and text2 corpus same as the training phase. See [here](../APIGuide/FeatureEngineering/text/#textset-transformations) for more details.
3. Generate all relation lists from validation relations. Each list is made up of one id1 and all id2 combined with id1.
We provide the following API to generate a `TextSet` for listwise evaluation:

**Scala**
```scala
val validateSet = TextSet.fromRelationLists(relations, corpus1, corpus2)
```

* `relations`: RDD or array of Relation.
* `corpus1`: TextSet that contains all id1 in relations.
* `corpus2`: TextSet that contains all id2 in relations.
* For corpus1 and corpus2 respectively, each text must have been transformed to indices of the same length as during the training process by 
calling [tokenize](../APIGuide/FeatureEngineering/text/#tokenization), [word2idx](../APIGuide/FeatureEngineering/text/#word-to-index) 
and [shapeSequence](../APIGuide/FeatureEngineering/text/#sequence-shaping) in order.
* If relations is an RDD, then corpus1 and corpus2 must both be DistributedTextSet.
If relations is an array, then corpus1 and corpus2 must both be LocalTextSet.

**Python**
```python
validate_set = TextSet.from_relation_lists(relations, corpus1, corpus2)
```

* `relations`: RDD or list of Relation.
* `corpus1`: TextSet that contains all id1 in relations.
* `corpus2`: TextSet that contains all id2 in relations.
* For corpus1 and corpus2 respectively, each text must have been transformed to indices of the same length as during the training process by 
calling [tokenize](../APIGuide/FeatureEngineering/text/#tokenization), [word2idx](../APIGuide/FeatureEngineering/text/#word-to-index) 
and [shape_sequence](../APIGuide/FeatureEngineering/text/#sequence-shaping) in order.
* If relations is an RDD, then corpus1 and corpus2 must both be DistributedTextSet.
If relations is a list, then corpus1 and corpus2 must both be LocalTextSet.

Call evaluateNDCG or evaluateMAP to evaluate the model:

**Scala**
```scala
knrm.evaluateNDCG(validateSet, k, threshold = 0.0)
knrm.evaluateMAP(validateSet, threshold = 0.0)
```

**Python**
```python
knrm.evaluate_ndcg(validate_set, k, threshold=0.0)
knrm.evaluate_map(validate_set, threshold=0.0)
```

* `k`: Positive integer. Rank position in NDCG.
* `threshold`: If label > threshold, then it will be considered as a positive record. Default is 0.0.

---
## **Examples**
We provide an example to train and evaluate a KNRM model on WikiQA dataset for ranking.

See [here](https://github.com/intel-analytics/analytics-zoo/tree/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/qaranker) for the Scala example.

See [here](https://github.com/intel-analytics/analytics-zoo/tree/master/pyzoo/zoo/examples/qaranker) for the Python example.