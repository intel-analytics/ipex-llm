Analytics Zoo provides a series of text related APIs for end-to-end text processing pipeline,
including text loading, pre-processing, training and inference, etc.

---
## **TextSet**
`TextSet` is a collection of TextFeatures where each `TextFeature` keeps information of a single text record.

`TextSet` can either be a `DistributedTextSet` consisting of text RDD or a `LocalTextSet` consisting of text array.

---
## **Read texts as TextSet**

#### **Read texts from a directory**
Read texts with labels from a directory.

Under this specified directory path, there are supposed to be several subdirectories, each of which contains a number of text files belonging to this category. 
Each category will be a given a label (starting from 0) according to its position in the ascending order sorted among all subdirectories. 
Each text will be a given a label according to the directory where it is located.

**Scala**
```scala
textSet = TextSet.read(path, sc = null, minPartitions = 1)
```

* `path`: String. Folder path to texts. Local file system and HDFS are supported. If you want to read from HDFS, sc needs to be specified.
* `sc`: An instance of SparkContext. If specified, texts will be read as a DistributedTextSet. 
Default is null and in this case texts will be read as a LocalTextSet. 
* `minPartitions`: Integer. A suggestion value of the minimal partition number for input texts.
Only need to specify this when sc is not null. Default is 1.


**Python**
```python
text_set = TextSet.read(path, sc=None, min_partitions=1)
```

* `path`: String. Folder path to texts. Local file system and HDFS are supported. If you want to read from HDFS, sc needs to be defined.
* `sc`: An instance of SparkContext. If specified, texts will be read as a DistributedTextSet. 
Default is None and in this case texts will be read as a LocalTextSet. 
* `min_partitions`: Int. A suggestion value of the minimal partition number for input texts.
Only need to specify this when sc is not None. Default is 1.


#### **Read texts from csv file**
Read texts with id from csv file.

Each record is supposed to contain id(String) and text(String) in order.

Note that the csv file should be without header.

**Scala**
```scala
textSet = TextSet.readCSV(path, sc = null, minPartitions = 1)
```

* `path`: String. The path to the csv file. Local file system and HDFS are supported. If you want to read from HDFS, sc needs to be specified.
* `sc`: An instance of SparkContext. If specified, texts will be read as a DistributedTextSet. 
Default is null and in this case texts will be read as a LocalTextSet. 
* `minPartitions`: Integer. A suggestion value of the minimal partition number for input texts.
Only need to specify this when sc is not null. Default is 1.

**Python**
```python
text_set = TextSet.read_csv(path, sc=None, min_partitions=1)
```

* `path`: String. The path to the csv file. Local file system and HDFS are supported. If you want to read from HDFS, sc needs to be defined.
* `sc`: An instance of SparkContext. If specified, texts will be read as a DistributedTextSet. 
Default is None and in this case texts will be read as a LocalTextSet. 
* `min_partitions`: Int. A suggestion value of the minimal partition number for input texts.
Only need to specify this when sc is not None. Default is 1.


#### **Read texts from parquet file**
Read texts with id from parquet file with schema id(String) and text(String). Return a DistributedTextSet.

**Scala**
```scala
textSet = TextSet.readParquet(path, sqlContext)
```

* `path`: The path to the parquet file.
* `sqlContext`: An instance of SQLContext.

**Python**
```python
text_set = TextSet.read_parquet(path, sc)
```

* `path`: The path to the parquet file.
* `sc`: An instance of SparkContext.


---
## **TextSet Transformations**
Analytics Zoo provides many transformation methods for a TextSet to form a text preprocessing pipeline, which will return the transformed TextSet that can be directly used for training and inference:

### **Tokenization**
Do tokenization on original text.

**Scala**
```scala
transformedTextSet = textSet.tokenize()
```

**Python**
```python
transformed_text_set = text_set.tokenize()
```


### **Normalization**
Removes all dirty (non English alphabet) characters from tokens and converts words to lower case. 
Need to tokenize first.

**Scala**
```scala
transformedTextSet = textSet.normalize()
```

**Python**
```python
transformed_text_set = text_set.normalize()
```


### **Word To Index**
Map word tokens to indices. 
Result index will start from 1 and corresponds to the occurrence frequency of each word sorted in descending order. 
Here we adopt the convention that index 0 will be reserved for unknown words.
Need to tokenize first.

After word2idx, you can get the generated wordIndex map by calling ```getWordIndex``` (Scala) or ```get_word_index()``` (Python) of the transformed TextSet.

**Scala**
```scala
transformedTextSet = textSet.word2idx(removeTopN = 0, maxWordsNum = -1, minFreq = 1, existingMap = null)
```

* `removeTopN`: Non-negative integer. Remove the topN words with highest frequencies in the case where those are treated as stopwords. Default is 0, namely remove nothing.
* `maxWordsNum`: Integer. The maximum number of words to be taken into consideration. Default is -1, namely all words will be considered. Otherwise, it should be a positive integer.
* `minFreq`: Positive integer. Only those words with frequency >= minFreq will be taken into consideration. Default is 1, namely all words that occur will be considered.
* `existingMap`: Existing map of word index if any. Default is null and in this case a new map with index starting from 1 will be generated. 
If not null, then the generated map will preserve the word index in existingMap and assign subsequent indices to new words.


**Python**
```python
transformed_text_set = text_set.word2idx(remove_topN=0, max_words_num=-1, min_freq=1, existing_map=None)
```

* `remove_topN`: Non-negative int. Remove the topN words with highest frequencies in the case where those are treated as stopwords. Default is 0, namely remove nothing.
* `max_words_num`: Int. The maximum number of words to be taken into consideration. Default is -1, namely all words will be considered. Otherwise, it should be a positive int.
* `min_freq`: Positive int. Only those words with frequency >= min_freq will be taken into consideration. Default is 1, namely all words that occur will be considered.
* `existing_map`: Existing dictionary of word index if any. Default is None and in this case a new map with index starting from 1 will be generated. 
If not None, then the generated map will preserve the word index in existing_map and assign subsequent indices to new words.


### **Sequence Shaping**
Shape the sequence of indices to a fixed length. 
Need to word2idx first.

**Scala**
```scala
transformedTextSet = textSet.shapeSequence(len, truncMode = TruncMode.pre, padElement = 0)
```

* `len`: Positive integer. The target length.
* `truncMode`: Truncation mode if the original sequence is longer than the target length. Either 'TruncMode.pre' or 'TruncMode.post'. 
If 'TruncMode.pre', the sequence will be truncated from the beginning. 
If 'TruncMode.post', the sequence will be truncated from the end. 
Default is 'TruncMode.post'.
* `padElement`: Integer. The index element to be padded to the end of the sequence if the original length is smaller than the target length.
Default is 0 with the convention that we reserve index 0 for unknown words.


**Python**
```python
transformed_text_set = text_set.shape_sequence(len, trunc_mode="pre", pad_element=0)
```

* `len`: Positive int. The target length.
* `truncMode`: String. Truncation mode if the original sequence is longer than the target length. Either 'pre' or 'post'. 
If 'pre', the sequence will be truncated from the beginning. 
If 'post', the sequence will be truncated from the end. 
Default is 'post'.
* `padElement`: Int. The index element to be padded to the end of the sequence if the original length is smaller than the target length.
Default is 0 with the convention that we reserve index 0 for unknown words.


### **BigDL Sample Generation**
Transform indices and label (if any) to a BigDL [Sample](https://bigdl-project.github.io/master/#APIGuide/Data/#sample). 
Need to word2idx first.

**Scala**
```scala
transformedTextSet = textSet.generateSample()
```

**Python**
```python
transformed_text_set = text_set.generate_sample()
```


---
## **WordEmbedding**
This is a special Embedding layer that directly loads pre-trained word vectors as weights, 
which turns non-negative integers (indices) into dense vectors of fixed size.

Currently only GloVe embedding is supported for this layer.

The input of this layer should be 2D.

**Scala**
```scala
embedding = WordEmbedding(embeddingFile, wordIndex = null, trainable = false, inputLength = -1)
```

* `embeddingFile`: The path to the word embedding file. Currently *glove.6B.50d.txt, glove.6B.100d.txt, glove.6B.200d.txt, glove.6B.300d.txt, glove.42B.300d.txt, glove.840B.300d.txt* are supported. You can download from [here](https://nlp.stanford.edu/projects/glove/).
* `wordIndex`: Map of word (String) and its corresponding index (integer). The index is supposed to __start from 1__ with 0 reserved for unknown words. During the prediction, if you have words that are not in the wordIndex for the training, you can map them to index 0. Default is null. In this case, all the words in the embeddingFile will be taken into account and you can call `WordEmbedding.getWordIndex(embeddingFile)` to retrieve the map.
* `trainable`: To configure whether the weights of this layer will be updated or not. Only false is supported for now.
* `inputLength`: Positive integer. The sequence length of each input.


**Python**
```python
embedding = WordEmbedding(embedding_file, word_index=None, trainable=False, input_length=None)
```

* `embedding_file` The path to the word embedding file. Currently *glove.6B.50d.txt, glove.6B.100d.txt, glove.6B.200d.txt, glove.6B.300d.txt, glove.42B.300d.txt, glove.840B.300d.txt* are supported. You can download from [here](https://nlp.stanford.edu/projects/glove/).
* `word_index` Dictionary of word (string) and its corresponding index (int). The index is supposed to __start from 1__ with 0 reserved for unknown words. During the prediction, if you have words that are not in the wordIndex for the training, you can map them to index 0. Default is None. In this case, all the words in the embedding_file will be taken into account and you can call `WordEmbedding.get_word_index(embedding_file)` to retrieve the dictionary.
* `trainable`: To configure whether the weights of this layer will be updated or not. Only False is supported for now.
* `inputLength`: Positive int. The sequence length of each input.