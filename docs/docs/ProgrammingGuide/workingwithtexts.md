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
## **Build Text Transformation Pipeline**
You can easily call transformation methods of a TextSet one by one to build the text transformation pipeline. Please refer to [here](../APIGuide/FeatureEngineering/text/#textset-transformations) for more details.

**Scala Example**
```scala
transformedTextSet = textSet.tokenize().normalize().word2idx().shapeSequence(len).generateSample()
```

**Python Example**
```python
transformed_text_set = text_set.tokenize().normalize().word2idx().shape_sequence(len).generate_sample()
```


---
## **Text Training**
After doing text transformation, you can directly feed the transformed TextSet into the model for training.

**Scala**
```scala
model.fit(transformedTextSet, batchSize, nbEpoch)
```

**Python**
```python
model.fit(transformed_text_set, batch_size, nb_epoch)
```


---
## **Text Prediction**
You can also directly input the transformed TextSet into the model for prediction and the prediction result will be stored in each TextFeature.

**Scala**
```scala
predictionTextSet = model.predict(transformedTextSet)
```

**Python**
```python
prediction_text_set = model.predict(transformed_text_set)
```


---
## **Examples**
You can refer to our TextClassification example for TextSet transformation, training and inference.

See [here](https://github.com/intel-analytics/analytics-zoo/tree/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/textclassification) for the Scala example.

See [here](https://github.com/intel-analytics/analytics-zoo/tree/master/pyzoo/zoo/examples/textclassification) for the Python example.
