A Relation represents the relationship between two items.

**Scala/Python**
```scala
relation = Relation(id1, id2, label)
```

* `id1`: String. The id of one item.
* `id2`: String. The id of the other item.
* `label`: Integer. The label between the two items. By convention you can use 0 if they are unrelated and a positive integer if they are related.

A RelationPair is made up of two relations of the same id1, namely:

* Relation(id1, id2Positive, label>0) (A positive Relation)
* Relation(id1, id2Negative, label=0) (A negative Relation)

---
## **Read Relations**
__From csv or txt file__

Each record is supposed to contain id1, id2 and label described above in the exact order.

For csv file, it should be without header.

For txt file, each line should contain one record with fields separated by comma.

**Scala**
```scala
relationsRDD = Relations.read(path, sc, minPartitions = 1)
relationsArray = Relations.read(path)
```

* `path`: The path to the relations file, which can either be a local file path or HDFS path (in this case sc needs to be specified).
* `sc`: An instance of SparkContext. If specified, return RDD of Relation. Otherwise, return array of Relation.
* `minPartitions`: Integer. A suggestion value of the minimal partition number for input
texts. Only takes effect when sc is specified. Default is 1.

**Python**
```python
relations_rdd = Relations.read(path, sc, min_partitions = 1)
relations_list = Relations.read(path)
```

* `path`: The path to the relations file, which can either be a local file path or HDFS path (in this case sc needs to be specified).
* `sc`: An instance of SparkContext. If specified, return RDD of Relation. Otherwise, return list of Relation.
* `min_partitions`: Integer. A suggestion value of the minimal partition number for input
texts. Only takes effect when sc is specified. Default is 1.

__From parquet file__

Read relations from parquet file exactly with the schema in Relation. Return RDD of Relation.

**Scala**
```scala
relationsRDD = Relations.readParquet(path, sqlContext)
```

* `path`: The path to the parquet file.
* `sqlContext`: An instance of SQLContext.

**Python**
```python
relations_rdd = Relations.read_parquet(path, sc)
```

* `path`: The path to the parquet file.
* `sc`: An instance of SparkContext.
