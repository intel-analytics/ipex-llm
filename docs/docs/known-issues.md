* If you encounter the following exception when calling the Python API of Analytics Zoo using Python 3.5 or 3.6:
```
Py4JJavaError: An error occurred while calling z:org.apache.spark.bigdl.api.python.BigDLSerDe.loads.
: net.razorvine.pickle.PickleException: expected zero arguments for construction of ClassDict (for numpy.dtype)
```
you may need to check whether your input argument involves Numpy types (such as `numpy.int64`). See [here](https://issues.apache.org/jira/browse/SPARK-12157) for the related issue.

For example, invoking `np.min`, `np.max`, `np.unique`, etc. will return type `numpy.int64`. One way to solve this is to use `int()` to convert a number of type `numpy.int64` to a Python int.

* If you use two StringIndexs to convert String feature to index using Spark 2.4, it will be very slow in the next `dataframe.rdd.map` opteration. 6,000,000 records cost 12 hours, and 1/3 of the time is GC.  

For example，one of our customer change [NCF recommender](https://github.com/intel-analytics/analytics-zoo/tree/master/apps/recommendation-ncf)'s preprocessing to match their data:
```
df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("xxx.csv") # The header is user, item, label. The type is string, string, int.
user_indexer = StringIndexer(inputCol='user',outputCol='user_index',handleInvalid="skip")
item_indexer = StringIndexer(inputCol='item',outputCol='item_index',handleInvalid="skip")
pipe = Pipeline(stages=[user_indexer, item_indexer])
pipe_fit = pip.fit(df)
df = pipe_fit.transform(df)
train_data = df.select('user_index','item_index','label')
```
Then they use a map to transform this `train_data` to RDD[Sample]:
```
def build_sample(user_id, item_id, rating):
    sample = Sample.from_ndarray(np.array([user_id, item_id]), np.array([rating]))
    return UserItemFeature(user_id, item_id, sample)
pairFeatureRdds = train_data.rdd.map(lambda x: build_sample(x[0], x[1], x[2]-1))
```
If they execute a `pairFeatureRdds.count()`, this counting job will cost 12 hours when the dataset has 6,000,000 records.  
It seems a bug of StringIndexer. But we find a good way to work around this, before transform `user_index`, `item_index` and `label` to `UserItemFeature` we need to cast the Double `index`s to Float, like this:
```
train_data = train_data.withColumn("user_index", df_r_new["user_index"].cast(FloatType()))
train_data = train_data.withColumn("item_index", df_r_new["item_index"].cast(FloatType()))
```
Then the job finish in about 30s, the GC is also disappeared.
