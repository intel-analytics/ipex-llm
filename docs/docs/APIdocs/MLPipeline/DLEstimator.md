## DLEstimator ##

**Scala:**
```scala
val estimator = new DLEstimator(
    model,
    criterion,
    featureSize,
    labelSize
  )
```

`DLEstimator` helps to train a BigDL Model with the Spark ML `Estimator`/`Transfomer` pattern,
thus Spark users can conveniently fit BigDL into Spark ML pipeline.

`DLEstimator` supports feature and label data in the format of `Array[Double], Array[Float],
org.apache.spark.mllib.linalg.{Vector, VectorUDT}` for Spark 1.5, 1.6 and
`org.apache.spark.ml.linalg.{Vector, VectorUDT}` for Spark 2.0+. Also label data can be of
Double type.
User should specify the feature data dimensions and label data dimensions via the constructor
parameters `featureSize` and `labelSize` respectively. E.g., a sample from
[MNist](http://yann.lecun.com/exdb/mnist/) may have the `featureSize` as Array(28, 28) and
`labelSize` as Array(1). And the feature column contains an array
or a vector of 576 numbers. Internally the feature and label data are converted to BigDL
tensors, to further train a BigDL model efficiently.


**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import org.apache.spark.ml.DLEstimator

val model = Sequential().add(Linear(2, 2))
val criterion = MSECriterion()
val estimator = new DLEstimator(model, criterion, Array(2), Array(2))
  .setBatchSize(4)
  .setMaxEpoch(10)
val data = sc.parallelize(Seq(
  (Array(2.0, 1.0), Array(1.0, 2.0)),
  (Array(1.0, 2.0), Array(2.0, 1.0)),
  (Array(2.0, 1.0), Array(1.0, 2.0)),
  (Array(1.0, 2.0), Array(2.0, 1.0))))
val df: DataFrame = sqlContext.createDataFrame(data).toDF("features", "label")

val dlModel = estimator.fit(df)
dlModel.transform(df).show(false)

```
Output is

|features  |label     |prediction                             |
|----------|----------|---------------------------------------|
|[2.0, 1.0]|[1.0, 2.0]|[1.0034767389297485, 2.006068706512451]|
|[1.0, 2.0]|[2.0, 1.0]|[2.006953001022339, 1.0039551258087158]|
|[2.0, 1.0]|[1.0, 2.0]|[1.0034767389297485, 2.006068706512451]|
|[1.0, 2.0]|[2.0, 1.0]|[2.006953001022339, 1.0039551258087158]|


