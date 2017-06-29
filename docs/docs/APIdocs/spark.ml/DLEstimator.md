## DLEstimator ##

**Scala:**
```scala
val estimator = new DLEstimator[Float](
    val model: Module[Float],
    val criterion : Criterion[Float],
    val featureSize : Array[Int],
    val labelSize : Array[Int]
  )
```

[[DLEstimator]] helps to train a BigDL Model with the Spark ML Estimator/Transfomer pattern,
thus Spark users can conveniently fit BigDL into Spark ML pipeline.

[[DLEstimator]] supports feature and label data in the format of Array[Double], Array[Float],
org.apache.spark.mllib.linalg.{Vector, VectorUDT} for Spark 1.5, 1.6 and
org.apache.spark.ml.linalg.{Vector, VectorUDT} for Spark 2.0+. Also label data can be of
DoubleType.
User should specify the feature data dimensions and label data dimensions via the constructor
parameters featureSize and labelSize respectively. Internally the feature and label data are
converted to BigDL tensors, to further train a BigDL model efficiently.


**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import org.apache.spark.ml.DLEstimator

val model = new Sequential[Float]().add(Linear[Float](2, 2)).add(LogSoftMax[Float])
val criterion = MultiLabelSoftMarginCriterion[Float]()
val estimator = new DLEstimator[Float](model, criterion, Array(2), Array(2))
  .setBatchSize(4)
  .setMaxEpoch(10)
val data = sc.parallelize(Seq(
  (Array(0.0, 1.0), Array(1.0, 0.0)),
  (Array(1.0, 0.0), Array(0.0, 1.0)),
  (Array(0.0, 1.0), Array(1.0, 0.0)),
  (Array(1.0, 0.0), Array(0.0, 1.0))))
val df: DataFrame = sqlContext.createDataFrame(data).toDF("features", "label")

val dlModel = estimator.fit(df)
dlModel.transform(df).show(false)
```
