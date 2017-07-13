## DLClassifier ##

**Scala:**
```scala
val classifier = new DLClassifier(
    model,
    criterion,
    featureSize
  )
```

`DLClassifier` is a specialized `DLEstimator` that simplifies the data format for
classification tasks. It only supports label column of DoubleType, and the fitted
`DLClassifierModel` will have the prediction column of DoubleType.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import org.apache.spark.ml.DLClassifier

val model = Sequential().add(Linear(2, 2)).add(LogSoftMax())
val criterion = ClassNLLCriterion()
val estimator = new DLClassifier(model, criterion, Array(2))
  .setBatchSize(4)
  .setMaxEpoch(10)
val data = sc.parallelize(Seq(
  (Array(0.0, 1.0), 1.0),
  (Array(1.0, 0.0), 2.0),
  (Array(0.0, 1.0), 1.0),
  (Array(1.0, 0.0), 2.0)))
val df: DataFrame = sqlContext.createDataFrame(data).toDF("features", "label")

val dlModel = estimator.fit(df)
dlModel.transform(df).show(false)
```
Output is

|features  |label|prediction|
|----------|-----|----------|
|[0.0, 1.0]|1.0  |1.0       |
|[1.0, 0.0]|2.0  |2.0       |
|[0.0, 1.0]|1.0  |1.0       |
|[1.0, 0.0]|2.0  |2.0       |