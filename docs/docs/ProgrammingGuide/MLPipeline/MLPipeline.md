
## Use BigDL with Spark ML pipeline ##

BigDL provides `DLEstimator` and `DLClassifier` for users with Spark MLlib experience, which
provides high level API for training a BigDL Model with the Spark ML `Estimator`/`Transfomer`
pattern, thus Spark users can conveniently fit BigDL into Spark ML pipeline.

Currently only scala interface are implemented for `DLEstimator` and `DLClassifier`. Python
support will be added soon.


## DLEstimator ##

`DLEstimator` extends spark `Estimator` and can be used to fit model from DataFrame/Dataset.
Some changes were introduced to better support deep learning applications. `DLEstimator`
supports more data types for the label column. E.g. in many applications, the label
data would be a sequence (text or audio).

`DLEstimator` supports feature and label data in the format of `Array[Double], Array[Float],
org.apache.spark.mllib.linalg.Vector` for Spark 1.5, 1.6 and
`org.apache.spark.ml.linalg.Vector` for Spark 2.0+. Also label data can be of
Double type.

To use `DLEstimator` for training, user should specify
1. the model structure constructed from BigDL layers. You can also use some predefined model
like LetNet or ResNet.
2. the model criterion, which calculates the loss and gradient from model output and label.
3. the feature data dimensions and label data dimensions (the constructor
parameters `featureSize` and `labelSize` respectively). E.g., a sample from
[MNist](http://yann.lecun.com/exdb/mnist/) may have the `featureSize` as Array(28, 28) and
`labelSize` as Array(1). And the feature column contains an array or a `Vector` of 576 numbers.
Internally the feature and label data are converted to BigDL tensors, to further train a
BigDL model efficiently.

The return result of `fit` function in `DLEstimator` is a `DLModel`, which contains the
trained BigDL models and extends Spark `org.apache.spark.ml.Transformer` to be used in prediction.


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


## DLClassifier ##

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


More examples and the full example code can be found from package
com.intel.analytics.bigdl.example.MLPipeline