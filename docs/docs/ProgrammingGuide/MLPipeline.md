
## **Overview**

BigDL provides `DLEstimator` and `DLClassifier` for users with Apache Spark MLlib experience, which
provides high level API for training a BigDL Model with the Apache Spark `Estimator`/`Transfomer`
pattern, thus users can conveniently fit BigDL into a ML pipeline. The fitted model `DLModel` and
`DLClassiferModel` contains the trained BigDL model and extends the Spark ML `Model` class.
Alternatively users may also construct a `DLModel` with a pre-trained BigDL model to use it in
Spark ML Pipeline for prediction.

Currently only scala interface are implemented for `DLEstimator` and `DLClassifier`. Python
support will be added soon.

---
## **DLEstimator**

`DLEstimator` extends `org.apache.spark.ml.Estimator` and supports model training from
Apache Spark DataFrame/Dataset. 
 
Different from many algorithms in Spark MLlib, `DLEstimator` supports more data types for the
label column. In many deep learning applications, the label data could be a sequence
or other data collection. `DLEstimator` supports feature and label data in the format
of `Array[Double]`, `Array[Float]`, `org.apache.spark.mllib.linalg.Vector` (for Apache
Spark 1.5, 1.6) and `org.apache.spark.ml.linalg.Vector` (for Apache Spark 2.0+). Also label
data can be of Double type.

To use `DLEstimator` for training, user should specify

* the model structure constructed from BigDL layers. You can also use some predefined model
like LetNet or ResNet.
* the model criterion, which calculates the loss and gradient from model output and label.
* the feature data dimensions and label data dimensions (the constructor
parameters `featureSize` and `labelSize` respectively). E.g., a sample from
[MNist](http://yann.lecun.com/exdb/mnist/) may have the `featureSize` as Array(28, 28) and
`labelSize` as Array(1). And the feature column contains an array or a `Vector` of 784 (28 * 28)
numbers. Internally the feature and label data are converted to BigDL tensors, to further train
a BigDL model efficiently.

The return result of `fit` function in `DLEstimator` is a `DLModel`, which contains the
trained BigDL models and extends `org.apache.spark.ml.Transformer` to be used in prediction.


**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.Engine
import org.apache.spark.SparkContext
import org.apache.spark.ml.DLEstimator
import org.apache.spark.sql.SQLContext

/**
 *  Multi-label regression with BigDL layers and DLEstimator
 */
object DLEstimatorMultiLabelLR {

  def main(args: Array[String]): Unit = {
    val conf = Engine.createSparkConf()
      .setAppName("DLEstimatorMultiLabelLR")
      .setMaster("local[1]")
    val sc = new SparkContext(conf)
    val sqlContext = SQLContext.getOrCreate(sc)
    Engine.init

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
    val df = sqlContext.createDataFrame(data).toDF("features", "label")
    val dlModel = estimator.fit(df)
    dlModel.transform(df).show(false)
  }
}

```
Output is

|features  |label     |prediction                             |
|----------|----------|---------------------------------------|
|[2.0, 1.0]|[1.0, 2.0]|[1.0034767389297485, 2.006068706512451]|
|[1.0, 2.0]|[2.0, 1.0]|[2.006953001022339, 1.0039551258087158]|
|[2.0, 1.0]|[1.0, 2.0]|[1.0034767389297485, 2.006068706512451]|
|[1.0, 2.0]|[2.0, 1.0]|[2.006953001022339, 1.0039551258087158]|


---
## **DLClassifier**

`DLClassifier` is a specialized `DLEstimator` that simplifies the data format for
classification tasks. It only supports label column of DoubleType, and the fitted
`DLClassifierModel` will have the prediction column of DoubleType.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, Linear, LogSoftMax, Sequential}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.utils.Engine
import org.apache.spark.SparkContext
import org.apache.spark.ml.DLClassifier
import org.apache.spark.sql.SQLContext

/**
 * Logistic Regression with BigDL layers and DLClassifier
 */
object DLClassifierLogisticRegression {

  def main(args: Array[String]): Unit = {
    val conf = Engine.createSparkConf()
      .setAppName("DLClassifierLogisticRegression")
      .setMaster("local[1]")
    val sc = new SparkContext(conf)
    val sqlContext = SQLContext.getOrCreate(sc)
    Engine.init

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
    val df = sqlContext.createDataFrame(data).toDF("features", "label")
    val dlModel = estimator.fit(df)
    dlModel.transform(df).show(false)
  }
}
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
