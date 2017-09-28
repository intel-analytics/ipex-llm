## DLEstimator ##

**Scala:**

```scala
val estimator = new DLEstimator(model: Module[T], criterion: Criterion[T], val featureSize: Array[Int], val labelSize: Array[Int])
```

**Python:**

```python
estimator = DLEstimator(model, criterion, feature_size, label_size)
```
Source code [here](https://github.com/intel-analytics/BigDL/blob/585818a3fb0e7339eb4e3831f08da82b7d5e47ba/spark/dl/src/main/scala/org/apache/spark/ml/DLEstimator.scala#L53).
In machine learning context, "Estimation" is regarded as a mechanism for choosing the relatively "best model" according to the observation of reality. The API design of the `Estimator` abstracts the concept of a learning algorithm or any algorithm that fits or trains on data to make the training pipeline more clear and convinient for our users. Technically, an Estimator takes the user-presumed model and criterion with specification on feature and label dimension to prepare for the training. Within its class definition, it implements a method `fit()`, which accepts dataset to start training and produce a optimized DLModel with more accurate prediction.

DLEstimator extends `org.apache.spark.ml.Estimator` and supports model training from Apache Spark DataFrame/Dataset. Different from many algorithms in Spark MLlib, DLEstimator supports more data types for the label column. In many deep learning applications, the label data could be a sequence or other data collection. DLEstimator supports feature and label data in the format of `Array[Double]`, `Array[Float]`, `org.apache.spark.mllib.linalg.Vector` (for Apache Spark 1.5, 1.6) and `org.apache.spark.ml.linalg.Vector` (for Apache Spark 2.0+). Also label data can be of `Double` type. User should specify the feature data dimensions and label data dimensions via the constructor parameters featureSize and labelSize respectively. Internally the feature and label data are converted to BigDL tensors, to further train a BigDL model efficiently.

* `model` BigDL module to be optimized in the fit() method
* `criterion` the criterion used to compute the loss and the gradient
* `featureSize` The size (Tensor dimensions) of the feature data. 
* `labelSize` The size (Tensor dimensions) of the label data



**Scala Example:**
```scala
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
**Python Example:**
```python
```

Output is

|features  |label     |prediction                             |
|----------|----------|---------------------------------------|
|[2.0, 1.0]|[1.0, 2.0]|[1.0034767389297485, 2.006068706512451]|
|[1.0, 2.0]|[2.0, 1.0]|[2.006953001022339, 1.0039551258087158]|
|[2.0, 1.0]|[1.0, 2.0]|[1.0034767389297485, 2.006068706512451]|
|[1.0, 2.0]|[2.0, 1.0]|[2.006953001022339, 1.0039551258087158]|
&nbsp;

---
## **DLClassifier**
Source code [here](https://github.com/intel-analytics/BigDL/blob/585818a3fb0e7339eb4e3831f08da82b7d5e47ba/spark/dl/src/main/scala/org/apache/spark/ml/DLClassifier.scala#L36).
`DLClassifier` is a specialized `DLEstimator` that simplifies the data format for
classification tasks where the label space is discrete. It only supports label column of DoubleType, and the fitted
`DLClassifierModel` will have the prediction column of DoubleType.

* `model` BigDL module to be optimized in the fit() method
* `criterion` the criterion used to compute the loss and the gradient
* `featureSize` The size (Tensor dimensions) of the feature data. 

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

&nbsp;

---
## DLModel ##
**Scala:**
```scala
val dlModel = new DLModel[T](model: Module[T], featureSize: Array[Int])
```
**Python:**
```python
dl_model = DLModel(model, feature_size)
```
Source code [here](https://github.com/intel-analytics/BigDL/blob/585818a3fb0e7339eb4e3831f08da82b7d5e47ba/spark/dl/src/main/scala/org/apache/spark/ml/DLEstimator.scala#L155).
You can see an internal method named [`wrapBigDLModel`](https://github.com/intel-analytics/BigDL/blob/master/spark/dl/src/main/scala/org/apache/spark/ml/DLEstimator.scala#L139) is called in our [`fit()`](https://github.com/intel-analytics/BigDL/blob/585818a3fb0e7339eb4e3831f08da82b7d5e47ba/spark/spark-version/2.0/src/main/scala/org/apache/spark/ml/DLEstimatorBase.scala#L74) method within the DLEstimatorBase class. It is designed to wrap the BigDL Module into a DLModel which is compatible with both spark 1.5-plus and 2.0 by entending the Spark's ML Transformer. It greatly improves the experience of Spark users because now you can use your BigDL DLModel as one of the transformers in your Spark ML pipeline stages.

DLModel supports feature data in the format of Array[Double], Array[Float], org.apache.spark.mllib.linalg.{Vector, VectorUDT} for Spark 1.5, 1.6 and org.apache.spark.ml.linalg.{Vector, VectorUDT} for Spark 2.0+. Internally [[DLModel]] use features column as storage of the feature data, and create Tensors according to the constructor parameter featureSize.

* `model` fitted BigDL module to use in prediction
* `featureSize` The size (Tensor dimensions) of the feature data. (e.g. an image may be with featureSize = 28 * 28)

&nbsp;

---


## DLClassifierModel ##
Source code [here](https://github.com/intel-analytics/BigDL/blob/585818a3fb0e7339eb4e3831f08da82b7d5e47ba/spark/dl/src/main/scala/org/apache/spark/ml/DLClassifier.scala#L63).
DLClassifierModel extends DLModel, which is a specialized DLModel for classification tasks. The prediction column will have the datatype of Double. 
* `model` fitted BigDL module to use in prediction
* `featureSize` The size (Tensor dimensions) of the feature data. (e.g. an image may be with featureSize = 28 * 28)

