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


## DLClassifierModel ##
Source code [here](https://github.com/intel-analytics/BigDL/blob/585818a3fb0e7339eb4e3831f08da82b7d5e47ba/spark/dl/src/main/scala/org/apache/spark/ml/DLClassifier.scala#L63).

DLClassifierModel extends DLModel, which is a specialized DLModel for classification tasks. The prediction column will have the datatype of Double. 
* `model` fitted BigDL module to use in prediction
* `featureSize` The size (Tensor dimensions) of the feature data. (e.g. an image may be with featureSize = 28 * 28)

