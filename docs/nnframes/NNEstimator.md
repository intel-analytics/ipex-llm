## NNEstimator

**Scala:**

```scala
val estimator = new NNEstimator(model: Module[T], criterion: Criterion[T], val featureSize: Array[Int], val labelSize: Array[Int])
```

**Python:**

```python
estimator = NNEstimator(model, criterion, feature_size, label_size)
```

`NNEstimator` extends BigDL [DLEsitmator](https://bigdl-project.github.io/master/#APIGuide/DLFrames/DLEstimator_DLClassifier/#dlestimator) API
(`org.apache.spark.ml.Estimator`) and supports model training on Apache Spark DataFrame/Dataset.

`NNEstimator` takes the user-specified [Model](https://bigdl-project.github.io/master/#APIGuide/Layers/Containers) and
[criterion](https://bigdl-project.github.io/master/#APIGuide/Losses/) with specification on feature and label dimension to prepare for the training. Different from Estimators in Spark ML, `DLEstimator` supports more data
types for the feature/label column. In many deep learning applications, the feature/label data could be a sequence
or other data collection. DLEstimator supports feature and label data in the format of `Array[Double]`,
`Array[Float]`, `org.apache.spark.mllib.linalg.Vector`, `org.apache.spark.ml.linalg.Vector` and image schema
as specified in [ImagePreprocessing](./ImagesProcessing.md) .

User should specify the feature data dimensions and label data dimensions via the constructor parameters
featureSize and labelSize respectively. Internally the feature and label data are converted to BigDL
tensors, to further train a BigDL model efficiently. E.g. an image record should have the feature size as
Array(3, 224, 224), representing the number of channels, width and height.

constructor parameters
* `model` BigDL module to be optimized in the fit() method
* `criterion` the criterion used to compute the loss and the gradient
* `featureSize` The size (Tensor dimensions) of the feature data.
* `labelSize` The size (Tensor dimensions) of the label data

**Scala Example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim.LBFGS
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericDouble

import com.intel.analytics.zoo.pipeline.nnframes.NNEstimator

val model = Sequential().add(Linear(2, 2))
val criterion = MSECriterion()
val estimator = new NNEstimator(model, criterion, Array(2), Array(2))
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
```

**Python Example:**
```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.util.common import *
from bigdl.dlframes.dl_classifier import *
from pyspark.sql.types import *

model = Sequential().add(Linear(2, 2))
criterion = MSECriterion()
estimator = NNEstimator(model, criterion, [2], [2]).setBatchSize(4).setMaxEpoch(10)
data = sc.parallelize([
    ((2.0, 1.0), (1.0, 2.0)),
    ((1.0, 2.0), (2.0, 1.0)),
    ((2.0, 1.0), (1.0, 2.0)),
    ((1.0, 2.0), (2.0, 1.0))])

schema = StructType([
    StructField("features", ArrayType(DoubleType(), False), False),
    StructField("label", ArrayType(DoubleType(), False), False)])
df = sqlContext.createDataFrame(data, schema)
dlModel = estimator.fit(df)
dlModel.transform(df).show(False)
```
---


## NNModel
**Scala:**
```scala
val nnModel = new NNModel[T](model: Module[T], featureSize: Array[Int])
```

**Python:**
```python
nn_model = NNModel(model, feature_size)
```

`NNModel` is designed to wrap the BigDL Module as a Spark's ML [Transformer](https://spark.apache.org/docs/2.1.1/ml-pipeline.html#transformers) which is compatible
with both spark 1.5-plus and 2.0. It greatly improves the
experience of Spark users because now you can **wrap a pre-trained BigDL Model into a NNModel,
and use it as a transformer in your Spark ML pipeline to predict the results**.

`NNModel` supports feature data in the format of
`Array[Double], Array[Float], org.apache.spark.mllib.linalg.{Vector, VectorUDT},
org.apache.spark.ml.linalg.{Vector, VectorUDT}` and image schema. Internally `DLModel` use
features column as storage of the feature data, and create Tensors according to the constructor
parameter featureSize.

* `model` fitted BigDL module to use in prediction
* `featureSize` The size (Tensor dimensions) of the feature data.
(e.g. an image may be with featureSize = 28 * 28)

---

## NNClassifier
**Scala:**
```scala
val classifer = new NNClassifer(model: Module[T], criterion: Criterion[T], val featureSize: Array[Int])
```

**Python:**
```python
classifier = NNClassifer(model, criterion, feature_size)
```

`NNClassifier` is a specialized `NNEstimator` that simplifies the data format for
classification tasks where the label space is discrete. It only supports label column of DoubleType or FloatType,
and the fitted `NNClassifierModel` will have the prediction column of DoubleType.

* `model` BigDL module to be optimized in the fit() method
* `criterion` the criterion used to compute the loss and the gradient
* `featureSize` The size (Tensor dimensions) of the feature data.

**Scala example:**
```scala
import com.intel.analytics.zoo.pipeline.nnframes.NNClassifier
import com.intel.analytics.bigdl.nn.{ClassNLLCriterion, Linear, LogSoftMax, Sequential}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val data = sc.parallelize(Seq(
      (Array(0.0, 1.0), 1.0),
      (Array(1.0, 0.0), 2.0),
      (Array(0.0, 1.0), 1.0),
      (Array(1.0, 0.0), 2.0)))
val df = sqlContext.createDataFrame(data).toDF("features", "label")
val model = Sequential().add(Linear(2, 2)).add(LogSoftMax())
val criterion = ClassNLLCriterion()
val estimator = new NNClassifier(model, criterion, Array(2))
  .setBatchSize(4)
  .setMaxEpoch(10)

val dlModel = estimator.fit(df)
dlModel.transform(df).show(false)
```

**Python Example:**

```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.util.common import *
from bigdl.dlframes.dl_classifier import *
from pyspark.sql.types import *

#Logistic Regression with BigDL layers and Analytics zoo NNClassifier
model = Sequential().add(Linear(2, 2)).add(LogSoftMax())
criterion = ClassNLLCriterion()
estimator = NNClassifier(model, criterion, [2]).setBatchSize(4).setMaxEpoch(10)
data = sc.parallelize([
    ((0.0, 1.0), [1.0]),
    ((1.0, 0.0), [2.0]),
    ((0.0, 1.0), [1.0]),
    ((1.0, 0.0), [2.0])])

schema = StructType([
    StructField("features", ArrayType(DoubleType(), False), False),
    StructField("label", ArrayType(DoubleType(), False), False)])
df = sqlContext.createDataFrame(data, schema)
dlModel = estimator.fit(df)
dlModel.transform(df).show(False)
```

## NNClassifierModel ##

**Scala:**
```scala
val dlClassifierModel = new NNClassifierModel[T](model: Module[T], featureSize: Array[Int])
```

**Python:**
```python
dl_classifier_model = NNClassifierModel(model, feature_size)
```

NNClassifierModel extends DLModel, which is a specialized DLModel for classification tasks.
The prediction column will have the datatype of Double.

* `model` fitted BigDL module to use in prediction
* `featureSize` The size (Tensor dimensions) of the feature data. (e.g. an image may be with
featureSize = 28 * 28)
---

## Hyperparameter setting

Prior to the commencement of the training process, you can modify the batch size, the epoch number of your
training, and learning rate to meet your goal or NNEstimator/NNClassifier will use the default value.

Continue the codes above, NNEstimator and NNClassifier can be setted in the same way.

**Scala:**

```scala
//for esitmator
estimator.setBatchSize(4).setMaxEpoch(10).setLearningRate(0.01)
//for classifier
classifier.setBatchSize(4).setMaxEpoch(10).setLearningRate(0.01)
```
**Python:**

```python
# for esitmator
estimator.setBatchSize(4).setMaxEpoch(10).setLearningRate(0.01)
# for classifier
classifier.setBatchSize(4).setMaxEpoch(10).setLearningRate(0.01)

```

## Prepare the data and start the training process

Users need to convert the data into Spark's
[DataFrame/DataSet](https://spark.apache.org/docs/latest/sql-programming-guide.html#datasets-and-dataframes)
to feed to the NNEstimator/NNCLassifer.
Then after these steps, we can start training now.

Suppose `df` is the training data, simple call `fit` method and let Analytics Zoo train the model for you. You will
get a NNClassifierModel if you use NNClassifier.

**Scala:**

```scala
//get a NNClassifierModel
val nnClassifierModel = classifier.fit(df)
```

**Python:**

```python
# get a NNClassifierModel
nnClassifierModel = classifier.fit(df)
```
## Make prediction on chosen data by using NNClassifierModel

Since NNClassifierModel inherits from Spark's Transformer abstract class, simply call `transform`
 method on NNClassifierModel to make prediction.

**Scala:**

```scala
nnModel.transform(df).show(false)
```

**Python:**

```python
nnModel.transform(df).show(false)
```

For the complete examples of NNFrames, please refer to:
[Scala examples](https://github.com/intel-analytics/zoo/tree/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/nnframes/)
[Python examples]()