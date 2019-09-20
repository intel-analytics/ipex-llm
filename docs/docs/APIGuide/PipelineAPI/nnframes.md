## NNEstimator

**Scala:**

```scala
val estimator = NNEstimator(model, criterion)
```

**Python:**

```python
estimator = NNEstimator(model, criterion)
```

`NNEstimator` extends `org.apache.spark.ml.Estimator` and supports training a BigDL
model with Spark DataFrame data. It can be integrated into a standard Spark ML Pipeline
to allow users to combine the components of BigDL and Spark MLlib.

`NNEstimator` supports different feature and label data types through `Preprocessing`.
During fit (training), NNEstimator will extract feature and label data from input DataFrame and use
the `Preprocessing` to convert data for the model, typically converts the feature and label
to Tensors or converts the (feature, option[Label]) tuple to a BigDL `Sample`. 

Each`Preprocessing` conducts a data conversion step in the preprocessing phase, multiple
`Preprocessing` can be combined into a `ChainedPreprocessing`. Some pre-defined 
`Preprocessing` for popular data types like Image, Array or Vector are provided in package
`com.intel.analytics.zoo.feature`, while user can also develop customized `Preprocessing`.

NNEstimator and NNClassifier also supports setting the caching level for the training data.
Options are "DRAM", "PMEM" or "DISK_AND_DRAM". If DISK_AND_DRAM(numSlice) is used, only 1/numSlice
data will be loaded into memory during training time. By default, DRAM mode is used and all data
are cached in memory.

By default, `SeqToTensor` is used to convert an array or Vector to a 1-dimension Tensor.
Using the `Preprocessing` allows `NNEstimator` to cache only the raw data and decrease the 
memory consumption during feature conversion and training, it also enables the model to digest
extra data types that DataFrame does not support currently.

More concrete examples are available in package `com.intel.analytics.zoo.examples.nnframes`

`NNEstimator` can be created with various parameters for different scenarios.

**1.** `NNEstimator(model, criterion)`

   Takes only model and criterion and use `SeqToTensor` as feature and label
   `Preprocessing`. `NNEstimator` will extract the data from feature and label columns (
   only Scalar, Array[_] or Vector data type are supported) and convert each feature/label to
   1-dimension Tensor. The tensors will be combined into BigDL `Sample` and send to model for
   training.

**2.** `NNEstimator(model, criterion, featureSize: Array[Int], labelSize: Array[Int])`

   Takes model, criterion, featureSize(Array of Int) and labelSize(Array of Int). `NNEstimator`
   will extract the data from feature and label columns (only Scalar, Array[_] or Vector data
   type are supported) and convert each feature/label to Tensor according to the specified Tensor
   size.

**3.** `NNEstimator(model, criterion, featureSize: Array[Array[Int]], labelSize: Array[Int])`

   This is the interface for multi-input model. It takes model, criterion, featureSize(Array of
   Int Array) and labelSize(Array of Int). `NNEstimator`
   will extract the data from feature and label columns (only Scalar, Array[_] or Vector data
   type are supported) and convert each feature/label to Tensor according to the specified Tensor
   size.

**4.** `NNEstimator(model, criterion, featurePreprocessing: Preprocessing[F, Tensor[T]],
labelPreprocessing: Preprocessing[F, Tensor[T]])`

   Takes model, criterion, featurePreprocessing and labelPreprocessing.  `NNEstimator`
   will extract the data from feature and label columns and convert each feature/label to Tensor
   with the featurePreprocessing and labelPreprocessing. This constructor provides more flexibility
   in supporting extra data types.

Meanwhile, for advanced use cases (e.g. model with multiple input tensor), `NNEstimator` supports:
`setSamplePreprocessing(value: Preprocessing[(Any, Option[Any]), Sample[T]])` to directly compose
Sample according to user-specified Preprocessing.


**Scala Example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.zoo.pipeline.nnframes.NNEstimator
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val model = Sequential().add(Linear(2, 2))
val criterion = MSECriterion()
val estimator = NNEstimator(model, criterion)
  .setLearningRate(0.2)
  .setMaxEpoch(40)
val data = sc.parallelize(Seq(
  (Array(2.0, 1.0), Array(1.0, 2.0)),
  (Array(1.0, 2.0), Array(2.0, 1.0)),
  (Array(2.0, 1.0), Array(1.0, 2.0)),
  (Array(1.0, 2.0), Array(2.0, 1.0))))
val df = sqlContext.createDataFrame(data).toDF("features", "label")
val nnModel = estimator.fit(df)
nnModel.transform(df).show(false)
```

**Python Example:**
```python
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.util.common import *
from zoo.pipeline.nnframes.nn_classifier import *
from zoo.feature.common import *

data = self.sc.parallelize([
    ((2.0, 1.0), (1.0, 2.0)),
    ((1.0, 2.0), (2.0, 1.0)),
    ((2.0, 1.0), (1.0, 2.0)),
    ((1.0, 2.0), (2.0, 1.0))])

schema = StructType([
    StructField("features", ArrayType(DoubleType(), False), False),
    StructField("label", ArrayType(DoubleType(), False), False)])
df = self.sqlContext.createDataFrame(data, schema)
model = Sequential().add(Linear(2, 2))
criterion = MSECriterion()
estimator = NNEstimator(model, criterion, SeqToTensor([2]), ArrayToTensor([2]))\
    .setBatchSize(4).setLearningRate(0.2).setMaxEpoch(40) \
nnModel = estimator.fit(df)
res = nnModel.transform(df)
```

***Example with multi-inputs Model.***
This example trains a model with 3 inputs. And users can
use VectorAssembler from Spark MLlib to combine different fields. With the specified sizes for
each model input, NNEstiamtor and NNClassifer will split the input features data and send
tensors to corresponding inputs.
```python
sparkConf = init_spark_conf().setAppName("testNNClassifer").setMaster('local[1]')
sc = init_nncontext(sparkConf)
spark = SparkSession\
    .builder\
    .getOrCreate()

df = spark.createDataFrame(
    [(1, 35, 109.0, Vectors.dense([2.0, 5.0, 0.5, 0.5]), 1.0),
     (2, 58, 2998.0, Vectors.dense([4.0, 10.0, 0.5, 0.5]), 2.0),
     (3, 18, 123.0, Vectors.dense([3.0, 15.0, 0.5, 0.5]), 1.0)],
    ["user", "age", "income", "history", "label"])

assembler = VectorAssembler(
    inputCols=["user", "age", "income", "history"],
    outputCol="features")

df = assembler.transform(df)

x1 = ZLayer.Input(shape=(1,))
x2 = ZLayer.Input(shape=(2,))
x3 = ZLayer.Input(shape=(2, 2,))

user_embedding = ZLayer.Embedding(5, 10)(x1)
flatten = ZLayer.Flatten()(user_embedding)
dense1 = ZLayer.Dense(2)(x2)
gru = ZLayer.LSTM(4, input_shape=(2, 2))(x3)

merged = ZLayer.merge([flatten, dense1, gru], mode="concat")
zy = ZLayer.Dense(2)(merged)

zmodel = ZModel([x1, x2, x3], zy)
criterion = ClassNLLCriterion()
classifier = NNClassifier(zmodel, criterion, [[1], [2], [2, 2]]) \
    .setOptimMethod(Adam()) \
    .setLearningRate(0.1)\
    .setBatchSize(2) \
    .setMaxEpoch(10)

nnClassifierModel = classifier.fit(df)
print(nnClassifierModel.getBatchSize())
res = nnClassifierModel.transform(df).collect()

```

---

## NNModel
**Scala:**
```scala
val nnModel = NNModel(bigDLModel)
```

**Python:**
```python
nn_model = NNModel(bigDLModel)
```

`NNModel` extends Spark's ML
[Transformer](https://spark.apache.org/docs/2.1.1/ml-pipeline.html#transformers). User can invoke
`fit` in `NNEstimator` to get a `NNModel`, or directly compose a `NNModel` from BigDLModel.
It enables users to wrap a pre-trained BigDL Model into a NNModel,
and use it as a transformer in your Spark ML pipeline to predict the results for `DataFrame
(DataSet)`. 

`NNModel` can be created with various parameters for different scenarios.

**1.** `NNModel(model)`

   Takes only model and use `SeqToTensor` as feature Preprocessing. `NNModel` will extract the
   data from feature column (only Scalar, Array[_] or Vector data type are supported) and
   convert each feature to 1-dimension Tensor. The tensors will be sent to model for inference.

**2.** `NNModel(model, featureSize: Array[Int])`

   Takes model and featureSize(Array of Int). `NNModel` will extract the data from feature
   column (only Scalar, Array[_] or Vector data type are supported) and convert each feature
   to Tensor according to the specified Tensor size. User can also set featureSize as
   Array[Array[Int]] for multi-inputs model.

**3.** `NNModel(model, featurePreprocessing: Preprocessing[F, Tensor[T]])`

   Takes model and featurePreprocessing. `NNModel` will extract the data from feature column
   and convert each feature to Tensor with the featurePreprocessing. This constructor provides
   more flexibility in supporting extra data types.

Meanwhile, for advanced use cases (e.g. model with multiple input tensor), `NNModel` supports:
`setSamplePreprocessing(value: Preprocessing[Any, Sample[T]])`to directly compose
Sample according to user-specified Preprocessing.

---

## NNClassifier
**Scala:**
```scala
val classifer =  NNClassifer(model, criterion)
```

**Python:**
```python
classifier = NNClassifer(model, criterion)
```

`NNClassifier` is a specialized `NNEstimator` that simplifies the data format for
classification tasks where the label space is discrete. It only supports label column of
DoubleType, and the fitted `NNClassifierModel` will have the prediction column of 
DoubleType.

* `model` BigDL module to be optimized in the fit() method
* `criterion` the criterion used to compute the loss and the gradient

`NNClassifier` can be created with various parameters for different scenarios.

**1.** `NNClassifier(model, criterion)`

   Takes only model and criterion and use `SeqToTensor` as feature and label
   Preprocessing. `NNClassifier` will extract the data from feature and label columns (
   only Scalar, Array[_] or Vector data type are supported) and convert each feature/label to
   1-dimension Tensor. The tensors will be combined into BigDL samples and send to model for
   training.

**2.** `NNClassifier(model, criterion, featureSize: Array[Int])`

   Takes model, criterion, featureSize(Array of Int). `NNClassifier`
   will extract the data from feature and label columns and convert each feature to Tensor
   according to the specified Tensor size. `ScalarToTensor` is used to convert the label column.
   User can also set featureSize as Array[Array[Int]] for multi-inputs model.

**3.** `NNClassifier(model, criterion, featurePreprocessing: Preprocessing[F, Tensor[T]])`

   Takes model, criterion and featurePreprocessing.  `NNClassifier`
   will extract the data from feature and label columns and convert each feature to Tensor
   with the featurePreprocessing. This constructor provides more flexibility
   in supporting extra data types.

Meanwhile, for advanced use cases (e.g. model with multiple input tensor), `NNClassifier` supports:
`setSamplePreprocessing(value: Preprocessing[(Any, Option[Any]), Sample[T]])` to directly compose
Sample with user-specified Preprocessing.

**Scala example:**
```scala
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.zoo.pipeline.nnframes.NNClassifier
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

val model = Sequential().add(Linear(2, 2))
val criterion = MSECriterion()
val estimator = NNClassifier(model, criterion)
  .setLearningRate(0.2)
  .setMaxEpoch(40)
val data = sc.parallelize(Seq(
  (Array(0.0, 1.0), 1.0),
  (Array(1.0, 0.0), 2.0),
  (Array(0.0, 1.0), 1.0),
  (Array(1.0, 0.0), 2.0)))
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
val nnClassifierModel = NNClassifierModel(model, featureSize)
```

**Python:**
```python
nn_classifier_model = NNClassifierModel(model)
```

NNClassifierModel is a specialized `NNModel` for classification tasks.
Both label and prediction column will have the datatype of Double.

`NNClassifierModel` can be created with various parameters for different scenarios.

**1.** `NNClassifierModel(model)`

   Takes only model and use `SeqToTensor` as feature Preprocessing. `NNClassifierModel` will
   extract the data from feature column (only Scalar, Array[_] or Vector data type are supported)
   and convert each feature to 1-dimension Tensor. The tensors will be sent to model for inference.

**2.** `NNClassifierModel(model, featureSize: Array[Int])`

   Takes model and featureSize(Array of Int). `NNClassifierModel` will extract the data from feature
   column (only Scalar, Array[_] or Vector data type are supported) and convert each feature
   to Tensor according to the specified Tensor size. User can also set featureSize as
   Array[Array[Int]] for multi-inputs model.

**3.** `NNClassifierModel(model, featurePreprocessing: Preprocessing[F, Tensor[T]])`

   Takes model and featurePreprocessing. `NNClassifierModel` will extract the data from feature
   column and convert each feature to Tensor with the featurePreprocessing. This constructor provides
   more flexibility in supporting extra data types.

Meanwhile, for advanced use cases (e.g. model with multiple input tensor), `NNClassifierModel`
supports: `setSamplePreprocessing(value: Preprocessing[Any, Sample[T]])`to directly compose
Sample according to user-specified Preprocessing.

---

## Hyperparameter setting

Prior to the commencement of the training process, you can modify the optimization algorithm, batch 
size, the epoch number of your training, and learning rate to meet your goal or
`NNEstimator`/`NNClassifier` will use the default value.

Continue the codes above, NNEstimator and NNClassifier can be set in the same way.

**Scala:**

```scala
//for esitmator
estimator.setBatchSize(4).setMaxEpoch(10).setLearningRate(0.01).setOptimMethod(new Adam())
//for classifier
classifier.setBatchSize(4).setMaxEpoch(10).setLearningRate(0.01).setOptimMethod(new Adam())
```
**Python:**

```python
# for esitmator
estimator.setBatchSize(4).setMaxEpoch(10).setLearningRate(0.01).setOptimMethod(Adam())
# for classifier
classifier.setBatchSize(4).setMaxEpoch(10).setLearningRate(0.01).setOptimMethod(Adam())

```

## Prepare the data and start the training process

NNEstimator/NNCLassifer supports training with Spark's
[DataFrame/DataSet](https://spark.apache.org/docs/latest/sql-programming-guide.html#datasets-and-dataframes)

Suppose `df` is the training data, simple call `fit` method and let Analytics Zoo train the model
for you.

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
User may also set validation DataFrame and validation frequency through `setValidation` method.
Train summay and validation summary can also be configured to log the training process for
visualization in Tensorboard. See [Visualization](../../ProgrammingGuide/visualization.md) for the details.


## Make prediction on chosen data

Since `NNModel`/`NNClassifierModel` inherits from Spark's `Transformer` abstract class, simply call 
`transform` method on `NNModel`/`NNClassifierModel` to make prediction.

**Scala:**

```scala
nnModel.transform(df).show(false)
```

**Python:**

```python
nnModel.transform(df).show(false)
```

For the complete examples of NNFrames, please refer to:
[Scala examples](https://github.com/intel-analytics/analytics-zoo/tree/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/nnframes)
[Python examples](https://github.com/intel-analytics/analytics-zoo/tree/master/pyzoo/zoo/examples/nnframes)


## NNImageReader

`NNImageReader` is the primary DataFrame-based image loading interface, defining API to read images
into DataFrame.

Scala:
```scala
    val imageDF = NNImageReader.readImages(imageDirectory, sc)
```

Python:
```python
    image_frame = NNImageReader.readImages(image_path, self.sc)
```

The output DataFrame contains a sinlge column named "image". The schema of "image" column can be
accessed from `com.intel.analytics.zoo.pipeline.nnframes.DLImageSchema.byteSchema`.
Each record in "image" column represents one image record, in the format of
Row(origin, height, width, num of channels, mode, data), where origin contains the URI for the image file,
and `data` holds the original file bytes for the image file. `mode` represents the OpenCV-compatible
type: CV_8UC3, CV_8UC1 in most cases.
```scala
  val byteSchema = StructType(
    StructField("origin", StringType, true) ::
      StructField("height", IntegerType, false) ::
      StructField("width", IntegerType, false) ::
      StructField("nChannels", IntegerType, false) ::
      // OpenCV-compatible type: CV_8UC3, CV_32FC3 in most cases
      StructField("mode", IntegerType, false) ::
      // Bytes in OpenCV-compatible order: row-wise BGR in most cases
      StructField("data", BinaryType, false) :: Nil)
```

After loading the image, user can compose the preprocess steps with the `Preprocessing` defined
in `com.intel.analytics.zoo.feature.image`.
