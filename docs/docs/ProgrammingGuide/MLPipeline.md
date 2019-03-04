
## **Overview**

BigDL provides `DLEstimator` and `DLClassifier` for users with Apache Spark MLlib experience, which
provides high level API for training a BigDL Model with the Apache Spark
[Estimator](https://spark.apache.org/docs/2.1.1/ml-pipeline.html#estimators)/
[Transformer](https://spark.apache.org/docs/2.1.1/ml-pipeline.html#transformers)
pattern, thus users can conveniently fit BigDL into a ML pipeline. The fitted model `DLModel` and
`DLClassiferModel` contains the trained BigDL model and extends the Spark ML `Model` class.
Alternatively users may also construct a `DLModel` with a pre-trained BigDL model to use it in
Spark ML Pipeline for prediction. We are going to show you how to define a DLEstimator and
DLClassifier and how to use it. For advanced users, please check our
[ML Pipeline API](../APIGuide/DLFrames/DLEstimator_DLClassifier.md) for detailed usage.


---
## **Define a DLEstimator**
Before we are trying to use DLEstimator to automate the training process, we need to make clear
which model is used to update parameters and gradients, which criterion is used to measure the loss and 
the dimension of the features and the label. These are the key elements the DLEstimator required to
prepare for the training. If you are unfamiliar with creating a model and criterion, check out
[Model](./Model/Sequential.md) and [Losses](../APIGuide/Losses.md) sections with provided links.

So, suppose we create a model with single linear layer and use  MSECriterion as loss function here.
You can choose any other model or criterion for your own good when you start your own training.

Then basically one can write code like this:

**Scala:**

```scala
import com.intel.analytics.bigdl.dlframes.DLEstimator
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import com.intel.analytics.bigdl.nn._

val model = Sequential().add(Linear(2, 2))
val estimator = new DLEstimator(model, MSECriterion(), Array(2), Array(2))
```

**Python:**

```python
linear_model = Sequential().add(Linear(2, 2))
mse_criterion = MSECriterion()
estimator = DLEstimator(model=linear_model, criterion=mse_criterion,
feature_size=[2], label_size=[2])
```

Now, you have a DLEstimator based on your own choice of model, criterion. Also, make sure your specified
feature size and label size consistent with the actual ones otherwise exception will be encountered.

## **Define a DLClassifier**
Since DLlassifier is the subclass of DLEstimator, the way of defining a DLClassifier is almost the same
like creating a DLEstimator except that you don't need to specify the label size because it's set to
default binary value and pay attention to choosing the criterion suitable for classification problem wisely.

Suppose we still create a model with single linear layer and use  ClassNLL criterion as loss
function here.

**Scala:**

```scala
val model = Sequential().add(Linear(2, 2)).add(LogSoftMax())
val criterion = ClassNLLCriterion()
val estimator = new DLClassifier(model, criterion, Array(2))
```
**Python:**

```python
linear_model = Sequential().add(Linear(2, 2))
classNLL_criterion = ClassNLLCriterion()
classifier = DLClassifier(model=linear_model, criterion=classNLL_criterion,
feature_size=[2])
```

## Hyperparameter setting

Prior to the commencement of the training process, you can modify the batch size, the epoch number of your
training, and learning rate to meet your goal or DLEstimator/DLClassifier will use the default value.

Continue the codes above, DLEstimator and DLClassifier can be setted in the same way.

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
to feed to the DLEstimator/DLCLassifer.
Then after these steps, we can start training now.

Suppose `df` is the training data, simple call `fit` method and let BigDL train the model for you. You will
get a DLModel or DLClassifierModel based on which one you choose from DLEstimator and DLClassifier.

**Scala:**

```scala
//get a DLModel
val dlModel = estimator.fit(df)
//get a DLClassifierModel
val dlClassifierModel = classifier.fit(df)
```

**Python:**

```python
# get a DLModel
dlModel = estimator.fit(df)
# get a DLClassifierModel
dlClassifierModel = classifier.fit(df)
```
## Make prediction on chosen data by using DLModel/DLClassifierModel

Since DLModel/DLClassifierModel inherits from Spark's Transformer abstract class, simply call `transform`
 method on DLModel/DLClassifierModel to make prediction.

**Scala:**

```scala
dlModel.transform(df).show(false)
```

**Python:**

```python
dlModel.transform(df).show(false)
```




