This page shows how to train, evaluate or predict a model using the Keras-Style API.

You may refer to the `User Guide` page to see how to define a model in [Python](../keras-api-python/) or [Scala](../keras-api-scala/) correspondingly.

You may refer to [`Layers`](../Layers/core/) section to find all the available layers.

After defining a model with the Keras-Style API, you can call the following __methods__ on the model:


---
## **Compile**

Configure the learning process. Must be called before [fit](#fit) or [evaluate](#evaluate).

**Scala:**
```scala
compile(optimizer, loss, metrics = null)
```

Parameters:

* `optimizer`: Optimization method to be used.
* `loss`: Criterion to be used.
* `metrics`: Validation method(s) to be used. Default is null if no validation is needed. 

Alternatively, one can pass in the corresponding Keras-Style string representations when calling compile. For example: optimizer = "sgd", loss = "mse", metrics = List("accuracy")

**Python**
```python
compile(optimizer, loss, metrics=None)
```

Parameters:

* `optimizer`: Optimization method to be used. One can alternatively pass in the corresponding string representation, such as 'sgd'.
* `loss`: Criterion to be used. One can alternatively pass in the corresponding string representation, such as 'mse'. (see [here](objectives/#available-objectives)).
* `metrics`: List of validation methods to be used. Default is None if no validation is needed. For convenience, string representations are supported: 'accuracy' (or 'acc'), 'top5accuracy' (or 'top5acc'), 'mae', 'auc', 'treennaccuracy' and 'loss'. For example, you can either use [Accuracy()] or ['accuracy'].

---
## **Fit**

Train a model for a fixed number of epochs on a DataSet.

**Scala:**
```scala
fit(x, batchSize = 32，nbEpoch = 10, validationData = null)
```

Parameters:

* `x`: Training dataset. RDD of Sample or [ImageSet](../../APIGuide/FeatureEngineering/image/) or [TextSet](../../APIGuide/FeatureEngineering/text).
* `batchSize`: Number of samples per gradient update. Default is 32.
* `nbEpoch`: Number of epochs to train. Default is 10.
* `validationData`: RDD of Sample or ImageSet or TextSet, or null if validation is not configured. Default is null.

**Python**
```python
fit(x, y=None, batch_size=32, nb_epoch=10, validation_data=None, distributed=True)
```

Parameters:

* `x`: Training data. A Numpy array or RDD of Sample or [ImageSet](../../APIGuide/FeatureEngineering/image/) or [TextSet](../../APIGuide/FeatureEngineering/text).
* `y`: Labels. A Numpy array. Default is None if x is already Sample RDD or ImageSet or TextSet.
* `batch_size`: Number of samples per gradient update. Default is 32.
* `nb_epoch`: Number of epochs to train. Default is 10.
* `validation_data`: Tuple (x_val, y_val) where x_val and y_val are both Numpy arrays.
                    Can also be RDD of Sample or ImageSet or TextSet.
                    Default is None if no validation is involved.
* `distributed`: Boolean. Whether to train the model in distributed mode or local mode.
                 Default is True. In local mode, x and y must both be Numpy arrays.

---
## **Evaluate**

Evaluate a model on a given dataset in distributed mode.

**Scala:**
```scala
evaluate(x, batchSize = 32)
```

Parameters:

* `x`: Evaluation dataset. RDD of Sample or [ImageSet](../../APIGuide/FeatureEngineering/image/) or [TextSet](../../APIGuide/FeatureEngineering/text).
* `batchSize`: Number of samples per batch. Default is 32.

**Python**
```python
evaluate(x, y=None, batch_size=32)
```

Parameters:

* `x`: Evaluation data. A Numpy array or RDD of Sample or [ImageSet](../../APIGuide/FeatureEngineering/image/) or [TextSet](../../APIGuide/FeatureEngineering/text).
* `y`: Labels. Default is None if x is set already. A Numpy array or RDD of Sample or ImageSet or TextSet.
* `batch_size`: Number of samples per batch. Default is 32.

---
## **Predict**

Use a model to do prediction.

**Scala:**
```scala
predict(x, batchPerThread = 4)
```

Parameters:

* `x`: Prediction dataset. RDD of Sample or [ImageSet](../../APIGuide/FeatureEngineering/image/) or [TextSet](../../APIGuide/FeatureEngineering/text).
* `batchPerThread`: The total batchSize is batchPerThread * numOfCores.

**Python**
```python
predict(x, batch_per_thread=4, distributed=True)
```

Parameters:

* `x`: Prediction data. A Numpy array or RDD of Sample or [ImageSet](../../APIGuide/FeatureEngineering/image/) or [TextSet](../../APIGuide/FeatureEngineering/text).
* `batch_per_thread`:
        The default value is 4.
        When distributed is True, the total batch size is batch_per_thread * rdd.getNumPartitions.
        When distributed is False, the total batch size is batch_per_thread * numOfCores.
* `distributed`: Boolean. Whether to do prediction in distributed mode or local mode.
                 Default is True. In local mode, x must be a Numpy array.
                 
Use a model to predict class labels.

**Scala:**
```scala
predictClasses(x, batchPerThread = 4, zeroBasedLabel = true)
```

Parameters:

* `x`: Prediction dataset. RDD of Sample or [ImageSet](../../APIGuide/FeatureEngineering/image/) or [TextSet](../../APIGuide/FeatureEngineering/text).
* `batchPerThread`: The default value is 4, and the total batchSize is batchPerThread * rdd.getNumPartitions.
* `zeroBasedLabel`: Boolean. Whether result labels start from 0. Default is true. If false, result labels start from 1.

**Python**
```python
predict_classes(x, batch_per_thread=4, zero_based_label=True)
```

Parameters:

* `x`: Prediction data. A Numpy array or RDD of Sample or [ImageSet](../../APIGuide/FeatureEngineering/image/) or [TextSet](../../APIGuide/FeatureEngineering/text).
* `batch_per_thread`:
        The default value is 4.
        When distributed is True,the total batch size is batch_per_thread * rdd.getNumPartitions.
        When distributed is False the total batch size is batch_per_thread * numOfCores.
* `zero_based_label`: Boolean. Whether result labels start from 0.
                      Default is True. If False, result labels start from 1.

## **Visualization**

We use tensorbroad-compatible tevent file to store the training and validation metrics. Then you could use tensorboard to visualize the training, or use analytics-zoo build-in API to read the metrics.

### **Enable training metrics**
The training metrics will be saved to `logDir/appName/training`, and validation metrics will be saved to `logDir/appName/validation`

**scala**
```scala
setTensorBoard(logDir, appName)
```
Parameters:

* `logDir`: The base directory path to store training and validation logs.
* `appName`: The name of the application.

**python**
```python
set_tensorboard(log_dir, app_name)
```
Parameters:

* `log_dir`: The base directory path to store training and validation logs.
* `app_name`: The name of the application.

### **Validation with tensorboard**

TODO: add link

### **Reading metrics with build-in API**
To get scalar metrics with build-in API, you can use following API. 

**scala**
```scala
getTrainSummary(tag)
```
Get training metrics by tag. Parameters:  

* `tag`: The string variable represents the parameter you want to return supported tags are "LearningRate", "Loss", "Throughput".

**scala**
```scala
getValidationSummary(tag)
```
Get validation metrics by tag. Parameters:  

* `tag`: The string variable represents the parameter you want to return supported tags are 'AUC', 'Accuracy', 'BinaryAccuracy', 'CategoricalAccuracy', 'HitRatio', 'Loss', 'MAE', 'NDCG', 'SparseCategoricalAccuracy', 'TFValidationMethod', 'Top1Accuracy', 'Top5Accuracy', 'TreeNNAccuracy'.


**python**
```python
get_train_summary(tag)
```
Get training metrics by tag. Parameters:  

* `tag`: The string variable represents the parameter you want to return supported tags are "LearningRate", "Loss", "Throughput".

**python**
```python
get_validation_summary(tag)
```
Get validation metrics by tag. Parameters:  

* `tag`: The string variable represents the parameter you want to return supported tags are 'AUC', 'Accuracy', 'BinaryAccuracy', 'CategoricalAccuracy', 'HitRatio', 'Loss', 'MAE', 'NDCG', 'SparseCategoricalAccuracy', 'TFValidationMethod', 'Top1Accuracy', 'Top5Accuracy', 'TreeNNAccuracy'.

