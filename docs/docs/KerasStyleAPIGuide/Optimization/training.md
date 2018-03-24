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
**Python**
```python
compile(optimizer, loss, metrics=None)
```

Parameters:

* `optimizer`: Optimization method to be used. Can either use the string representation of an optimization method (see [here](optimizer/#available-optimizers)) or an instance of [OptimMethod](../../APIGuide/Optimizers/Optim-Methods/). 
* `loss`: Criterion to be used. Can either use the string representation of a criterion (see [here](loss/#available-losses)) or an instance of [Loss](../../APIGuide/Losses/).
* `metrics`: One or more validation methods to be used. Default is null if no validation needs to be configured. Can either use the string representation `Array("accuracy")`(Scala) `["accuracy"]`(Python) or instances of [ValidationMethod](../../APIGuide/Metrics/).

---
## **Fit**

Train a model for a fixed number of epochs on a dataset. Need to first [compile](#compile) the model beforehand.

**Scala:**
```scala
fit(x, nbEpoch = 10, validationData = null)
```
**Python**
```python
fit(x, y=None, batch_size=32, nb_epoch=10, validation_data=None, distributed=True)
```

Parameters:

* `x`: Training dataset.
* `batchSize`: Number of samples per gradient update.
* `nbEpoch`: Number of iterations to train.
* `validationData`: Dataset for validation. Default is null if validation is not configured.

**Remark**

- For __Scala__, x can either be RDD of [Sample](../../APIGuide/Data/#sample) (specifying `batchSize`) or an instance of [DataSet](../../APIGuide/Data/#dataset).
- For __Python__, you can use x (a Numpy array) as features with y (a Numpy array) as labels; or only x (RDD of [Sample](../../APIGuide/Data/#sample)) without specifying y.
- The parameter `distributed` is to choose whether to train the model using distributed mode or local mode in __Python__. Default is true. If in local mode, x and y must both be Numpy arrays.


---
## **Evaluate**

Evaluate a model on a given dataset using the metrics specified when you [compile](#compile) the model.

**Scala:**
```scala
evaluate(x)
```
**Python**
```python
evaluate(x, y=None, batch_size=32)
```

Parameters:

* `x`: Evaluation dataset.
* `batchSize`: Number of samples per batch.

**Remark**

- For __Scala__, x can either be RDD of [Sample](../../APIGuide/Data/#sample) (specifying `batchSize`) or an instance of [DataSet](../../APIGuide/Data/#dataset).
- For __Python__, you can use x (a Numpy array) as features with y (a Numpy array) as labels; or only x (RDD of [Sample](../../APIGuide/Data/#sample)) without specifying y. Currently only evaluation in distributed mode is supported in Python.

---
## **Predict**

Use a model to do prediction.

**Scala:**
```scala
predict(x)
```
**Python**
```python
predict(x, distributed=True)
```

Parameters:

* `x`: Prediction data.

**Remark**

- For __Scala__, x can either be RDD of [Sample](../../APIGuide/Data/#sample) (specifying `batchSize`) or an instance of `LocalDataSet`.
- For __Python__, x can either be a Numpy array representing labels or RDD of [Sample](../../APIGuide/Data/#sample).
- The parameter `distributed` is to choose whether to do prediction using distributed mode or local mode in __Python__. Default is true. If in local mode, x must be a Numpy array.
