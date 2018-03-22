This page shows how to train, predict or evaluate a model using the Keras-Style API.

You may refer to `User Guide` page to see how to define a model in [Python](../keras-api-python) or [Scala](../keras-api-scala.md) correspondingly.

You may refer to [`Layers`](../Layers/core.md) section to find all the available layers.

After defining a model with the Keras-Style API, you can call the following methods:


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

---
## **Fit**

Train a model for a fixed number of epochs on a dataset.

**Scala:**
```scala
fit(x, batchSize = 32, nbEpoch = 10, validationData = null)
```
**Python**
```python
fit(x, y=None, batch_size=32, nb_epoch=10, validation_data=None, distributed=True)
```

---
## **Evaluate**

Evaluate a model on a given dataset.

**Scala:**
```scala
evaluate(x, batchSize = 32)
```
**Python**
```python
evaluate(x, y=None, batch_size=32)
```

---
## **Predict**

Use a model to do prediction.

**Scala:**
```scala
predict(x, batchSize = 32)
```
**Python**
```python
predict(x, distributed=True)
```
