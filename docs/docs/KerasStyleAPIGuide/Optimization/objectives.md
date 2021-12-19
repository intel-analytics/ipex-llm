## Usage of objectives

An objective function (or loss function, or optimization score function) is one of the two parameters required to compile a model:

**Scala:**

```scala
model.compile(loss = "mean_squared_error", optimizer = "sgd")
```

**Python:**

```python
model.compile(loss='mean_squared_error', optimizer='sgd')
```

**Scala:**

```scala
model.compile(loss = MeanSquaredError(sizeAverage = true), optimizer = "sgd")
```

**Python:**

```python
model.compile(loss=MeanSquaredError(size_average=True), optimizer='sgd')
```

---

## Available objectives

### MeanSquaredError

The mean squared error criterion e.g. input: a, target: b, total elements: n

```
loss(a, b) = 1/n * sum(|a_i - b_i|^2)
```

**Scala:**

```scala
loss = MeanSquaredError(sizeAverage = true)
```

Parameters:

 * `sizeAverage` a boolean indicating whether to divide the sum of squared error by n. 
                 Default: true

**Python:**

```python
loss = MeanSquaredError(size_average=True)
```

Parameters:

 * `size_average` a boolean indicating whether to divide the sum of squared error by n. 
                 Default: True

### MeanAbsoluteError

Measures the mean absolute value of the element-wise difference between input and target

**Scala:**

```scala
loss = MeanAbsoluteError(sizeAverage = true)
```

Parameters:

 * `sizeAverage` a boolean indicating whether to divide the sum of squared error by n. 
                 Default: true

**Python:**

```python
loss = MeanAbsoluteError(size_average=True)
```

Parameters:

 * `size_average` a boolean indicating whether to divide the sum of squared error by n. 
                 Default: True

### BinaryCrossEntropy

Also known as logloss. 

**Scala:**

```scala
loss = BinaryCrossEntropy(weights = null, sizeAverage = true)
```

Parameters:

* `weights` A tensor assigning weight to each of the classes
* `sizeAverage` whether to divide the sequence length. Default is true.

**Python:**

```python
loss = BinaryCrossEntropy(weights=None, size_average=True)
```

Parameters:

* `weights` A tensor assigning weight to each of the classes
* `size_average` whether to divide the sequence length. Default is True.

### SparseCategoricalCrossEntropy

A loss often used in multi-class classification problems with SoftMax as the last layer of the neural network. By default, input(y_pred) is supposed to be probabilities of each class, and target(y_true) is supposed to be the class label starting from 0.

**Scala:**

```scala
loss = SparseCategoricalCrossEntropy(logProbAsInput = false, zeroBasedLabel = true, weights = null, sizeAverage = true, paddingValue = -1)
```

Parameters:

 * `logProbAsInput` Boolean. Whether to accept log-probabilities or probabilities as input. Default is false and inputs should be probabilities.
 * `zeroBasedLabel` Boolean. Whether target labels start from 0. Default is true. If false, labels start from 1.
 * `weights` Tensor. Weights of each class if you have an unbalanced training set. Default is null.
 * `sizeAverage` Boolean. Whether losses are averaged over observations for each mini-batch. Default is true. If false, the losses are instead summed for each mini-batch.
 * `paddingValue` Integer. If the target is set to this value, the training process will skip this sample. In other words, the forward process will return zero output and the backward process will also return zero gradInput. Default is -1.

**Python:**

```python
loss = SparseCategoricalCrossEntropy(log_prob_as_input=False, zero_based_label=True, weights=None, size_average=True, padding_value=-1)
```

Parameters:

 * `log_prob_as_input` Boolean. Whether to accept log-probabilities or probabilities as input. Default is false and inputs should be probabilities.
 * `zero_based_label` Boolean. Whether target labels start from 0. Default is true. If false, labels start from 1.
 * `weights` A Numpy array. Weights of each class if you have an unbalanced training set. Default is None.
 * `size_average` Boolean. Whether losses are averaged over observations for each mini-batch. Default is True. If False, the losses are instead summed for each mini-batch.
 * `padding_value` Integer. If the target is set to this value, the training process will skip this sample. In other words, the forward process will return zero output and the backward process will also return zero gradInput. Default is -1.

### MeanAbsolutePercentageError

Compute mean absolute percentage error for intput and target

**Scala:**

```scala
loss = MeanAbsolutePercentageError()
```

Parameters:

 * `sizeAverage` a boolean indicating whether to divide the sum of squared error by n. 
                 Default: true

**Python:**

```python
loss = MeanAbsolutePercentageError()
```

Parameters:

 * `size_average` a boolean indicating whether to divide the sum of squared error by n. 
                 Default: True

### MeanSquaredLogarithmicError

Compute mean squared logarithmic error for input and target

**Scala:**

```scala
loss = MeanSquaredLogarithmicError()
```

**Python:**

```python
loss = MeanSquaredLogarithmicError()
```

### CategoricalCrossEntropy

This is same with cross entropy criterion, except the target tensor is a
one-hot tensor.

**Scala:**

```scala
loss = CategoricalCrossEntropy()
```

**Python:**

```python
loss = CategoricalCrossEntropy()
```

### Hinge

Creates a criterion that optimizes a two-class classification hinge loss (margin-based loss) between input x (a Tensor of dimension 1) and output y.

**Scala:**

```scala
loss = Hinge(margin = 1.0, sizeAverage = true)
```

Parameters:

 * `margin` if unspecified, is by default 1.
 * `sizeAverage` whether to average the loss, is by default true

**Python:**

```python
loss = Hinge(margin=1.0, size_average=True)
```

Parameters:

 * `margin` if unspecified, is by default 1.
 * `size_average` whether to average the loss, is by default True

### RankHinge

Hinge loss for pairwise ranking problems.

**Scala:**

```scala
loss = RankHinge(margin = 1.0)
```

Parameters:

 * `margin` if unspecified, is by default 1.

**Python:**

```python
loss = RankHinge(margin=1.0)
```

Parameters:

 * `margin` if unspecified, is by default 1.

### SquaredHinge

Creates a criterion that optimizes a two-class classification squared hinge loss (margin-based loss) between input x (a Tensor of dimension 1) and output y.

**Scala:**

```scala
loss = SquaredHinge(margin = 1.0, sizeAverage = true)
```

Parameters:

 * `margin` if unspecified, is by default 1.
 * `sizeAverage` whether to average the loss, is by default true

**Python:**

```python
loss = SquaredHinge(margin=1.0, size_average=True)
```

Parameters:

 * `margin` if unspecified, is by default 1.
 * `size_average` whether to average the loss, is by default True

### Poisson

Compute Poisson error for intput and target

**Scala:**

```scala
loss = Poisson()
```

**Python:**

```python
loss = Poisson()
```

### CosineProximity

Computes the negative of the mean cosine proximity between predictions and targets.

**Scala:**

```scala
loss = CosineProximity()
```

**Python:**

```python
loss = CosineProximity()
```

### KullbackLeiblerDivergence

Loss calculated as:
```
y_true = K.clip(y_true, K.epsilon(), 1)
y_pred = K.clip(y_pred, K.epsilon(), 1)
```
and output K.sum(y_true * K.log(y_true / y_pred), axis=-1)

**Scala:**

```scala
loss = KullbackLeiblerDivergence()
```

**Python:**

```python
loss = KullbackLeiblerDivergence()
```
