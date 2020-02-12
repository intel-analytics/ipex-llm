# TFEstimator

TFEstimator wraps a model defined by `model_fn`. The `model_fn` is almost identical to TensorFlow's `model_fn`
except users are required to use ZooOptimizer, which takes a `tf.train.Optimzer` as input, to derive a train_op.



__Remarks__:

- You need to install __tensorflow==1.15.0__ on your driver node.
- Your operating system (OS) is required to be one of the following 64-bit systems:
__Ubuntu 16.04 or later__ and __macOS 10.12.6 or later__.
- To run on other systems, you need to manually compile the TensorFlow source code. Instructions can
  be found [here](https://github.com/tensorflow/tensorflow/tree/v1.10.0/tensorflow/java).


**Create a TFEstimator from a model_fn**:
```python
import tensorflow as tf
from zoo.tfpark import TFEstimator, ZooOptimizer
def model_fn(features, labels, mode):

    hidden = tf.layers.dense(features, 32, activation=tf.nn.relu)
    
    logits = tf.layers.dense(hidden, 10)

    if mode == tf.estimator.ModeKeys.EVAL or mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.reduce_mean(
            tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels))
        train_op = ZooOptimizer(tf.train.AdamOptimizer()).minimize(loss)
        return tf.estimator.EstimatorSpec(mode, train_op=train_op, predictions=logits, loss=loss)
    else:
        return tf.estimator.EstimatorSpec(mode, predictions=logits)

estimator = TFEstimator.from_model_fn(model_fn, model_dir="/tmp/estimator")
```

**Create a TFEstimator from a pre-made estimator**:
```python
import tensorflow as tf
linear = tf.estimator.LinearClassifier(feature_columns=feature_columns,
                                           optimizer=ZooOptimizer(tf.train.FtrlOptimizer(0.2)))
estimator = TFEstimator(linear)
```

## Methods

### \_\_init\_\_

Create a TFEstimator from a tf.estimator.Estimator

```python
TFEstimator(estimator)
```

### from_model_fn

Create a TFEstimator from a model_fn

```python
TFEstimator.from_model_fn(model_fn, model_dir=None, config=None, params=None, warm_start_from=None)
```

#### Arguments

* **model_fn**: Model function. Follows the signature:

            * Args:

                * `features`: This is the first item returned from the `input_fn`
                    passed to `train`, `evaluate`, and `predict`. This should be a
                    single `tf.Tensor` or `dict` of same.
                * `labels`: This is the second item returned from the `input_fn`
                    passed to `train`, `evaluate`, and `predict`. This should be a
                    single `tf.Tensor` or `dict` of same (for multi-head models).
                    If mode is `tf.estimator.ModeKeys.PREDICT`, `labels=None` will
                    be passed. If the `model_fn`'s signature does not accept
                    `mode`, the `model_fn` must still be able to handle
                    `labels=None`.
                * `mode`: Optional. Specifies if this training, evaluation or
                    prediction. See `tf.estimator.ModeKeys`.
                * `params`: Optional `dict` of hyperparameters.  Will receive what
                    is passed to Estimator in `params` parameter. This allows
                    to configure Estimators from hyper parameter tuning.
                * `config`: Optional `estimator.RunConfig` object. Will receive what
                    is passed to Estimator as its `config` parameter, or a default
                    value. Allows setting up things in your `model_fn` based on
                    configuration such as `num_ps_replicas`, or `model_dir`.

            * Returns:
                `tf.estimator.EstimatorSpec`
            For the train_op in tf.estimator.EstimatorSpec, it derive from and only from
                        `zoo.tfpark.ZooOptimizer`
* **model_dir**: Directory to save model parameters, graph and etc. This can
            also be used to load checkpoints from the directory into an estimator to
            continue training a previously saved model. If `PathLike` object, the
            path will be resolved. If `None`, the model_dir in `config` will be used
            if set. If both are set, they must be same. If both are `None`, a
            temporary directory will be used.
* **config**: `estimator.RunConfig` configuration object.
* **params**: `dict` of hyper parameters that will be passed into `model_fn`.
              Keys are names of parameters, values are basic python types.
* **warm_start_from**: Optional string filepath to a checkpoint or SavedModel to
                       warm-start from, or a `tf.estimator.WarmStartSettings`
                       object to fully configure warm-starting.  If the string
                       filepath is provided instead of a
                       `tf.estimator.WarmStartSettings`, then all variables are
                       warm-started, and it is assumed that vocabularies
                       and `tf.Tensor` names are unchanged.


### train

```python
train(input_fn, steps=None)
```

#### Arguments

* **input_fn**: A function that constructs the input data for evaluation. The
            function should construct and return one of the following:
            
            * A `TFDataset` object, each elements of which is a tuple `(features, labels)`.
            * A `tf.data.Dataset` object: Outputs of `Dataset` object must be a tuple
            `(features, labels)` with same constraints as below.
            * A tuple `(features, labels)`: Where `features` is a `tf.Tensor` or a dictionary
            of string feature name to `Tensor` and `labels` is a `Tensor` or a
            dictionary of string label name to `Tensor`. Both `features` and
            `labels` are consumed by `model_fn`. They should satisfy the expectation
            of `model_fn` from inputs.
* **steps**: Number of steps for which to train the model.


### evaluate

```python
evaluate(input_fn, eval_methods, steps=None, checkpoint_path=None)
```

#### Arguments

* **input_fn**: A function that constructs the input data for evaluation. The
            function should construct and return one of the following:
            
            * A `TFDataset` object, each elements of which is a tuple `(features, labels)`.
            * A `tf.data.Dataset` object: Outputs of `Dataset` object must be a tuple
            `(features, labels)` with same constraints as below.
            * A tuple `(features, labels)`: Where `features` is a `tf.Tensor` or a dictionary
            of string feature name to `Tensor` and `labels` is a `Tensor` or a
            dictionary of string label name to `Tensor`. Both `features` and
            `labels` are consumed by `model_fn`. They should satisfy the expectation
            of `model_fn` from inputs.
* **eval_methods**: a list of strings to specify the evaluation metrics to
                    be used in this model
* **steps**: Number of steps for which to evaluate model.
* **checkpoint_path**: Path of a specific checkpoint to evaluate. If `None`, the
            latest checkpoint in `model_dir` is used.  If there are no checkpoints
            in `model_dir`, evaluation is run with newly initialized `Variables`
            instead of ones restored from checkpoint.

### predict

```python
predict(input_fn, checkpoint_path=None)
```

#### Arguments

* **input_fn**: A function that constructs the features.
              
              * A `TFDataset` object, each elements of which is a tuple `(features, None)`.
              * A `tf.data.Dataset` object: Outputs of `Dataset` object must have
                same constraints as below.
              * features: A `tf.Tensor` or a dictionary of string feature name to
                `Tensor`. features are consumed by `model_fn`. They should satisfy
                the expectation of `model_fn` from inputs.
              * A tuple, in which case the first item is extracted as features.

* **checkpoint_path**: Path of a specific checkpoint to predict. If `None`, the
            latest checkpoint in `model_dir` is used.  If there are no checkpoints
            in `model_dir`, prediction is run with newly initialized `Variables`
            instead of ones restored from checkpoint.


