---
## **Introduction**

Analytics Zoo Orca BigDL Estimator provides a set APIs for running BigDL model on Spark in a distributed fashion.

---

### Orca BigDL Estimator

Orca BigDL Estimator is an estimator to do BigDL training/evaluation/prediction on Spark in a distributed fashion.

It can support various data types, like XShards, Spark DataFrame, etc.

It supports BigDL backend in the unified APIs.

### Create Estimator from BigDL Model

You can create Orca BigDL Estimator with BigDL model.

```
from zoo.orca.learn.bigdl import Estimator
Estimator.from_bigdl(*, model, loss=None, optimizer=None, feature_preprocessing=None,
                   label_preprocessing=None, model_dir=None)
```
* `model`: BigDL Model to be trained.
* `optimizer`: BigDL optimizer.
* `loss`: BigDL criterion.
* `feature_preprocessing`: Used if the input data in fit function is Spark DataFrame.The param converts the data in feature column to a Tensor or to a Sample directly. It expects a List of Int as the size of the converted Tensor, or a Preprocessing[F, Tensor[T]]
    If a List of Int is set as feature_preprocessing, it can only handle the case that feature column contains the following data types: Float, Double, Int, Array[Float], Array[Double], Array[Int] and MLlib Vector. The feature data are converted to Tensors with the specified sizes before sending to the model. Internally, a SeqToTensor is generated according to the size, and used as the feature_preprocessing.
    Alternatively, user can set feature_preprocessing as Preprocessing[F, Tensor[T]] that transforms the feature data to a Tensor[T]. Some pre-defined Preprocessing are provided in package zoo.feature. Multiple Preprocessing can be combined as a ChainedPreprocessing.
    The feature_preprocessing will also be copied to the generated NNModel and applied to feature column during transform.
* `label_preprocessing`: Similar to feature_preprocessing, but applies to Label data.
* `model_dir`: The path to save model. During the training, if checkpoint_trigger is defined and triggered, the model will be saved to model_dir.

### Train BigDL model with orca BigDL Estimator
After an Estimator is created, you can call estimator API to train BigDL model:
```
fit(self, data, epochs, batch_size=32, feature_cols="features", label_cols="label",
    caching_sample=True, validation_data=None, validation_trigger=None,
    validation_metrics=None, checkpoint_trigger=None)
```
* `data`: Training data. SparkXShard and Spark DataFrame are supported.
* `epochs`: (int) Number of epochs to train the model.
* `batch_size`: (int) Batch size used for training. Default: 32.
* `feature_cols`: (string or list of string) Feature column name(s) of data. Only used when data is a Spark DataFrame. Default: "features".
* `label_cols`: (string or list of string) Label column name(s) of data. Only used when data is a Spark DataFrame. Default: "label".
* `caching_sample`: (Boolean) Whether to cache the Samples after preprocessing. Default: True.
* `validation_data`: Validation data. SparkXShard and Spark DataFrame are supported. Default: None.
* `validation_trigger`: Orca Trigger to validate model.
* `validation_metrics`: Orca validation methods.
* `checkpoint_trigger`: Orca Trigger to set a checkpoint.

### Inference with orca BigDL Estimator
After training or loading trained model, you can call estimator API to inference:
```
predict(self, data, batch_size=4, feature_cols="features", sample_preprocessing=None)
```
* `data`: Inference data. SparkXShard and Spark DataFrame are supported.
* `batch_size`: (int) Batch size used for inference. Default: 4.
* `feature_cols`:  (string or list of string) Feature column name(s) of data. Only used when data is a Spark DataFrame.
* `sample_preprocessing`: Used if the input data in predict function is Spark DataFrame. The user defined sample_preprocessing will directly compose Sample according to user-specified Preprocessing.

### Evaluate model
After Training, you can call estimator API to evaluate BigDL model:
```
evaluate(self, data, batch_size=32, feature_cols=None, label_cols=None, validation_metrics=None)
```
* `data`: Validation data. SparkXShard and Spark DataFrame are supported.
* `batch_size`: Batch size used for evaluation. Only used when data is a SparkXShard.
* `feature_cols`: (string or list of string) Feature column name(s) of data. Only used when data is a Spark DataFrame. Default: None.
* `label_cols`: (string or list of string) Label column name(s) of data. Only used when data is a Spark DataFrame. Default: None.
* `validation_metrics`: Orca validation methods.


### Get model
You can get model using `get_model(self)`

### Save model
You can save model using `save(self, model_path)`
* `model_path`: (str) Path to model saved folder.

### Load model
You can load saved model using
```
load(self, checkpoint, optimizer=None, loss=None, feature_preprocessing=None,
             label_preprocessing=None, model_dir=None, is_checkpoint=False):
```
* `checkpoint`: (str) Path to target checkpoint file or saved model folder.
* `optimizer`: BigDL optimizer.
* `loss`: BigDL criterion.
* `feature_preprocessing`: Used if the input data in fit function is Spark DataFrame.The param converts the data in feature column to a Tensor or to a Sample directly. It expects a List of Int as the size of the converted Tensor, or a Preprocessing[F, Tensor[T]]
    If a List of Int is set as feature_preprocessing, it can only handle the case that feature column contains the following data types: Float, Double, Int, Array[Float], Array[Double], Array[Int] and MLlib Vector. The feature data are converted to Tensors with the specified sizes before sending to the model. Internally, a SeqToTensor is generated according to the size, and used as the feature_preprocessing.
    Alternatively, user can set feature_preprocessing as Preprocessing[F, Tensor[T]] that transforms the feature data to a Tensor[T]. Some pre-defined Preprocessing are provided in package zoo.feature. Multiple Preprocessing can be combined as a ChainedPreprocessing.
    The feature_preprocessing will also be copied to the generated NNModel and applied to feature column during transform.
* `label_preprocessing`: Similar to feature_preprocessing, but applies to Label data.
* `model_dir`: The path to save model. During the training, if checkpoint_trigger is defined and triggered, the model will be saved to model_dir.
* `is_checkpoint`: (boolean) Whether load BigDL saved model or checkpoint.

### Set TensorBoard & get Training and Validation Summary
During training and validation, Orca BigDL Estimator would save summary data to specified log_dir. This data can be visualized in TensorBoard, or you can use estimator's APIs to retrieve it. You can set the logdir and app name with such API:
```
set_tensorboard(log_dir, app_name)
```
* `log_dir`: The base directory path to store training and validation logs.
* `app_name`: The name of the application.

This method sets summary information during the training process for visualization purposes. Saved summary can be viewed via TensorBoard. In order to take effect, it needs to be called before fit.

Training summary will be saved to 'log_dir/app_name/train' and validation summary (if any) will be saved to 'log_dir/app_name/validation'.

You can get Training summary with `get_train_summary(self, tag=None)` and Validation summary with `get_validation_summary(self, tag=None)`.

### Clear gradient clipping
You can clear gradient clipping parameters using `clear_gradient_clipping(self)`. In this case, gradient clipping will not be applied.
**Note:** In order to take effect, it needs to be called before fit.

### Set constant gradient clipping
You can Set constant gradient clipping during the training process using `set_constant_gradient_clipping(self, min, max)`.
* `min`: The minimum value to clip by.
* `max`: The maximum value to clip by.
**Note:** In order to take effect, it needs to be called before fit.

### Set clip gradient to a maximum L2-Norm
You can set clip gradient to a maximum L2-Norm during the training process using `set_l2_norm_gradient_clipping(self, clip_norm)`.
* `clip_norm`: Gradient L2-Norm threshold.
**Note:** In order to take effect, it needs to be called before fit.



