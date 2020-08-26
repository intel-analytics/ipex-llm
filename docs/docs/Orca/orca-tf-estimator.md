---
## **Introduction**

Analytics Zoo Orca Tenorflow Estimator provides a set APIs for running TensorFlow model on Spark in a distributed fashion. 

__Remarks__:

- You need to install __tensorflow==1.15.0__ on your driver node.
- Your operating system (OS) is required to be one of the following 64-bit systems:
__Ubuntu 16.04 or later__ and __macOS 10.12.6 or later__.
- To run on other systems, you need to manually compile the TensorFlow source code. Instructions can
  be found [here](https://github.com/tensorflow/tensorflow/tree/v1.10.0/tensorflow/java).
---

### Orca TF Estimator

Orca TF Estimator is an estimator to do Tensorflow training/evaluation/prediction on Spark in a d distributed fashion. 

It can support various data types, like XShards, Spark DataFrame, tf.data.Dataset, numpy.ndarrays, etc. 

It supports both native Tensorflow Graph model and Tensorflow Keras model in the unified APIs. 

#### Create Estimator with graph model

You can creating Orca TF Estimator with native Tensorflow graph model.
```
from zoo.orca.learn.tf.estimator import Estimator

Estimator.from_graph(*, inputs, outputs=None, labels=None, loss=None, optimizer=None,
           clip_norm=None, clip_value=None,
           metrics=None, updates=None,
           sess=None, model_dir=None, backend="bigdl")
``` 
* `inputs`: input tensorflow tensors.
* `outputs`: output tensorflow tensors.
* `labels`: label tensorflow tensors.
* `loss`: The loss tensor of the TensorFlow model, should be a scalar
* `optimizer`: tensorflow optimization method.
* `clip_norm`: float >= 0. Gradients will be clipped when their L2 norm exceeds this value.
* `clip_value`:  a float >= 0 or a tuple of two floats.
        
  If clip_value is a float, gradients will be clipped when their absolute value exceeds this value.
  
  If clip_value is a tuple of two floats, gradients will be clipped when their value less than clip_value[0] or larger than clip_value[1].
* `metrics`: dictionary of {metric_name: metric tensor}.
* `sess`: the current TensorFlow Session, if you want to used a pre-trained model, you should use the Session to load the pre-trained variables and pass it to estimator
* `model_dir`: location to save model checkpoint and summaries.
* `backend`: backend for estimator. Now it only can be "bigdl".

This method returns an Estimator object.

#### Train graph model with Estimator
After an Estimator created, you can call estimator API to train Tensorflow graph model:
```
fit(data,
    epochs=1,
    batch_size=32,
    feature_cols=None,
    labels_cols=None,
    validation_data=None,
    hard_code_batch_size=False,
    session_config=None,
    feed_dict=None,
    checkpoint_trigger=None
    )
```
* `data`: train data. It can be XShards, Spark DataFrame, tf.data.Dataset.
        
   If data is XShards, each element needs to be {'x': a feature numpy array
         or a tuple of feature numpy arrays, 'y': a label numpy array or a tuple of
         label numpy arrays}
   
   If data is tf.data.Dataset, each element is a tuple of input tensors.
* `epochs`: number of epochs to train.
* `batch_size`: total batch size for each iteration.
* `feature_cols`: feature column names if train data is Spark DataFrame.
* `labels_cols`: label column names if train data is Spark DataFrame.
* `validation_data`: validation data. Validation data type should be the same as train data.
* `hard_code_batch_size`: whether hard code batch size for training. Default is False.
* `session_config`: tensorflow session configuration for training. Should be object of tf.ConfigProto
* `feed_dict`: a dictionary. The key is TensorFlow tensor, usually a placeholder, the value of the dictionary is a tuple of two elements. 
  
   The first one of the tuple is the value to feed to the tensor in training phase and the second one is the value to feed to the tensor in validation phase.
* `checkpoint_trigger`: when to trigger checkpoint during training. Should be bigdl optimzer trigger, like EveryEpoch(), SeveralIteration(num_iterations),etc.

#### Create Estimator with Keras model

You can creating Orca TF Estimator with Tensorflow Keras model. The model must be compiled.
```
from zoo.orca.learn.tf.estimator import Estimator

Estimator.from_keras(keras_model, metrics=None, model_dir=None, backend="bigdl")
```
* `keras_model`: the tensorflow.keras model, which must be compiled.
* `metrics`: user specified metric.
* `model_dir`: location to save model checkpoint and summaries.
* `backend`: backend for estimator. Now it only can be "bigdl".

This method returns an Estimator object.

#### Train Keras model with Estimator
After an Estimator created, you can call estimator's API to train Tensorflow Keras model:
```
fit(data,
    epochs=1,
    batch_size=32,
    feature_cols=None,
    labels_cols=None,
    validation_data=None,
    hard_code_batch_size=False,
    session_config=None,
    checkpoint_trigger=None
    )
```
* `data`: train data. It can be XShards, Spark DataFrame, tf.data.Dataset.
        
   If data is XShards, each element needs to be {'x': a feature numpy array
         or a tuple of feature numpy arrays, 'y': a label numpy array or a tuple of
         label numpy arrays}
   
   If data is tf.data.Dataset, each element is a tuple of input tensors.
* `epochs`: number of epochs to train.
* `batch_size`: total batch size for each iteration.
* `feature_cols`: feature column names if train data is Spark DataFrame.
* `labels_cols`: label column names if train data is Spark DataFrame.
* `validation_data`: validation data. Validation data type should be the same as train data.
* `hard_code_batch_size`: whether hard code batch size for training. Default is False.
* `session_config`: tensorflow session configuration for training. Should be object of tf.ConfigProto
* `checkpoint_trigger`: when to trigger checkpoint during training. Should be bigdl optimzer trigger, like EveryEpoch(), SeveralIteration(num_iterations),etc.

#### Evaluate with Estimator

You can call estimator's API to evaluate Tensorflow graph model or keras model.
```
evaluate(data, batch_size=4,
         feature_cols=None,
         labels_cols=None,
         hard_code_batch_size=False
        )
```
* `data`: evaluation data. It can be XShards, Spark DataFrame, tf.data.Dataset.
        
   If data is XShards, each element needs to be {'x': a feature numpy array or a tuple of feature numpy arrays, 'y': a label numpy array or a tuple of label numpy arrays}
   
   If data is tf.data.Dataset, each element is [feature tensor tuple, label tensor tuple]
* `batch_size`: batch size per thread.
* `feature_cols`: feature_cols: feature column names if train data is Spark DataFrame.
* `labels_cols`: label column names if train data is Spark DataFrame.
* `hard_code_batch_size`: whether to hard code batch size for evaluation.

This method returns evaluation result as a dictionary in the format of {'metric name': metric value}

#### Predict with Estimator

You can call estimator's such APIs to predict with trained model.
```
predict(data, batch_size=4,
        feature_cols=None,
        hard_code_batch_size=False
        ):
```
* `data`: data to be predicted.
        It can be XShards, Spark DataFrame, or tf.data.Dataset.
        
        If data is XShard, each element needs to be {'x': a feature numpy array
         or a tuple of feature numpy arrays}.
         
        If data is tf.data.Dataset, each element is feature tensor tuple
* `batch_size`: batch size per thread
* `feature_cols`: list of feature column names if input data is Spark DataFrame.
* `hard_code_batch_size`: if require hard code batch size for prediction. The default value is False.

This method returns a predicted result.
        
If input data is XShards or tf.data.Dataset, the predict result is also a XShards, and the schema for each result is: {'prediction': predicted numpy array or list of predicted numpy arrays}.

If input data is Spark DataFrame, the predict result is a DataFrame which includes original columns plus 'prediction' column. The 'prediction' column can be FloatType, VectorUDT or Array of VectorUDT depending on model outputs shape.

#### Checkpointing and Resume Training

During training, Orca TF Estimator would save Orca checkpoint every epoch. You can also specify `checkpoint_trigger` in fit() to set checkpoint interval. The Orca checckpoints are saved in `model_dir` which is specified when you create estimator. You can load previous Orca checkpoint and resume train with it with such APIs:
 
```
load_latest_orca_checkpoint(path)
``` 
* path: directory containing Orca checkpoint files.

This method load latest checkpoint under specified directory.

If you want to load specified version of checkpoint, you can use:

```
load_orca_checkpoint(path, version)
```
* path: checkpoint directory which contains model.* and optimMethod-TFParkTraining.* files.
* version: checkpoint version, which is the suffix of model.* file, i.e., for modle.4 file, the version is 4.

After loading checkpoint, you can resume training with fit(). 

#### Tensorboard support

During training and validation, Orca TF Estimator would save Tensorflow summary data under `model_dir` which is specified when you create estimator. This data can be visualized in TensorBoard, or you can use estimator's APIs to retrieve it. If you want to save train/validation summary data to different directory or you don't specify `model_dir`, you can set the logdir and app name with such API:
```
set_tensorboard(log_dir, app_name)
```
* `log_dir`: The base directory path to store training and validation logs.
* `app_name`: The name of the application.

This method sets summary information during the training process for visualization purposes. Saved summary can be viewed via TensorBoard. In order to take effect, it needs to be called before fit.

Training summary will be saved to 'log_dir/app_name/train' and validation summary (if any) will be saved to 'log_dir/app_name/validation'.

Now, you can see the summary data in TensorBoard. Or else, you can get summary data with such APIs:
```
get_train_summary(tag)
```
* `tag`: The string variable represents the scalar wanted. It can only be "Loss", "LearningRate", or "Throughput".

This method get the scalar from model train summary. Return list of summary data of [iteration_number, scalar_value, timestamp].

```
get_validation_summary(tag)
```
* `tag`: The string variable represents the scalar wanted.

This method gets the scalar from model validation summary. Return list of summary data of [iteration_number, scalar_value, timestamp]

#### Save model
After training, you can save model in the estimator with such APIs:
```
save_tf_checkpoint(path)
```
* `path`: tensorflow checkpoint path.

If you use tensorflow graph model in this estimator, this method would save tensorflow checkpoint.
```
save_keras_model(path)
```
* `path`: keras model save path.

If you use tensorflow keras model in this estimator, this method would save keras model in specified path.






