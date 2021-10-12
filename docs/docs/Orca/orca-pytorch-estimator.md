---
## **Introduction**

Analytics Zoo Orca PyTorch Estimator provides a set APIs for running PyTorch model on Spark in a distributed fashion.

__Remarks__:

- You need to install __torch==1.5.0__ and __torchvision==0.6.0__ on your driver node.
- Your operating system (OS) is required to be one of the following 64-bit systems:
__Ubuntu 16.04 or later__ and __macOS 10.12.6 or later__.
---

### Orca PyTorch Estimator

Orca PyTorch Estimator is an estimator to do PyTorch training/evaluation/prediction on Spark in a distributed fashion.

It can support various data types, like XShards, PyTorch DataLoader, PyTorch DataLoader creator, etc.

It supports horovod backend and BigDL backend in the unified APIs.

### Create Estimator from pyTorch Model

You can create Orca PyTorch Estimator with native PyTorch model.

```
from zoo.orca.learn.pytorch import Estimator
Estimator.from_torch(*,
                   model,
                   optimizer,
                   loss=None,
                   metrics=None,
                   scheduler_creator=None,
                   training_operator_cls=TrainingOperator,
                   initialization_hook=None,
                   config=None,
                   scheduler_step_freq="batch",
                   use_tqdm=False,
                   workers_per_node=1,
                   model_dir=None,
                   backend="bigdl"):
```
* `model`: PyTorch model if `backend="bigdl"`, PyTorch model creator if `backend="horovod" or "torch_distributed"`
* `optimizer`: Orca optimizer or PyTorch optimizer if `backend="bigdl"`, PyTorch optimizer creator if `backend="horovod" or "torch_distributed"`
* `loss`: PyTorch loss if `backend="bigdl"`, PyTorch loss creator if `backend="horovod" or "torch_distributed"`
* `metrics`: Orca validation methods for evaluate.
* `scheduler_creator`: parameter for `horovod` and `torch_distributed` backends. a learning rate scheduler wrapping the optimizer. You will need to set ``scheduler_step_freq="epoch"`` for the scheduler to be incremented correctly.
* `training_operator_cls`: parameter for `horovod` and `torch_distributed` backends. Custom training operator class that subclasses the TrainingOperator class. This class will be copied onto all remote workers and used to specify custom training and validation operations. Defaults to TrainingOperator.
* `initialization_hook`: parameter for `horovod` and `torch_distributed` backends.
* `config`: parameter for `horovod` and `torch_distributed` backends. Config dict to create model, optimizer loss and data.
* `scheduler_step_freq`: parameter for `horovod` and `torch_distributed` backends. "batch", "epoch" or None. This will determine when ``scheduler.step`` is called. If "batch", ``step`` will be called after every optimizer step. If "epoch", ``step`` will be called after one pass of the DataLoader. If a scheduler is passed in, this value is expected to not be None.
* `use_tqdm`: parameter for `horovod` and `torch_distributed` backends. You can monitor training progress if use_tqdm=True.
* `workers_per_node`: parameter for `horovod` and `torch_distributed` backends. worker number on each node. default: 1.
* `model_dir`: parameter for `bigdl` backend. The path to save model. During the training, if checkpoint_trigger is defined and triggered, the model will be saved to model_dir.
* `backend`: You can choose "horovod",  "torch_distributed" or "bigdl" as backend. Default: bigdl.

### Use horovod Estimator
#### **Train model**
After an Estimator is created, you can call estimator API to train PyTorch model:
```
fit(self, data, epochs=1, profile=False, reduce_results=True, info=None)
```
* `data`: (callable) a funtion that takes a config dict and `batch_size` as input and return a data loader containing the training data.
* `epochs`: (int) Number of epochs to train the model
* `profile`: (bool) Returns time stats for the training procedure.
* `reduce_results`: (bool) Whether to average all metrics across all workers into one dict. If a metric is a non-numerical value (or nested dictionaries), one value will be randomly selected among the workers. If False, returns a list of dicts.
* `info`: (dict) Optional dictionary passed to the training operator for ``train_epoch`` and ``train_batch``.

#### **Evaluate model**
After Training, you can call estimator API to evaluate PyTorch model:
```
evaluate(self, data, num_steps=None, profile=False, info=None)
```
* `data`: (callable) a funtion that takes a config dict and `batch_size` as input and return a data loader containing the validation data.
* `num_steps`: (int) Number of batches to compute update steps on. This corresponds also to the number of times ``TrainingOperator.validate_batch`` is called.
* `profile`: (bool) Returns time stats for the evaluation procedure.
* `info`: (dict) Optional dictionary passed to the training operator for `validate` and `validate_batch`.

#### **Get model**
You can get the trained model using `get_model(self)`

#### **Save model**
You can save model using `save(self, model_path)`
* `model_path`: (str) Path to save the model.

#### **Load model**
You can load an exsiting model saved by `save(self, model_path)` using `load(self, model_path)`
* `model_path`: (str) Path to the existing model.

#### **Shutdown workers**
You can shut down workers and releases resources using `shutdown(self, force=False)`

### Use BigDL Estimator

#### **Train model**
After an Estimator is created, you can call estimator API to train PyTorch model:
```
fit(self, data, epochs=1, batch_size=32, feature_cols=None, label_cols=None, validation_data=None, checkpoint_trigger=None)
```
* `data`: Training data. SparkXShard, PyTorch DataLoader and PyTorch DataLoader creator are supported.
* `epochs`: Number of epochs to train the model.Default: 32.
* `batch_size`: Batch size used for training. Only used when data is a SparkXShard.
* `feature_cols`: Feature column name(s) of data. Only used when data is a Spark DataFrame. Default: None.
* `label_cols`: Label column name(s) of data. Only used when data is a Spark DataFrame. Default: None.
* `validation_data`: Validation data. SparkXShard, PyTorch DataLoader and PyTorch DataLoader creator are supported.
* `checkpoint_trigger`: Orca Trigger to set a checkpoint.

#### **Evaluate model**
After Training, you can call estimator API to evaluate PyTorch model:
```
evaluate(self, data, batch_size=32, feature_cols=None, label_cols=None)
```
* `data`: Validation data. SparkXShard, PyTorch DataLoader and PyTorch DataLoader creator are supported.
* `batch_size`: Batch size used for evaluation. Only used when data is a SparkXShard.
* `feature_cols`: (Not supported yet) Feature column name(s) of data. Only used when data is a Spark DataFrame. Default: None.
* `label_cols`: (Not supported yet) Label column name(s) of data. Only used when data is a Spark DataFrame. Default: None.

#### **Inference**
After training or loading trained model, you can call estimator API to inference:
```
predict(self, data, batch_size=4, feature_cols=None)
```
* `data`: Inference data. Only SparkXShards is supported.
* `batch_size`: Batch size used for inference.
* `feature_cols`: Feature column name(s) of data. Only used when data is a Spark DataFrame. Default: None.

#### **Get model**
You can get model using `get_model(self)`

#### **Save model**
You can save model using `save(self, model_path)`
* `model_path`: (str) Path to save the model.

#### **Load model**
You can load an exsiting model saved by `save(self, model_path)` using `load(self, model_path)`
* `model_path`: (str) Path to the existing model.

#### **Load orca checkpoint**
You can load saved orca checkpoint using `load_orca_checkpoint(self, path, version, prefix)`. To load a specific checkpoint, 
please provide both `version` and `perfix`. If `version` is None, then the latest checkpoint will be loaded.
* `path`: Path to the existing checkpoint (or directory containing Orca checkpoint files if `version` is None).
* `version`: checkpoint version, which is the suffix of model.* file, i.e., for
               modle.4 file, the version is 4. If it is `None`, then load the latest checkpoint.
* `prefix`: optimMethod prefix, for example 'optimMethod-TorchModelf53bddcc'. If loading the latest checkpoint, just leave it as `None`.

#### **Clear gradient clipping**
You can clear gradient clipping parameters using `clear_gradient_clipping(self)`. In this case, gradient clipping will not be applied.
**Note:** In order to take effect, it needs to be called before fit.

#### **Set constant gradient clipping**
You can Set constant gradient clipping during the training process using `set_constant_gradient_clipping(self, min, max)`.
* `min`: The minimum value to clip by.
* `max`: The maximum value to clip by.
**Note:** In order to take effect, it needs to be called before fit.

#### **Set clip gradient to a maximum L2-Norm**
You can set clip gradient to a maximum L2-Norm during the training process using `set_l2_norm_gradient_clipping(self, clip_norm)`.
* `clip_norm`: Gradient L2-Norm threshold.
**Note:** In order to take effect, it needs to be called before fit.



