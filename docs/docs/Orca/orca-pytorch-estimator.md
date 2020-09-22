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
                   scheduler_creator=None,
                   training_operator_cls=TrainingOperator,
                   initialization_hook=None,
                   config=None,
                   scheduler_step_freq="batch",
                   use_tqdm=False,
                   workers_per_node=1,
                   model_dir=None,
                   backend="horovod"):
```
* `model`: PyTorch model if `backend="bigdl"`, PyTorch model creator if `backend="horovod"`
* `optimizer`: bigdl optimizer if `backend="bigdl"`, PyTorch optimizer creator if `backend="horovod"`
* `loss`: PyTorch loss if `backend="bigdl"`, PyTorch loss creator if `backend="horovod"`
* `scheduler_creator`: parameter for horovod. a learning rate scheduler wrapping the optimizer. You will need to set ``TorchTrainer(scheduler_step_freq="epoch")`` for the scheduler to be incremented correctly. If using a scheduler for validation loss, be sure to call ``trainer.update_scheduler(validation_loss)``
* `training_operator_cls`: parameter for horovod. Custom training operator class that subclasses the TrainingOperator class. This class will be copied onto all remote workers and used to specify custom training and validation operations. Defaults to TrainingOperator.
* `initialization_hook`: parameter for horovod.
* `config`: parameter for horovod. Config dict to create model, optimizer loss and data.
* `scheduler_step_freq`: parameter for horovod. "batch", "epoch", "manual", or None. This will determine when ``scheduler.step`` is called. If "batch", ``step`` will be called after every optimizer step. If "epoch", ``step`` will be called after one pass of the DataLoader. If "manual", the scheduler will not be incremented automatically - you are expected to call ``trainer.update_schedulers`` manually. If a scheduler is passed in, this value is expected to not be None.
* `use_tqdm`: parameter for horovod. You can monitor training progress if use_tqdm=True.
* `workers_per_node`: parameter for horovod. worker number on each node. default: 1.
* `model_dir`: parameter for `bigdl`. The path to save model. During the training, if checkpoint_trigger is defined and triggered, the model will be saved to model_dir.
* `backend`: You can choose "horovod" or "bigdl" as backend.

### Use horovod Estimator
#### **Train model**
After an Estimator is created, you can call estimator API to train PyTorch model:
```
fit(self, data, epochs=1, profile=False, reduce_results=True, info=None)
```
* `data`: (callable) a funtion that takes a config dict as input and return a data loader containing the training data.
* `epochs`: (int) Number of epochs to train the model
* `profile`: (bool) Returns time stats for the training procedure.
* `reduce_results`: (bool) Whether to average all metrics across all workers into one dict. If a metric is a non-numerical value (or nested dictionaries), one value will be randomly selected among the workers. If False, returns a list of dicts.
* `info`: (dict) Optional dictionary passed to the training operator for ``train_epoch`` and ``train_batch``.

#### **Evaluate model**
After Training, you can call estimator API to evaluate PyTorch model:
```
evaluate(self, data, num_steps=None, profile=False, info=None)
```
* `data`: (callable) a funtion that takes a config dict as input and return a data loader containing the validation data.
* `num_steps`: (int) Number of batches to compute update steps on. This corresponds also to the number of times ``TrainingOperator.validate_batch`` is called.
* `profile`: (bool) Returns time stats for the evaluation procedure.
* `info`: (dict) Optional dictionary passed to the training operator for `validate` and `validate_batch`.

#### **Get model**
You can get the trained model using `get_model(self)`

#### **Save model**
You can save model using `save(self, checkpoint)`
* `checkpoint`: (str) Path to target checkpoint file.

#### **Load model**
You can load saved model using `load(self, checkpoint)`
* `checkpoint`: (str) Path to target checkpoint file.

#### **Shutdown workers**
You can shut down workers and releases resources using `shutdown(self, force=False)`

### Use BigDL Estimator

#### **Train model**
After an Estimator is created, you can call estimator API to train PyTorch model:
```
fit(self, data, epochs=1, batch_size=32, validation_data=None, validation_methods=None, checkpoint_trigger=None):
```
* `data`: Training data. SparkXShard, PyTorch DataLoader and PyTorch DataLoader creator are supported.
* `epochs`: Number of epochs to train the model.
* `batch_size`: Batch size used for training. Only used when data is a SparkXShard.
* `validation_data`: Validation data. SparkXShard, PyTorch DataLoader and PyTorch DataLoader creator are supported.
* `validation_methods`: BigDL validation methods.
* `checkpoint_trigger`: BigDL Trigger to set a checkpoint.

#### **Evaluate model**
After Training, you can call estimator API to evaluate PyTorch model:
```
evaluate(self, data, validation_methods=None, batch_size=32)
```
* `data`: Validation data. SparkXShard, PyTorch DataLoader and PyTorch DataLoader creator are supported.
* `validation_methods`: BigDL validation methods.
* `batch_size`: Batch size used for evaluation. Only used when data is a SparkXShard.

#### **Get model**
You can get model using `get_model(self)`





