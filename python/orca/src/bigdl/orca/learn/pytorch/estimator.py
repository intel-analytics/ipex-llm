#
# Copyright 2018 Analytics Zoo Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from zoo.common.utils import enable_multi_fs_load, enable_multi_fs_save
from zoo.orca.data.utils import row_to_sample, xshard_to_sample
from zoo.orca.learn.utils import convert_predict_rdd_to_dataframe, bigdl_metric_results_to_dict, \
    process_xshards_of_pandas_dataframe
from zoo.pipeline.estimator.estimator import Estimator as SparkEstimator
from zoo.orca.learn.ray_estimator import Estimator as OrcaRayEstimator
from zoo.orca.learn.pytorch.training_operator import TrainingOperator
from zoo.orca.learn.spark_estimator import Estimator as OrcaSparkEstimator
from zoo.orca.learn.optimizers import Optimizer as OrcaOptimizer, SGD
from zoo.orca.learn.metrics import Accuracy
from zoo.orca.data import SparkXShards
from bigdl.optim.optimizer import MaxEpoch, OptimMethod
from zoo.feature.common import FeatureSet
from torch.optim.optimizer import Optimizer as TorchOptimizer
from torch.utils.data import DataLoader
from pyspark.sql import DataFrame
import warnings
import torch
import types


class Estimator(object):
    @staticmethod
    def from_torch(*,
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
        """
        Create an Estimator for torch.

        :param model: PyTorch model or model creator function if backend="bigdl", PyTorch
               model creator function if backend="horovod" or "torch_distributed"
        :param optimizer: Orca/PyTorch optimizer or optimizer creator function if backend="bigdl"
               , PyTorch optimizer creator function if backend="horovod" or "torch_distributed"
        :param loss: PyTorch loss or loss creator function if backend="bigdl", PyTorch loss creator
               function if backend="horovod" or "torch_distributed"
        :param metrics: Orca validation methods for evaluate.
        :param scheduler_creator: parameter for `horovod` and `torch_distributed` backends. a
               learning rate scheduler wrapping the optimizer. You will need to set
               ``scheduler_step_freq="epoch"`` for the scheduler to be incremented correctly.
        :param config: parameter config dict to create model, optimizer loss and data.
        :param scheduler_step_freq: parameter for `horovod` and `torch_distributed` backends.
               "batch", "epoch" or None. This will determine when ``scheduler.step`` is called. If
               "batch", ``step`` will be called after every optimizer step. If "epoch", ``step``
               will be called after one pass of the DataLoader. If a scheduler is passed in, this
               value is expected to not be None.
        :param use_tqdm: parameter for `horovod` and `torch_distributed` backends. You can monitor
               training progress if use_tqdm=True.
        :param workers_per_node: parameter for `horovod` and `torch_distributed` backends. worker
               number on each node. default: 1.
        :param model_dir: parameter for `bigdl` backend. The path to save model. During the
               training, if checkpoint_trigger is defined and triggered, the model will be saved to
               model_dir.
        :param backend: You can choose "horovod",  "torch_distributed" or "bigdl" as backend.
               Default: `bigdl`.
        :return: an Estimator object.
        """
        if backend in {"horovod", "torch_distributed"}:
            return PyTorchRayEstimator(model_creator=model,
                                       optimizer_creator=optimizer,
                                       loss_creator=loss,
                                       metrics=metrics,
                                       scheduler_creator=scheduler_creator,
                                       training_operator_cls=training_operator_cls,
                                       initialization_hook=initialization_hook,
                                       config=config,
                                       scheduler_step_freq=scheduler_step_freq,
                                       use_tqdm=use_tqdm,
                                       workers_per_node=workers_per_node,
                                       backend=backend)
        elif backend == "bigdl":
            return PyTorchSparkEstimator(model=model,
                                         loss=loss,
                                         optimizer=optimizer,
                                         config=config,
                                         metrics=metrics,
                                         model_dir=model_dir,
                                         bigdl_type="float")
        else:
            raise ValueError("Only horovod, torch_distributed and bigdl backends are supported"
                             f" for now, got backend: {backend}")


class PyTorchRayEstimator(OrcaRayEstimator):
    def __init__(self,
                 *,
                 model_creator,
                 optimizer_creator,
                 loss_creator=None,
                 metrics=None,
                 scheduler_creator=None,
                 training_operator_cls=TrainingOperator,
                 initialization_hook=None,
                 config=None,
                 scheduler_step_freq="batch",
                 use_tqdm=False,
                 backend="torch_distributed",
                 workers_per_node=1):
        if config is not None and "batch_size" in config:
            raise Exception("Please do not specify batch_size in config. Input batch_size in the"
                            " fit/evaluate/predict function of the estimator instead.")

        from zoo.orca.learn.pytorch.pytorch_ray_estimator import PyTorchRayEstimator
        self.estimator = PyTorchRayEstimator(model_creator=model_creator,
                                             optimizer_creator=optimizer_creator,
                                             loss_creator=loss_creator,
                                             metrics=metrics,
                                             scheduler_creator=scheduler_creator,
                                             training_operator_cls=training_operator_cls,
                                             initialization_hook=initialization_hook,
                                             config=config,
                                             scheduler_step_freq=scheduler_step_freq,
                                             use_tqdm=use_tqdm,
                                             backend=backend,
                                             workers_per_node=workers_per_node)

    def fit(self, data, epochs=1, batch_size=32, profile=False, reduce_results=True, info=None,
            feature_cols=None, label_cols=None):
        """
        Trains a PyTorch model given training data for several epochs.
        Calls `TrainingOperator.train_epoch()` on N parallel workers simultaneously
        underneath the hood.

        :param data: An instance of SparkXShards, a Spark DataFrame or a function that
               takes config and batch_size as argument and returns a PyTorch DataLoader for
               training.
        :param epochs: The number of epochs to train the model. Default is 1.
        :param batch_size: The number of samples per batch for each worker. Default is 32.
               The total batch size would be workers_per_node*num_nodes.
               If your training data is a function, you can set batch_size to be the input
               batch_size of the function for the PyTorch DataLoader.
        :param profile: Boolean. Whether to return time stats for the training procedure.
               Default is False.
        :param reduce_results: Boolean. Whether to average all metrics across all workers into
               one dict. If a metric is a non-numerical value (or nested dictionaries), one value
               will be randomly selected among the workers. If False, returns a list of dicts for
               all workers.
               Default is True.
        :param info: An optional dictionary that can be passed to the TrainingOperator for
               train_epoch and train_batch.
        :param feature_cols: feature column names if data is Spark DataFrame.
        :param label_cols: label column names if data is Spark DataFrame.

        :return: A list of dictionary of metrics for every training epoch. If reduce_results is
                False, this will return a nested list of metric dictionaries whose length will be
                equal to the total number of workers.
                You can also provide custom metrics by passing in a custom training_operator_cls
                when creating the Estimator.
        """
        return self.estimator.train(data=data, epochs=epochs, batch_size=batch_size,
                                    profile=profile, reduce_results=reduce_results,
                                    info=info, feature_cols=feature_cols,
                                    label_cols=label_cols)

    def predict(self, data, batch_size=32, feature_cols=None, profile=False):
        """
        Using this PyTorch model to make predictions on the data.

        :param data: An instance of SparkXShards or a Spark DataFrame
        :param batch_size: The number of samples per batch for each worker. Default is 32.
        :param profile: Boolean. Whether to return time stats for the training procedure.
               Default is False.
        :param feature_cols: feature column names if data is a Spark DataFrame.
        :return: A SparkXShards that contains the predictions with key "prediction" in each shard
        """
        return self.estimator.predict(data, batch_size=batch_size,
                                      feature_cols=feature_cols,
                                      profile=profile)

    def evaluate(self, data, batch_size=32, num_steps=None, profile=False, info=None,
                 feature_cols=None, label_cols=None):
        """
        Evaluates a PyTorch model given validation data.
        Note that only accuracy for classification with zero-based label is supported by
        default. You can override validate_batch in TrainingOperator for other metrics.
        Calls `TrainingOperator.validate()` on N parallel workers simultaneously
        underneath the hood.

        :param data: An instance of SparkXShards, a Spark DataFrame or a function that
               takes config and batch_size as argument and returns a PyTorch DataLoader for
               validation.
        :param batch_size: The number of samples per batch for each worker. Default is 32.
               The total batch size would be workers_per_node*num_nodes.
               If your validation data is a function, you can set batch_size to be the input
               batch_size of the function for the PyTorch DataLoader.
        :param num_steps: The number of batches to compute the validation results on. This
               corresponds to the number of times `TrainingOperator.validate_batch` is called.
        :param profile: Boolean. Whether to return time stats for the training procedure.
               Default is False.
        :param info: An optional dictionary that can be passed to the TrainingOperator
               for validate.
        :param feature_cols: feature column names if train data is Spark DataFrame.
        :param label_cols: label column names if train data is Spark DataFrame.

        :return: A dictionary of metrics for the given data, including validation accuracy and loss.
                You can also provide custom metrics by passing in a custom training_operator_cls
                when creating the Estimator.
        """
        return self.estimator.validate(data=data, batch_size=batch_size, num_steps=num_steps,
                                       profile=profile, info=info, feature_cols=feature_cols,
                                       label_cols=label_cols)

    def get_model(self):
        """
        Returns the learned PyTorch model.

        :return: The learned PyTorch model.
        """
        return self.estimator.get_model()

    @enable_multi_fs_save
    def save(self, model_path):
        """
        Saves the Estimator state (including model and optimizer) to the provided model_path.

        :param model_path: (str) Path to save the model.
        :return:
        """
        return self.estimator.save(model_path)

    @enable_multi_fs_load
    def load(self, model_path):
        """
        Loads the Estimator state (including model and optimizer) from the provided model_path.

        :param model_path: (str) Path to the existing model.
        """
        return self.estimator.load(model_path)

    def shutdown(self, force=False):
        """
        Shuts down workers and releases resources.

        :return:
        """
        return self.estimator.shutdown(force=force)


class PyTorchSparkEstimator(OrcaSparkEstimator):
    def __init__(self, model, loss, optimizer, config=None, metrics=None, model_dir=None,
                 bigdl_type="float"):
        from zoo.pipeline.api.torch import TorchModel, TorchLoss, TorchOptim
        self.loss = loss
        self.optimizer = optimizer
        self.config = {} if config is None else config

        if self.loss is None:
            self.loss = TorchLoss()
        else:
            self.loss = TorchLoss.from_pytorch(loss)
        if isinstance(model, types.FunctionType):
            def model_creator(self):
                return model(self.config)
            model = model_creator(self)
        if self.optimizer is None:
            from zoo.orca.learn.optimizers.schedule import Default
            self.optimizer = SGD(learningrate_schedule=Default()).get_optimizer()
        elif isinstance(self.optimizer, TorchOptimizer):
            self.optimizer = TorchOptim.from_pytorch(self.optimizer)
        elif isinstance(self.optimizer, OrcaOptimizer):
            self.optimizer = self.optimizer.get_optimizer()
        else:
            raise ValueError("Only PyTorch optimizer and orca optimizer are supported")
        from zoo.orca.learn.metrics import Metric
        self.metrics = Metric.convert_metrics_list(metrics)
        self.log_dir = None
        self.app_name = None
        self.model_dir = model_dir
        self.model = TorchModel.from_pytorch(model)
        self.estimator = SparkEstimator(self.model, self.optimizer, model_dir,
                                        bigdl_type=bigdl_type)

    def _handle_dataframe(self, data, validation_data, feature_cols, label_cols):
        schema = data.schema
        train_rdd = data.rdd.map(lambda row: row_to_sample(row, schema, feature_cols, label_cols))
        train_feature_set = FeatureSet.sample_rdd(train_rdd)
        if validation_data is None:
            val_feature_set = None
        else:
            assert isinstance(validation_data, DataFrame), "validation_data should also be a " \
                                                           "DataFrame"
            val_feature_set = FeatureSet.sample_rdd(validation_data.rdd.map(
                lambda row: row_to_sample(row, schema, feature_cols, label_cols)))

        return train_feature_set, val_feature_set

    def _handle_xshards(self, data, validation_data):
        train_rdd = data.rdd.flatMap(xshard_to_sample)
        train_feature_set = FeatureSet.sample_rdd(train_rdd)
        if validation_data is None:
            val_feature_set = None
        else:
            assert isinstance(validation_data, SparkXShards), "validation_data should be a " \
                                                              "SparkXShards"
            val_feature_set = FeatureSet.sample_rdd(validation_data.rdd.flatMap(xshard_to_sample))
        return train_feature_set, val_feature_set

    def _handle_data_loader(self, data, validation_data):
        train_feature_set = FeatureSet.pytorch_dataloader(data, "", "")
        if validation_data is None:
            val_feature_set = None
        else:
            assert isinstance(validation_data, DataLoader) or callable(data), \
                "validation_data should be a pytorch DataLoader or a callable data_creator"
            val_feature_set = FeatureSet.pytorch_dataloader(validation_data)

        return train_feature_set, val_feature_set

    def fit(self, data, epochs=1, batch_size=None, feature_cols=None, label_cols=None,
            validation_data=None, checkpoint_trigger=None):
        """
        Train this torch model with train data.

        :param data: train data. It can be a XShards, Spark Dataframe, PyTorch DataLoader and
               PyTorch DataLoader creator function that takes config and batch_size as argument and
               returns a PyTorch DataLoader for training.
               If data is an XShards, each partition can be a Pandas DataFrame or a dictionary of
               {'x': feature, 'y': label}, where feature(label) is a numpy array or
               a list of numpy arrays.
        :param epochs: Number of epochs to train the model. Default: 1.
        :param batch_size: Batch size used for training. Only used when data is an XShards.
               Default: 32.
        :param feature_cols: Feature column name(s) of data. Only used when data
               is a Spark DataFrame or an XShards of Pandas DataFrame. Default: None.
        :param label_cols: Label column name(s) of data. Only used when data is
               a Spark DataFrame or an XShards of Pandas DataFrame. Default: None.
        :param validation_data: Validation data. XShards, PyTorch DataLoader and PyTorch DataLoader
               creator function are supported.
               If data is XShards, each partition can be a Pandas DataFrame or a dictionary of
               {'x': feature, 'y': label}, where feature(label) is a numpy array or a list of
               numpy arrays.
        :param checkpoint_trigger: Orca Trigger to set a checkpoint.
        :return: The trained estimator object.
        """
        from zoo.orca.learn.trigger import Trigger

        end_trigger = MaxEpoch(epochs)
        if isinstance(data, DataLoader):
            assert batch_size is None and data.batch_size > 0, "When using PyTorch Dataloader as " \
                                                               "input, you need to specify the " \
                                                               "batch size in DataLoader and " \
                                                               "don't specify batch_size " \
                                                               "in the fit method."
        else:
            assert batch_size is not None and batch_size > 0, "batch_size should be greater than 0"
        checkpoint_trigger = Trigger.convert_trigger(checkpoint_trigger)

        if self.log_dir is not None and self.app_name is not None:
            self.estimator.set_tensorboard(self.log_dir, self.app_name)

        if validation_data:
            assert self.metrics is not None, "You should provide metrics when creating this " \
                                             "estimator if you provide validation_data."

        if isinstance(data, SparkXShards):
            if data._get_class_name() == 'pandas.core.frame.DataFrame':
                data, validation_data = process_xshards_of_pandas_dataframe(data, feature_cols,
                                                                            label_cols,
                                                                            validation_data,
                                                                            mode="fit")
            train_fset, val_fset = self._handle_xshards(data, validation_data)
            self.estimator.train(train_fset, self.loss, end_trigger, checkpoint_trigger,
                                 val_fset, self.metrics, batch_size)
        elif isinstance(data, DataFrame):
            train_fset, val_fset = self._handle_dataframe(data, validation_data,
                                                          feature_cols, label_cols)
            self.estimator.train(train_fset, self.loss, end_trigger, checkpoint_trigger,
                                 val_fset, self.metrics, batch_size)
        elif isinstance(data, DataLoader) or callable(data) or isinstance(data, types.FunctionType):
            if isinstance(data, types.FunctionType):
                data, validation_data = data(self.config, batch_size), validation_data(self.config,
                                                                                       batch_size)
            train_fset, val_fset = self._handle_data_loader(data, validation_data)
            self.estimator.train_minibatch(train_fset, self.loss, end_trigger,
                                           checkpoint_trigger, val_fset, self.metrics)
        else:
            raise ValueError("Data and validation data should be SparkXShards, DataLoaders or "
                             "callable data_creators but get " + data.__class__.__name__)

        return self

    def predict(self, data, batch_size=4, feature_cols=None):
        """
        Predict input data.

        :param data: data to be predicted. It can be an XShards or a Spark Dataframe.
               If it is an XShards, each partition can be a Pandas DataFrame or a dictionary of
               {'x': feature}, where feature is a numpy array or a list of numpy arrays.
        :param batch_size: batch size used for inference.
        :param feature_cols: Feature column name(s) of data. Only used when data
               is a Spark DataFrame or an XShards of Pandas DataFrame. Default: None.
        :return: predicted result. The predict result is a XShards, each partition of the XShards
                 is a dictionary of {'prediction': result}, where result is a numpy array or a list
                 of numpy arrays.
        """
        from zoo.orca.learn.utils import convert_predict_rdd_to_xshard
        if isinstance(data, SparkXShards):
            if data._get_class_name() == 'pandas.core.frame.DataFrame':
                data = process_xshards_of_pandas_dataframe(data, feature_cols)
            from zoo.orca.data.utils import xshard_to_sample
            data_rdd = data.rdd.flatMap(xshard_to_sample)

        elif isinstance(data, DataFrame):
            schema = data.schema
            data_rdd = data.rdd.map(lambda row: row_to_sample(row, schema,
                                                              feature_cols, None))
        else:
            raise ValueError("Data should be XShards, each element needs to be {'x': a feature "
                             "numpy array}.")
        predicted_rdd = self.model.predict(data_rdd, batch_size=batch_size)

        if isinstance(data, SparkXShards):
            result = convert_predict_rdd_to_xshard(data, predicted_rdd)
        else:
            result = convert_predict_rdd_to_dataframe(data, predicted_rdd)
        return result

    def evaluate(self, data, batch_size=None, feature_cols=None, label_cols=None,
                 validation_metrics=None):
        """
        Evaluate model.

        :param data: data: evaluation data. It can be an XShards, Spark Dataframe,
               PyTorch DataLoader and PyTorch DataLoader creator function.
               If data is an XShards, each partition can be a Pandas DataFrame or a dictionary of
               {'x': feature, 'y': label}, where feature(label) is a numpy array or a list of
               numpy arrays.
        :param batch_size: Batch size used for evaluation. Only used when data is a SparkXShard.
        :param feature_cols: Feature column name(s) of data. Only used when data
               is a Spark DataFrame or an XShards of Pandas DataFrame. Default: None.
        :param label_cols: Label column name(s) of data. Only used when data is
               a Spark DataFrame or an XShards of Pandas DataFrame. Default: None.
        :param validation_metrics: Orca validation metrics to be computed on validation_data.
        :return: validation results.
        """
        from zoo.orca.data.utils import xshard_to_sample

        assert data is not None, "validation data shouldn't be None"
        assert self.metrics is not None, "metrics shouldn't be None, please specify the metrics" \
                                         " argument when creating this estimator."
        if isinstance(data, DataLoader):
            assert batch_size is None and data.batch_size > 0, "When using PyTorch Dataloader as " \
                                                               "input, you need to specify the " \
                                                               "batch size in DataLoader and " \
                                                               "don't specify batch_size " \
                                                               "in the fit method."
        else:
            assert batch_size is not None and batch_size > 0, "batch_size should be greater than 0"

        if isinstance(data, SparkXShards):
            if data._get_class_name() == 'pandas.core.frame.DataFrame':
                data = process_xshards_of_pandas_dataframe(data, feature_cols, label_cols)
            val_feature_set = FeatureSet.sample_rdd(data.rdd.flatMap(xshard_to_sample))
            result = self.estimator.evaluate(val_feature_set, self.metrics, batch_size)
        elif isinstance(data, DataFrame):
            schema = data.schema
            val_feature_set = FeatureSet.sample_rdd(data.rdd.map(
                lambda row: row_to_sample(row, schema, feature_cols, label_cols)))
            result = self.estimator.evaluate(val_feature_set, self.metrics, batch_size)
        elif isinstance(data, DataLoader) or callable(data) or isinstance(data, types.FunctionType):
            if isinstance(data, types.FunctionType):
                data = data(self.config, batch_size)
            val_feature_set = FeatureSet.pytorch_dataloader(data)
            result = self.estimator.evaluate_minibatch(val_feature_set, self.metrics)
        else:
            raise ValueError("Data should be a SparkXShards, a DataLoader or a callable "
                             "data_creator, but get " + data.__class__.__name__)
        return bigdl_metric_results_to_dict(result)

    def get_model(self):
        """
        Get the trained PyTorch model.

        :return: The trained PyTorch model.
        """
        return self.model.to_pytorch()

    def _get_optimizer_path(self, model_path):
        if "." in model_path:
            path_split = model_path.rsplit('.', 1)
            return path_split[0] + "_optim." + path_split[1]
        else:
            return model_path + "_optim"

    @enable_multi_fs_save
    def save(self, model_path):
        """
        Saves the Estimator state (including model and optimizer) to the provided model_path.

        :param model_path: path to save the model.
        :return: model_path
        """

        optim_path = self._get_optimizer_path(model_path)
        torch.save(self.get_model().state_dict(), model_path)
        if self.optimizer is not None:
            self.optimizer.save(path=optim_path, overWrite=True)

        return model_path

    @enable_multi_fs_load
    def load(self, model_path):
        """
        Load the Estimator state (model and possibly with optimizer) from provided model_path.
        The model file should be generated by the save method of this estimator, or by
        ``torch.save(state_dict, model_path)``, where `state_dict` can be obtained by
        the ``state_dict()`` method of a pytorch model.

        :param model_path: path to the saved model.
        :return:
        """

        from zoo.pipeline.api.torch import TorchModel
        import os

        try:
            pytorch_model = self.get_model()
            pytorch_model.load_state_dict(torch.load(model_path))
            self.model = TorchModel.from_pytorch(pytorch_model)
        except Exception:
            raise ValueError("Cannot load the PyTorch model. Please check your model path.")

        optim_path = self._get_optimizer_path(model_path)
        if os.path.isfile(optim_path):
            try:
                self.optimizer = OptimMethod.load(optim_path)
            except Exception:
                raise ValueError("Cannot load the optimizer. Only `bigdl.optim.optimizer."
                                 "OptimMethod` is supported for loading.")
        else:
            self.optimizer = None

        self.estimator = SparkEstimator(self.model, self.optimizer, self.model_dir)

    def load_orca_checkpoint(self, path, version=None, prefix=None):
        """
        Load existing checkpoint. To load a specific checkpoint, please provide both `version` and
        `perfix`. If `version` is None, then the latest checkpoint will be loaded.

        :param path: Path to the existing checkpoint (or directory containing Orca checkpoint
               files).
        :param version: checkpoint version, which is the suffix of model.* file, i.e., for
               modle.4 file, the version is 4. If it is None, then load the latest checkpoint.
        :param prefix: optimMethod prefix, for example 'optimMethod-TorchModelf53bddcc'.
        :return:
        """
        import os
        from bigdl.nn.layer import Model
        from bigdl.optim.optimizer import OptimMethod
        from zoo.orca.learn.utils import find_latest_checkpoint
        from zoo.pipeline.api.torch import TorchModel

        if version is None:
            path, prefix, version = find_latest_checkpoint(path, model_type="pytorch")
            if path is None:
                raise ValueError("Cannot find PyTorch checkpoint, please check your checkpoint"
                                 " path.")
        else:
            assert prefix is not None, "You should provide optimMethod prefix, " \
                                       "for example 'optimMethod-TorchModelf53bddcc'"

        try:
            loaded_model = Model.load(os.path.join(path, "model.{}".format(version)))
            self.model = TorchModel.from_value(loaded_model.value)
            self.optimizer = OptimMethod.load(os.path.join(path, "{}.{}".format(prefix, version)))
        except Exception as e:
            raise ValueError("Cannot load PyTorch checkpoint, please check your checkpoint path "
                             "and checkpoint type." + str(e))
        self.estimator = SparkEstimator(self.model, self.optimizer, self.model_dir)

    def get_train_summary(self, tag=None):
        """
        Get the scalar from model train summary.

        This method will return a list of summary data of
        [iteration_number, scalar_value, timestamp].

        :param tag: The string variable represents the scalar wanted
        """
        return self.estimator.get_train_summary(tag=tag)

    def get_validation_summary(self, tag=None):
        """
        Get the scalar from model validation summary.

        This method will return a list of summary data of
        [iteration_number, scalar_value, timestamp].
        Note that the metric and tag may not be consistent.
        Please look up following form to pass tag parameter.
        Left side is your metric during compile.
        Right side is the tag you should pass.

        >>> 'Accuracy'                  |   'Top1Accuracy'
        >>> 'BinaryAccuracy'            |   'Top1Accuracy'
        >>> 'CategoricalAccuracy'       |   'Top1Accuracy'
        >>> 'SparseCategoricalAccuracy' |   'Top1Accuracy'
        >>> 'AUC'                       |   'AucScore'
        >>> 'HitRatio'                  |   'HitRate@k' (k is Top-k)
        >>> 'Loss'                      |   'Loss'
        >>> 'MAE'                       |   'MAE'
        >>> 'NDCG'                      |   'NDCG'
        >>> 'TFValidationMethod'        |   '${name + " " + valMethod.toString()}'
        >>> 'Top5Accuracy'              |   'Top5Accuracy'
        >>> 'TreeNNAccuracy'            |   'TreeNNAccuracy()'
        >>> 'MeanAveragePrecision'      |   'MAP@k' (k is Top-k) (BigDL)
        >>> 'MeanAveragePrecision'      |   'PascalMeanAveragePrecision' (Zoo)
        >>> 'StatelessMetric'           |   '${name}'

        :param tag: The string variable represents the scalar wanted
        """
        return self.estimator.get_validation_summary(tag=tag)

    def clear_gradient_clipping(self):
        """
        Clear gradient clipping parameters. In this case, gradient clipping will not be applied.
        In order to take effect, it needs to be called before fit.

        :return:
        """
        self.estimator.clear_gradient_clipping()

    def set_constant_gradient_clipping(self, min, max):
        """
        Set constant gradient clipping during the training process.
        In order to take effect, it needs to be called before fit.

        :param min: The minimum value to clip by.
        :param max: The maximum value to clip by.
        :return:
        """
        self.estimator.set_constant_gradient_clipping(min=min, max=max)

    def set_l2_norm_gradient_clipping(self, clip_norm):
        """
        Clip gradient to a maximum L2-Norm during the training process.
        In order to take effect, it needs to be called before fit.

        :param clip_norm: Gradient L2-Norm threshold.
        :return:
        """
        self.estimator.set_l2_norm_gradient_clipping(clip_norm=clip_norm)
