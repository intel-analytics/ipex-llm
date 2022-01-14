#
# Copyright 2016 The BigDL Authors.
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

from bigdl.chronos.forecaster.abstract import Forecaster
from bigdl.chronos.forecaster.utils import\
    np_to_creator, set_pytorch_seed, check_data, xshard_to_np, np_to_xshard, loader_to_creator
from bigdl.chronos.metric.forecast_metrics import Evaluator

import numpy as np
import warnings
import torch
from functools import partial
from torch.utils.data import TensorDataset, DataLoader


class BasePytorchForecaster(Forecaster):
    '''
    Forecaster base model for lstm, seq2seq and tcn forecasters.
    '''
    def __init__(self, **kwargs):
        if self.distributed:
            from bigdl.orca.learn.pytorch.estimator import Estimator
            from bigdl.orca.learn.metrics import MSE, MAE
            ORCA_METRICS = {"mse": MSE, "mae": MAE}

            def model_creator_orca(config):
                set_pytorch_seed(self.seed)
                model = self.model_creator({**self.model_config, **self.data_config})
                model.train()
                return model
            self.internal = Estimator.from_torch(model=model_creator_orca,
                                                 optimizer=self.optimizer_creator,
                                                 loss=self.loss_creator,
                                                 metrics=[ORCA_METRICS[name]()
                                                          for name in self.metrics],
                                                 backend=self.distributed_backend,
                                                 use_tqdm=True,
                                                 config={"lr": self.lr},
                                                 workers_per_node=self.workers_per_node)
        else:
            # seed setting
            from pytorch_lightning import seed_everything
            from bigdl.nano.pytorch.trainer import Trainer
            seed_everything(seed=self.seed)

            # Model preparation
            self.fitted = False
            model = self.model_creator({**self.model_config, **self.data_config})
            loss = self.loss_creator(self.loss_config)
            optimizer = self.optimizer_creator(model, self.optim_config)
            self.internal = Trainer.compile(model=model, loss=loss,
                                            optimizer=optimizer, onnx=self.onnx_available)

    def fit(self, data, epochs=1, batch_size=32):
        # TODO: give an option to close validation during fit to save time.
        """
        Fit(Train) the forecaster.

        :param data: The data support following formats:

               | 1. a numpy ndarray tuple (x, y):
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.
               | y's shape is (num_samples, horizon, target_dim), where horizon and target_dim
               | should be the same as future_seq_len and output_feature_num.
               |
               | 2. a xshard item:
               | each partition can be a dictionary of {'x': x, 'y': y}, where x and y's shape
               | should follow the shape stated before.
               |
               | 3. pytorch dataloader:
               | the dataloader should return x, y in each iteration with the shape as following:
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.
               | y's shape is (num_samples, horizon, target_dim), where horizon and target_dim
               | should be the same as future_seq_len and output_feature_num.

        :param epochs: Number of epochs you want to train. The value defaults to 1.
        :param batch_size: Number of batch size you want to train. The value defaults to 32.
               if you input a pytorch dataloader for `data`, the batch_size will follow the
               batch_size setted in `data`.if the forecaster is distributed, the batch_size will be
               evenly distributed to all workers.

        :return: Evaluation results on data.
        """
        # input transform
        if isinstance(data, DataLoader) and self.distributed:
            data = loader_to_creator(data)
        if isinstance(data, tuple) and self.distributed:
            data = np_to_creator(data)
        try:
            from bigdl.orca.data.shard import SparkXShards
            if isinstance(data, SparkXShards) and not self.distributed:
                warnings.warn("Xshards is collected to local since the "
                              "forecaster is non-distribued.")
                data = xshard_to_np(data)
        except ImportError:
            pass

        # fit on internal
        if self.distributed:
            # for cluster mode
            from bigdl.orca.common import OrcaContext
            sc = OrcaContext.get_spark_context().getConf()
            num_nodes = 1 if sc.get('spark.master').startswith('local') \
                else int(sc.get('spark.executor.instances'))
            if batch_size % self.workers_per_node != 0:
                raise RuntimeError("Please make sure that batch_size can be divisible by "
                                   "the product of worker_per_node and num_nodes, "
                                   f"but 'batch_size' is {batch_size}, 'workers_per_node' "
                                   f"is {self.workers_per_node}, 'num_nodes' is {num_nodes}")
            batch_size //= (self.workers_per_node * num_nodes)
            return self.internal.fit(data=data,
                                     epochs=epochs,
                                     batch_size=batch_size)
        else:
            from bigdl.nano.pytorch.trainer import Trainer

            # numpy data shape checking
            if isinstance(data, tuple):
                check_data(data[0], data[1], self.data_config)
            else:
                warnings.warn("Data shape checking is not supported by dataloader input.")

            # data transformation
            if isinstance(data, tuple):
                data = DataLoader(TensorDataset(torch.from_numpy(data[0]),
                                                torch.from_numpy(data[1])),
                                  batch_size=batch_size,
                                  shuffle=True)

            # Trainer init and fitting
            self.trainer = Trainer(logger=False, max_epochs=epochs,
                                   checkpoint_callback=self.checkpoint_callback,
                                   num_processes=self.num_processes, use_ipex=self.use_ipex)
            self.trainer.fit(self.internal, data)
            self.fitted = True

    def predict(self, data, batch_size=32):
        """
        Predict using a trained forecaster.

        if you want to predict on a single node(which is common practice), please call
        .to_local().predict(x, ...)

        :param data: The data support following formats:

               | 1. a numpy ndarray x:
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.
               | 2. a xshard item:
               | each partition can be a dictionary of {'x': x}, where x's shape
               | should follow the shape stated before.

        :param batch_size: predict batch size. The value will not affect predict
               result but will affect resources cost(e.g. memory and time).

        :return: A numpy array with shape (num_samples, horizon, target_dim)
                 if data is a numpy ndarray. A xshard item with format {‘prediction’: result},
                 where result is a numpy array with shape (num_samples, horizon, target_dim)
                 if data is a xshard item.
        """
        # data transform
        is_local_data = isinstance(data, np.ndarray)
        if is_local_data and self.distributed:
            data = np_to_xshard(data)
        if not is_local_data and not self.distributed:
            data = xshard_to_np(data, mode="predict")

        if self.distributed:
            yhat = self.internal.predict(data, batch_size=batch_size)
            if is_local_data:
                expand_dim = []
                if self.data_config["future_seq_len"] == 1:
                    expand_dim.append(1)
                if self.data_config["output_feature_num"] == 1:
                    expand_dim.append(2)
                yhat = xshard_to_np(yhat, mode="yhat", expand_dim=expand_dim)
            return yhat
        else:
            if not self.fitted:
                raise RuntimeError("You must call fit or restore first before calling predict!")
            yhat = self.internal.inference(torch.from_numpy(data), backend=None).numpy()
            if not is_local_data:
                yhat = np_to_xshard(yhat, prefix="prediction")
            return yhat

    def predict_with_onnx(self, data, batch_size=32):
        """
        Predict using a trained forecaster with onnxruntime. The method can only be
        used when forecaster is a non-distributed version.

        Directly call this method without calling build_onnx is valid and Forecaster will
        automatically build an onnxruntime session with default settings.

        :param data: The data support following formats:

               | 1. a numpy ndarray x:
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.

        :param batch_size: predict batch size. The value will not affect predict
               result but will affect resources cost(e.g. memory and time). Defaults
               to 32. None for all-data-single-time inference.

        :return: A numpy array with shape (num_samples, horizon, target_dim).
        """
        if self.distributed:
            raise NotImplementedError("ONNX inference has not been supported for distributed "
                                      "forecaster. You can call .to_local() to transform the "
                                      "forecaster to a non-distributed version.")
        if not self.fitted:
            raise RuntimeError("You must call fit or restore first before calling predict!")
        return self.internal.inference(data, batch_size=batch_size)

    def evaluate(self, data, batch_size=32, multioutput="raw_values"):
        """
        Evaluate using a trained forecaster.

        Please note that evaluate result is calculated by scaled y and yhat. If you scaled
        your data (e.g. use .scale() on the TSDataset) please follow the following code
        snap to evaluate your result if you need to evaluate on unscaled data.

        if you want to evaluate on a single node(which is common practice), please call
        .to_local().evaluate(data, ...)

        >>> from bigdl.orca.automl.metrics import Evaluator
        >>> y_hat = forecaster.predict(x)
        >>> y_hat_unscaled = tsdata.unscale_numpy(y_hat) # or other customized unscale methods
        >>> y_unscaled = tsdata.unscale_numpy(y) # or other customized unscale methods
        >>> Evaluator.evaluate(metric=..., y_unscaled, y_hat_unscaled, multioutput=...)

        :param data: The data support following formats:

               | 1. a numpy ndarray tuple (x, y):
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.
               | y's shape is (num_samples, horizon, target_dim), where horizon and target_dim
               | should be the same as future_seq_len and output_feature_num.
               | 2. a xshard item:
               | each partition can be a dictionary of {'x': x, 'y': y}, where x and y's shape
               | should follow the shape stated before.

        :param batch_size: evaluate batch size. The value will not affect evaluate
               result but will affect resources cost(e.g. memory and time).
        :param multioutput: Defines aggregating of multiple output values.
               String in ['raw_values', 'uniform_average']. The value defaults to
               'raw_values'.The param is only effective when the forecaster is a
               non-distribtued version.

        :return: A list of evaluation results. Each item represents a metric.
        """
        # data transform
        is_local_data = isinstance(data, tuple)
        if not is_local_data and not self.distributed:
            data = xshard_to_np(data, mode="fit")
        if self.distributed:
            if is_local_data:
                return self.internal.evaluate(data=np_to_creator(data),
                                              batch_size=batch_size)
            else:
                return self.internal.evaluate(data=data,
                                              batch_size=batch_size)
        else:
            if not self.fitted:
                raise RuntimeError("You must call fit or restore first before calling evaluate!")
            yhat_torch = self.internal.inference(torch.from_numpy(data[0]), backend=None)

            aggregate = 'mean' if multioutput == 'uniform_average' else None
            return Evaluator.evaluate(self.metrics, data[1],
                                      yhat_torch.numpy(), aggregate=aggregate)

    def evaluate_with_onnx(self, data,
                           batch_size=32,
                           multioutput="raw_values"):
        """
        Evaluate using a trained forecaster with onnxruntime. The method can only be
        used when forecaster is a non-distributed version.

        Directly call this method without calling build_onnx is valid and Forecaster will
        automatically build an onnxruntime session with default settings.

        Please note that evaluate result is calculated by scaled y and yhat. If you scaled
        your data (e.g. use .scale() on the TSDataset) please follow the following code
        snap to evaluate your result if you need to evaluate on unscaled data.

        >>> from bigdl.orca.automl.metrics import Evaluator
        >>> y_hat = forecaster.predict(x)
        >>> y_hat_unscaled = tsdata.unscale_numpy(y_hat) # or other customized unscale methods
        >>> y_unscaled = tsdata.unscale_numpy(y) # or other customized unscale methods
        >>> Evaluator.evaluate(metric=..., y_unscaled, y_hat_unscaled, multioutput=...)

        :param data: The data support following formats:

               | 1. a numpy ndarray tuple (x, y):
               | x's shape is (num_samples, lookback, feature_dim) where lookback and feature_dim
               | should be the same as past_seq_len and input_feature_num.
               | y's shape is (num_samples, horizon, target_dim), where horizon and target_dim
               | should be the same as future_seq_len and output_feature_num.

        :param batch_size: evaluate batch size. The value will not affect evaluate
               result but will affect resources cost(e.g. memory and time).
        :param multioutput: Defines aggregating of multiple output values.
               String in ['raw_values', 'uniform_average']. The value defaults to
               'raw_values'.

        :return: A list of evaluation results. Each item represents a metric.
        """
        if self.distributed:
            raise NotImplementedError("ONNX inference has not been supported for distributed "
                                      "forecaster. You can call .to_local() to transform the "
                                      "forecaster to a non-distributed version.")
        if not self.fitted:
            raise RuntimeError("You must call fit or restore first before calling evaluate!")
        yhat = self.internal.inference(data[0], batch_size=batch_size)

        aggregate = 'mean' if multioutput == 'uniform_average' else None
        return Evaluator.evaluate(self.metrics, data[1], yhat, aggregate=aggregate)

    def save(self, checkpoint_file):
        """
        Save the forecaster.

        Please note that if you only want the pytorch model or onnx model
        file, you can call .get_model() or .export_onnx_file(). The checkpoint
        file generated by .save() method can only be used by .load().

        :param checkpoint_file: The location you want to save the forecaster.
        """
        if self.distributed:
            self.internal.save(checkpoint_file)
        else:
            if not self.fitted:
                raise RuntimeError("You must call fit or restore first before calling save!")
            self.trainer.save_checkpoint(checkpoint_file)

    def load(self, checkpoint_file):
        """
        restore the forecaster.

        :param checkpoint_file: The checkpoint file location you want to load the forecaster.
        """
        if self.distributed:
            self.internal.load(checkpoint_file)
        else:
            from bigdl.nano.pytorch.lightning import LightningModuleFromTorch
            from bigdl.nano.pytorch.trainer import Trainer

            model = self.model_creator({**self.model_config, **self.data_config})
            loss = self.loss_creator(self.loss_config)
            optimizer = self.optimizer_creator(model, self.optim_config)
            self.internal = LightningModuleFromTorch.load_from_checkpoint(checkpoint_file,
                                                                          model=model,
                                                                          loss=loss,
                                                                          optimizer=optimizer)
            self.internal = Trainer.compile(self.internal, onnx=self.onnx_available)
            self.fitted = True

    def to_local(self):
        """
        Transform a distributed forecaster to a local (non-distributed) one.

        Common practice is to use distributed training (fit) and predict/
        evaluate with onnx or other frameworks on a single node. To do so,
        you need to call .to_local() and transform the forecaster to a non-
        distributed one.

        The optimizer is refreshed, incremental training after to_local
        might have some problem.

        :return: a forecaster instance.
        """
        from bigdl.nano.pytorch.trainer import Trainer

        # TODO: optimizer is refreshed, which is not reasonable
        if not self.distributed:
            raise RuntimeError("The forecaster has become local.")
        model = self.internal.get_model()
        self.internal.shutdown()

        loss = self.loss_creator(self.loss_config)
        optimizer = self.optimizer_creator(model, self.optim_config)
        self.internal = Trainer.compile(model=model, loss=loss,
                                        optimizer=optimizer, onnx=self.onnx_available)

        self.distributed = False
        self.fitted = True
        return self

    def get_model(self):
        """
        Returns the learned PyTorch model.

        :return: a pytorch model instance
        """
        if self.distributed:
            return self.internal.get_model()
        else:
            return self.internal.model

    def build_onnx(self, thread_num=None, sess_options=None):
        '''
        Build onnx model to speed up inference and reduce latency.
        The method is Not required to call before predict_with_onnx,
        evaluate_with_onnx or export_onnx_file.
        It is recommended to use when you want to:

        | 1. Strictly control the thread to be used during inferencing.
        | 2. Alleviate the cold start problem when you call predict_with_onnx
             for the first time.

        :param thread_num: int, the num of thread limit. The value is set to None by
               default where no limit is set.
        :param sess_options: an onnxruntime.SessionOptions instance, if you set this
               other than None, a new onnxruntime session will be built on this setting
               and ignore other settings you assigned(e.g. thread_num...).

        Example:
            >>> # to pre build onnx sess
            >>> forecaster.build_onnx(thread_num=1)  # build onnx runtime sess for single thread
            >>> pred = forecaster.predict_with_onnx(data)
            >>> # ------------------------------------------------------
            >>> # directly call onnx related method is also supported
            >>> pred = forecaster.predict_with_onnx(data)
        '''
        import onnxruntime
        if sess_options is not None and not isinstance(sess_options, onnxruntime.SessionOptions):
            raise RuntimeError("sess_options should be an onnxruntime.SessionOptions instance"
                               f", but found {type(sess_options)}")
        if sess_options is None:
            sess_options = onnxruntime.SessionOptions()
            if thread_num is not None:
                sess_options.intra_op_num_threads = thread_num
        if self.distributed:
            raise NotImplementedError("build_onnx has not been supported for distributed "
                                      "forecaster. You can call .to_local() to transform the "
                                      "forecaster to a non-distributed version.")
        dummy_input = torch.rand(1, self.data_config["past_seq_len"],
                                 self.data_config["input_feature_num"])
        self.internal.update_ortsess(dummy_input,
                                     sess_options=sess_options)

    def export_onnx_file(self, dirname="model.onnx"):
        """
        Save the onnx model file to the disk.

        :param dirname: The dir location you want to save the onnx file.
        """
        if self.distributed:
            raise NotImplementedError("export_onnx_file has not been supported for distributed "
                                      "forecaster. You can call .to_local() to transform the "
                                      "forecaster to a non-distributed version.")
        dummy_input = torch.rand(1, self.data_config["past_seq_len"],
                                 self.data_config["input_feature_num"])
        self.internal.update_ortsess(dummy_input,
                                     dirname=dirname)
