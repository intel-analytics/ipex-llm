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
from pyspark.sql import DataFrame

from bigdl.optim.optimizer import MaxEpoch

from zoo.tfpark.tf_dataset import _standardize_keras_target_data
from zoo.common.utils import enable_multi_fs_load, enable_multi_fs_load_static, \
    enable_multi_fs_save
from zoo.orca import OrcaContext
from zoo.orca.data.tf.data import Dataset, TFDataDataset2
from zoo.orca.data import SparkXShards
from zoo.orca.learn.tf.utils import *
from zoo.orca.learn.trigger import Trigger
from zoo.orca.learn.utils import find_latest_checkpoint, convert_predict_rdd_to_xshard, \
    convert_predict_rdd_to_dataframe, process_xshards_of_pandas_dataframe
from zoo.tfpark import KerasModel
from zoo.tfpark import TFOptimizer, TFNet, ZooOptimizer
from zoo.tfpark.tf_optimizer import StatelessMetric
from zoo.tfpark.utils import evaluate_metrics
from zoo.util import nest
from zoo.util.tf import save_tf_checkpoint, load_tf_checkpoint
from zoo.orca.learn.spark_estimator import Estimator as SparkEstimator


class Estimator(SparkEstimator):

    def fit(self, data, epochs, batch_size=32, feature_cols=None, label_cols=None,
            validation_data=None, session_config=None, checkpoint_trigger=None,
            auto_shard_files=False):
        """
        Train the model with train data.

        :param data: train data. It can be XShards, Spark DataFrame, tf.data.Dataset.
               If data is XShards, each partition can be a Pandas DataFrame or a dictionary of
               {'x': feature, 'y': label}, where feature(label) is a numpy array or a tuple of
               numpy arrays.
        :param epochs: number of epochs to train.
        :param batch_size: total batch size for each iteration. Default: 32.
        :param feature_cols: feature column names if train data is Spark DataFrame or XShards
               of Pandas DataFrame.
        :param label_cols: label column names if train data is Spark DataFrame or XShards
               of Pandas DataFrame.
        :param validation_data: validation data. Validation data type should be the same
               as train data.
        :param session_config: tensorflow session configuration for training.
               Should be object of tf.ConfigProto
        :param checkpoint_trigger: when to trigger checkpoint during training.
               Should be a zoo.orca.learn.trigger, like EveryEpoch(), SeveralIteration(
               num_iterations),etc.
        :param auto_shard_files: whether to automatically detect if the dataset is file-based and
               and apply sharding on files, otherwise sharding on records. Default is False.
        """
        raise NotImplementedError

    def predict(self, data, batch_size=4, feature_cols=None, auto_shard_files=False):
        """
        Predict input data

        :param data: data to be predicted. It can be XShards, Spark DataFrame.
               If data is XShards, each partition can be a Pandas DataFrame or a dictionary of
               {'x': feature}, where feature is a numpy array or a tuple of numpy arrays.
        :param batch_size: batch size per thread
        :param feature_cols: list of feature column names if input data is Spark DataFrame or
               XShards of Pandas DataFrame.
        :param auto_shard_files: whether to automatically detect if the dataset is file-based and
               and apply sharding on files, otherwise sharding on records. Default is False.
        :return: predicted result.
                 If input data is XShards or tf.data.Dataset, the predict result is a XShards, each
                 partition of the XShards is a dictionary of {'prediction': result}, where the
                 result is a numpy array or a list of numpy arrays.
                 If input data is Spark DataFrame, the predict result is a DataFrame which includes
                 original columns plus 'prediction' column. The 'prediction' column can be
                 FloatType, VectorUDT or Array of VectorUDT depending on model outputs shape.
        """
        raise NotImplementedError

    def evaluate(self, data, batch_size=32, feature_cols=None, label_cols=None,
                 auto_shard_files=False):
        """
        Evaluate model.

        :param data: evaluation data. It can be XShards, Spark DataFrame, tf.data.Dataset.
               If data is XShards, each partition can be a Pandas DataFrame or a dictionary of
               {'x': feature, 'y': label}, where feature(label) is a numpy array or a tuple of
               numpy arrays.
               If data is tf.data.Dataset, each element is a tuple of input tensors.
        :param batch_size: batch size per thread.
        :param feature_cols: feature_cols: feature column names if train data is Spark DataFrame or
               XShards of Pandas DataFrame.
        :param label_cols: label column names if train data is Spark DataFrame or XShards
               of Pandas DataFrame.
        :param auto_shard_files: whether to automatically detect if the dataset is file-based and
               and apply sharding on files, otherwise sharding on records. Default is False.
        :return: evaluation result as a dictionary of {'metric name': metric value}
        """
        raise NotImplementedError

    def get_model(self):
        """
        Get the trained Tensorflow model

        :return: Trained model
        """
        raise NotImplementedError

    def save(self, model_path):
        """
        Save model to model_path

        :param model_path: path to save the trained model.
        :return:
        """
        raise NotImplementedError

    def load(self, model_path):
        """
        Load existing model

        :param model_path: Path to the existing model.
        :return:
        """
        raise NotImplementedError

    def clear_gradient_clipping(self):
        """
        Clear gradient clipping parameters. In this case, gradient clipping will not be applied.
        In order to take effect, it needs to be called before fit.

        :return:
        """
        raise NotImplementedError

    def set_constant_gradient_clipping(self, min, max):
        """
        Set constant gradient clipping during the training process.
        In order to take effect, it needs to be called before fit.

        :param min: The minimum value to clip by.
        :param max: The maximum value to clip by.
        :return:
        """
        raise NotImplementedError

    def set_l2_norm_gradient_clipping(self, clip_norm):
        """
        Clip gradient to a maximum L2-Norm during the training process.
        In order to take effect, it needs to be called before fit.

        :param clip_norm: Gradient L2-Norm threshold.
        :return:
        """
        raise NotImplementedError

    def get_train_summary(self, tag=None):
        """
        Get the scalar from model train summary.

        This method will return a list of summary data of
        [iteration_number, scalar_value, timestamp].

        :param tag: The string variable represents the scalar wanted
        """
        if self.tf_optimizer:
            return self.tf_optimizer.estimator.get_train_summary(tag)

        return None

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
        if self.tf_optimizer:
            for val_method in self.tf_optimizer.tf_model.val_methods:
                if isinstance(val_method, StatelessMetric):
                    if tag == val_method.name:
                        return self.tf_optimizer.estimator.get_validation_summary(tag)
                else:
                    if tag == str(val_method.val_method):
                        return self.tf_optimizer.estimator. \
                            get_validation_summary("{} {}".format(val_method.name, tag))
                continue
        return None

    def save_tf_checkpoint(self, path):
        """
        Save tensorflow checkpoint in this estimator.

        :param path: tensorflow checkpoint path.
        """
        raise NotImplementedError

    def load_tf_checkpoint(self, path):
        """
        Load tensorflow checkpoint to this estimator.

        :param path: tensorflow checkpoint path.
        """
        raise NotImplementedError

    def save_keras_model(self, path, overwrite=True):
        """
        Save tensorflow keras model in this estimator.

        :param path: keras model save path.
        :param overwrite: Whether to silently overwrite any existing file at the target location.
        """
        raise NotImplementedError

    def save_keras_weights(self, filepath, overwrite=True, save_format=None):
        """
        Save tensorflow keras model weights in this estimator.

        :param filepath: keras model weights save path.
        :param overwrite: Whether to silently overwrite any existing file at the target location.
        :param save_format: Either 'tf' or 'h5'. A `filepath` ending in '.h5' or
               '.keras' will default to HDF5 if `save_format` is `None`. Otherwise
               `None` defaults to 'tf'.
        """
        raise NotImplementedError

    def load_keras_weights(self, filepath, by_name=False):
        """
        Save tensorflow keras model in this estimator.

        :param filepath: keras model weights save path.
        :param by_name: Boolean, whether to load weights by name or by topological
               order. Only topological loading is supported for weight files in
               TensorFlow format.
        """
        raise NotImplementedError

    def load_orca_checkpoint(self, path, version=None):
        """
        Load Orca checkpoint. To load a specific checkpoint, please provide a `version`.
        If `version` is None, then the latest checkpoint will be loaded.

        :param path: checkpoint directory which contains model.* and
               optimMethod-TFParkTraining.* files.
        :param version: checkpoint version, which is the suffix of model.* file,
               i.e., for modle.4 file, the version is 4.
        """
        if version is None:
            path, _, version = find_latest_checkpoint(path, model_type="tf")
            if path is None:
                raise ValueError("Cannot find tf checkpoint, please check your checkpoint"
                                 " path.")

        self.load_checkpoint = True
        self.checkpoint_path = path
        self.checkpoint_version = version

    @staticmethod
    def from_graph(*, inputs, outputs=None,
                   labels=None, loss=None, optimizer=None,
                   metrics=None, clip_norm=None, clip_value=None,
                   updates=None, sess=None, model_dir=None, backend="bigdl"):
        """
        Create an Estimator for tesorflow graph.

        :param inputs: input tensorflow tensors.
        :param outputs: output tensorflow tensors.
        :param labels: label tensorflow tensors.
        :param loss: The loss tensor of the TensorFlow model, should be a scalar
        :param optimizer: tensorflow optimization method.
        :param clip_norm: float >= 0. Gradients will be clipped when their L2 norm exceeds
               this value.
        :param clip_value:  a float >= 0 or a tuple of two floats.
               If clip_value is a float, gradients will be clipped when their absolute value
               exceeds this value.
               If clip_value is a tuple of two floats, gradients will be clipped when their value
               less than clip_value[0] or larger than clip_value[1].
        :param metrics: metric tensor.
        :param updates: Collection for the update ops. For example, when performing batch
               normalization, the moving_mean and moving_variance should be updated and the user
               should add tf.GraphKeys.UPDATE_OPS to updates. Default is None.
        :param sess: the current TensorFlow Session, if you want to used a pre-trained model,
               you should use the Session to load the pre-trained variables and pass it to estimator
        :param model_dir: location to save model checkpoint and summaries.
        :param backend: backend for estimator. Now it only can be "bigdl".
        :return: an Estimator object.
        """
        assert backend == "bigdl", "only bigdl backend is supported for now"
        return TensorFlowEstimator(inputs=inputs,
                                   outputs=outputs,
                                   labels=labels,
                                   loss=loss,
                                   optimizer=optimizer,
                                   metrics=metrics,
                                   clip_norm=clip_norm,
                                   clip_value=clip_value,
                                   updates=updates,
                                   sess=sess,
                                   model_dir=model_dir
                                   )

    @staticmethod
    def from_keras(keras_model, metrics=None, model_dir=None, optimizer=None, backend="bigdl"):
        """
        Create an Estimator from a tensorflow.keras model. The model must be compiled.

        :param keras_model: the tensorflow.keras model, which must be compiled.
        :param metrics: user specified metric.
        :param model_dir: location to save model checkpoint and summaries.
        :param optimizer: an optional orca optimMethod that will override the optimizer in
               keras_model.compile
        :param backend: backend for estimator. Now it only can be "bigdl".
        :return: an Estimator object.
        """
        assert backend == "bigdl", "only bigdl backend is supported for now"
        return KerasEstimator(keras_model, metrics, model_dir, optimizer)

    @staticmethod
    @enable_multi_fs_load_static
    def load_keras_model(path):
        """
        Create Estimator by loading an existing keras model (with weights) from HDF5 file.

        :param path: String. The path to the pre-defined model.
        :return: Orca TF Estimator.
        """
        from tensorflow.python.keras import models
        model = models.load_model(path)
        return Estimator.from_keras(keras_model=model)


def is_tf_data_dataset(data):
    is_dataset = isinstance(data, tf.data.Dataset)
    is_dataset_v2 = isinstance(data, tf.python.data.ops.dataset_ops.DatasetV2)
    return is_dataset or is_dataset_v2


def to_dataset(data, batch_size, batch_per_thread, validation_data,
               feature_cols, label_cols, hard_code_batch_size,
               sequential_order, shuffle, auto_shard_files, memory_type="DRAM"):
    # todo wrap argument into kwargs
    if validation_data:
        if isinstance(data, SparkXShards):
            assert isinstance(validation_data, SparkXShards), \
                "train data and validation data should be both SparkXShards"
        if isinstance(data, Dataset):
            assert isinstance(validation_data, Dataset), \
                "train data and validation data should be both orca.data.tf.Dataset"
        if isinstance(data, DataFrame):
            assert isinstance(validation_data, DataFrame), \
                "train data and validation data should be both Spark DataFrame"
        if isinstance(data, tf.data.Dataset):
            assert isinstance(validation_data, tf.data.Dataset), \
                "train data and validation data should be both tf.data.Dataset"

    if isinstance(data, SparkXShards):
        dataset = xshards_to_tf_dataset(data,
                                        batch_size,
                                        batch_per_thread,
                                        validation_data,
                                        hard_code_batch_size=hard_code_batch_size,
                                        memory_type=memory_type,
                                        sequential_order=sequential_order,
                                        shuffle=shuffle)
    elif isinstance(data, Dataset):
        dataset = TFDataDataset2(data, batch_size=batch_size,
                                 batch_per_thread=batch_per_thread,
                                 validation_dataset=validation_data)
    elif isinstance(data, DataFrame):
        dataset = TFDataset.from_dataframe(data, feature_cols, label_cols,
                                           batch_size,
                                           batch_per_thread,
                                           hard_code_batch_size,
                                           validation_data,
                                           memory_type,
                                           sequential_order,
                                           shuffle
                                           )
    elif is_tf_data_dataset(data):
        dataset = TFDataset.from_tf_data_dataset(data,
                                                 batch_size,
                                                 batch_per_thread,
                                                 hard_code_batch_size,
                                                 validation_data,
                                                 sequential_order,
                                                 shuffle, auto_shard_files=auto_shard_files)
    else:
        raise ValueError("data must be SparkXShards or orca.data.tf.Dataset or "
                         "Spark DataFrame or tf.data.Dataset")

    return dataset


class TensorFlowEstimator(Estimator):
    def __init__(self, *, inputs, outputs, labels, loss,
                 optimizer, clip_norm, clip_value,
                 metrics, updates, sess, model_dir):
        self.inputs = inputs
        self.outputs = outputs
        self.labels = labels
        self.loss = loss
        self.use_bigdl_optim = False
        self.clip_norm = clip_norm
        self.clip_value = clip_value
        if optimizer is not None:
            from zoo.orca.learn.optimizers import Optimizer
            if isinstance(optimizer, Optimizer):
                self.train_op = None
                self.optimizer = optimizer.get_optimizer()
                self.use_bigdl_optim = True
            else:
                assert isinstance(optimizer, tf.train.Optimizer), \
                    "optimizer is of type {}, ".format(type(optimizer)) + \
                    "it should be an instance of tf.train.Optimizer"
                self.optimizer = ZooOptimizer(optimizer)
                if clip_norm or clip_value:
                    gvs = self.optimizer.compute_gradients(self.loss)
                    if clip_norm:
                        gvs = [(tf.clip_by_norm(g_v[0], clip_norm), g_v[1]) for g_v in gvs]
                    if clip_value:
                        if isinstance(clip_value, tuple):
                            assert len(clip_value) == 2 and clip_value[0] < clip_value[1], \
                                "clip value should be (clip_min, clip_max)"
                            gvs = [(tf.clip_by_value(g_v[0], clip_value[0], clip_value[1]), g_v[1])
                                   for g_v in gvs]
                        if isinstance(clip_value, (int, float)):
                            assert clip_value > 0, "clip value should be larger than 0"
                            gvs = [(tf.clip_by_value(g_v[0], -clip_value, clip_value), g_v[1])
                                   for g_v in gvs]
                        else:
                            raise Exception("clip_value should be a tuple or one number")
                    self.train_op = self.optimizer.apply_gradients(gvs)
                else:
                    self.train_op = self.optimizer.minimize(self.loss)
        else:
            self.optimizer = None
            self.train_op = None
        self.metrics = metrics
        self.updates = updates
        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess
        self.model_dir = model_dir
        self.load_checkpoint = False
        self.tf_optimizer = None
        self.log_dir = None
        self.app_name = None

    def fit(self, data,
            epochs=1,
            batch_size=32,
            feature_cols=None,
            label_cols=None,
            validation_data=None,
            session_config=None,
            checkpoint_trigger=None,
            auto_shard_files=False,
            feed_dict=None
            ):
        """
        Train this graph model with train data.

        :param data: train data. It can be XShards, Spark DataFrame, tf.data.Dataset.
               If data is XShards, each partition can be a Pandas DataFrame or a dictionary of
               {'x': feature, 'y': label}, where feature(label) is a numpy array or a tuple of
               numpy arrays.
               If data is tf.data.Dataset, each element is a tuple of input tensors.
        :param epochs: number of epochs to train.
        :param batch_size: total batch size for each iteration.
        :param feature_cols: feature column names if train data is Spark DataFrame or XShards
               of Pandas DataFrame.
        :param label_cols: label column names if train data is Spark DataFrame or XShards of
               Pandas DataFrame.
        :param validation_data: validation data. Validation data type should be the same
               as train data.
        :param auto_shard_files: whether to automatically detect if the dataset is file-based and
               and apply sharding on files, otherwise sharding on records. Default is False.
        :param session_config: tensorflow session configuration for training.
               Should be object of tf.ConfigProto
        :param feed_dict: a dictionary. The key is TensorFlow tensor, usually a
               placeholder, the value of the dictionary is a tuple of two elements. The first one of
               the tuple is the value to feed to the tensor in training phase and the second one
               is the value to feed to the tensor in validation phase.
        :param checkpoint_trigger: when to trigger checkpoint during training.
               Should be a zoo.orca.learn.trigger, like EveryEpoch(), SeveralIteration(
               num_iterations),etc.
        """

        assert self.labels is not None, \
            "labels is None; it should not be None in training"
        assert self.loss is not None, \
            "loss is None; it should not be None in training"
        assert self.optimizer is not None, \
            "optimizer is None; it should not be None in training"

        if isinstance(data, DataFrame):
            assert feature_cols is not None, \
                "feature columns is None; it should not be None in training"
            assert label_cols is not None, \
                "label columns is None; it should not be None in training"

        if isinstance(data, SparkXShards):
            if data._get_class_name() == 'pandas.core.frame.DataFrame':
                assert feature_cols is not None, \
                    "feature columns is None; it should not be None in training"
                assert label_cols is not None, \
                    "label columns is None; it should not be None in training"
                data, validation_data = process_xshards_of_pandas_dataframe(data, feature_cols,
                                                                            label_cols,
                                                                            validation_data, "fit")

        if checkpoint_trigger is not None:
            checkpoint_trigger = Trigger.convert_trigger(checkpoint_trigger)

        memory_type = OrcaContext.train_data_store
        dataset = to_dataset(data, batch_size=batch_size, batch_per_thread=-1,
                             validation_data=validation_data,
                             feature_cols=feature_cols, label_cols=label_cols,
                             hard_code_batch_size=False,
                             sequential_order=False, shuffle=True,
                             auto_shard_files=auto_shard_files,
                             memory_type=memory_type
                             )

        if feed_dict is not None:
            tensor_with_value = {key: (value[0], value[1]) for key, value in feed_dict.items()}
        else:
            tensor_with_value = None

        if self.use_bigdl_optim:
            self.tf_optimizer = TFOptimizer.from_loss(
                self.loss, self.optimizer,
                session=self.sess, inputs=(self.inputs, self.labels), dataset=dataset,
                clip_norm=self.clip_norm, clip_value=self.clip_value, metrics=self.metrics,
                tensor_with_value=tensor_with_value, session_config=session_config,
                model_dir=self.model_dir, updates=self.updates)
        else:

            self.tf_optimizer = TFOptimizer.from_train_op(
                train_op=self.train_op,
                loss=self.loss,
                inputs=self.inputs,
                labels=self.labels,
                dataset=dataset,
                metrics=self.metrics,
                updates=self.updates, sess=self.sess,
                tensor_with_value=tensor_with_value,
                session_config=session_config,
                model_dir=self.model_dir)

        if self.load_checkpoint:
            self.tf_optimizer.load_checkpoint(self.checkpoint_path, self.checkpoint_version)

        if self.log_dir and self.app_name:
            self.tf_optimizer.estimator.set_tensorboard(self.log_dir, self.app_name)

        self.tf_optimizer.optimize(end_trigger=MaxEpoch(epochs),
                                   checkpoint_trigger=checkpoint_trigger)
        return self

    def predict(self, data, batch_size=4,
                feature_cols=None,
                auto_shard_files=False,
                ):
        """
        Predict input data

        :param data: data to be predicted. It can be XShards, Spark DataFrame.
               If data is XShards, each partition can be a Pandas DataFrame or a dictionary of
               {'x': feature}, where feature is a numpy array or a tuple of numpy arrays.
        :param batch_size: batch size per thread
        :param feature_cols: list of feature column names if input data is Spark DataFrame
               or XShards of Pandas DataFrame.
        :param auto_shard_files: whether to automatically detect if the dataset is file-based and
               and apply sharding on files, otherwise sharding on records. Default is False.
        :return: predicted result.
                 If input data is XShards or tf.data.Dataset, the predict result is a XShards, each
                 partition of the XShards is a dictionary of {'prediction': result}, where the
                 result is a numpy array or a list of numpy arrays.
                 If input data is Spark DataFrame, the predict result is a DataFrame which includes
                 original columns plus 'prediction' column. The 'prediction' column can be
                 FloatType, VectorUDT or Array of VectorUDT depending on model outputs shape.
        """

        assert self.outputs is not None, \
            "output is None, it should not be None in prediction"
        if isinstance(data, DataFrame):
            assert feature_cols is not None, \
                "feature columns is None; it should not be None in prediction"
        if isinstance(data, SparkXShards):
            if data._get_class_name() == 'pandas.core.frame.DataFrame':
                assert feature_cols is not None, \
                    "feature columns is None; it should not be None in prediction"
                data = process_xshards_of_pandas_dataframe(data, feature_cols)

        assert not is_tf_data_dataset(data), "tf.data.Dataset currently cannot be used for" \
                                             "estimator prediction"

        dataset = to_dataset(data, batch_size=-1, batch_per_thread=batch_size,
                             validation_data=None,
                             feature_cols=feature_cols, label_cols=None,
                             hard_code_batch_size=False,
                             sequential_order=True,
                             shuffle=False,
                             auto_shard_files=auto_shard_files,
                             )

        flat_inputs = nest.flatten(self.inputs)
        flat_outputs = nest.flatten(self.outputs)
        tfnet = TFNet.from_session(sess=self.sess, inputs=flat_inputs, outputs=flat_outputs)
        predicted_rdd = tfnet.predict(dataset)
        if isinstance(data, DataFrame):
            return convert_predict_rdd_to_dataframe(data, predicted_rdd)
        elif isinstance(data, SparkXShards):
            return convert_predict_rdd_to_xshard(data, predicted_rdd)
        else:
            return predicted_rdd

    def evaluate(self, data, batch_size=32,
                 feature_cols=None,
                 label_cols=None,
                 auto_shard_files=False,
                 ):
        """
        Evaluate model.

        :param data: evaluation data. It can be XShards, Spark DataFrame, tf.data.Dataset.
               If data is XShards, each partition can be a Pandas DataFrame or a dictionary of
               {'x': feature, 'y': label}, where feature(label) is a numpy array or a tuple of
               numpy arrays.
               If data is tf.data.Dataset, each element is a tuple of input tensors.
        :param batch_size: batch size per thread.
        :param feature_cols: feature_cols: feature column names if train data is Spark DataFrame
               or XShards of Pandas DataFrame.
        :param label_cols: label column names if train data is Spark DataFrame or XShards
               of Pandas DataFrame.
        :param auto_shard_files: whether to automatically detect if the dataset is file-based and
               and apply sharding on files, otherwise sharding on records. Default is False.
        :return: evaluation result as a dictionary of {'metric name': metric value}
        """

        assert self.metrics is not None, \
            "metrics is None, it should not be None in evaluate"

        if isinstance(data, DataFrame):
            assert feature_cols is not None, \
                "feature columns is None; it should not be None in evaluation"
            assert label_cols is not None, \
                "label columns is None; it should not be None in evaluation"

        if isinstance(data, SparkXShards):
            if data._get_class_name() == 'pandas.core.frame.DataFrame':
                assert feature_cols is not None, \
                    "feature columns is None; it should not be None in evaluation"
                assert label_cols is not None, \
                    "label columns is None; it should not be None in evaluation"
                data = process_xshards_of_pandas_dataframe(data, feature_cols, label_cols)

        dataset = to_dataset(data, batch_size=-1, batch_per_thread=batch_size,
                             validation_data=None,
                             feature_cols=feature_cols, label_cols=label_cols,
                             hard_code_batch_size=False,
                             sequential_order=True,
                             shuffle=False,
                             auto_shard_files=auto_shard_files,
                             )

        flat_inputs = nest.flatten(self.inputs)
        flat_labels = nest.flatten(self.labels)

        return evaluate_metrics(flat_inputs + flat_labels,
                                sess=self.sess,
                                dataset=dataset, metrics=self.metrics)

    def save_tf_checkpoint(self, path):
        """
        Save tensorflow checkpoint in this estimator.

        :param path: tensorflow checkpoint path.
        """
        save_tf_checkpoint(self.sess, path)

    def load_tf_checkpoint(self, path):
        """
        Load tensorflow checkpoint to this estimator.
        :param path: tensorflow checkpoint path.
        """
        load_tf_checkpoint(self.sess, path)

    def get_model(self):
        """
        Get_model is not supported in tensorflow graph estimator
        """
        raise NotImplementedError

    def save(self, model_path):
        """
        Save model (tensorflow checkpoint) to model_path

        :param model_path: path to save the trained model.
        :return:
        """
        self.save_tf_checkpoint(model_path)

    def load(self, model_path):
        """
        Load existing model (tensorflow checkpoint) from model_path
        :param model_path: Path to the existing tensorflow checkpoint.
        :return:
        """
        self.load_tf_checkpoint(model_path)

    def clear_gradient_clipping(self):
        """
        Clear gradient clipping is not supported in TensorFlowEstimator.
        """
        raise NotImplementedError

    def set_constant_gradient_clipping(self, min, max):
        """
        Set constant gradient clipping is not supported in TensorFlowEstimator. Please pass the
        clip_value to Estimator.from_graph.
        """
        raise NotImplementedError

    def set_l2_norm_gradient_clipping(self, clip_norm):
        """
        Set l2 norm gradient clipping is not supported in TensorFlowEstimator. Please pass the
        clip_norm to Estimator.from_graph.
        """
        raise NotImplementedError

    def shutdown(self):
        """
        Close TensorFlow session and release resources.
        """
        self.sess.close()


class KerasEstimator(Estimator):
    def __init__(self, keras_model, metrics, model_dir, optimizer):
        self.model = KerasModel(keras_model, model_dir)
        self.load_checkpoint = False
        self.metrics = metrics
        self.tf_optimizer = None
        self.optimizer = optimizer
        from zoo.orca.learn.optimizers import Optimizer
        if self.optimizer is not None and isinstance(self.optimizer, Optimizer):
            self.optimizer = self.optimizer.get_optimizer()
        self.log_dir = None
        self.app_name = None
        self.clip_norm = None
        self.clip_min = None
        self.clip_max = None

    def fit(self, data,
            epochs=1,
            batch_size=32,
            feature_cols=None,
            label_cols=None,
            validation_data=None,
            session_config=None,
            checkpoint_trigger=None,
            auto_shard_files=True
            ):
        """
        Train this keras model with train data.

        :param data: train data. It can be XShards, Spark DataFrame, tf.data.Dataset.
               If data is XShards, each partition can be a Pandas DataFrame or a dictionary of
               {'x': feature, 'y': label}, where feature(label) is a numpy array or a tuple of
               numpy arrays.
               If data is tf.data.Dataset, each element is [feature tensor tuple, label tensor
               tuple]
        :param epochs: number of epochs to train.
        :param batch_size: total batch size for each iteration.
        :param feature_cols: feature column names if train data is Spark DataFrame or XShards
               of Pandas DataFrame.
        :param label_cols: label column names if train data is Spark DataFrame or XShards of
               Pandas DataFrame.
        :param validation_data: validation data. Validation data type should be the same
               as train data.
        :param session_config: tensorflow session configuration for training.
               Should be object of tf.ConfigProto
        :param checkpoint_trigger: when to trigger checkpoint during training.
               Should be a zoo.orca.learn.trigger, like EveryEpoch(), SeveralIteration(
               num_iterations),etc.
        :param auto_shard_files: whether to automatically detect if the dataset is file-based and
               and apply sharding on files, otherwise sharding on records. Default is False.
        """

        if isinstance(data, DataFrame):
            assert feature_cols is not None, \
                "feature columns is None; it should not be None in training"
            assert label_cols is not None, \
                "label columns is None; it should not be None in training"

        if isinstance(data, tf.data.Dataset):
            assert isinstance(data.element_spec, tuple), \
                "If data is tf.data.Dataset, each element should be " \
                "(feature tensors, label tensor), where each feature/label tensor can be " \
                "either a single tensor or a tuple of tensors"
            if validation_data is not None:
                assert isinstance(validation_data, tf.data.Dataset), \
                    "train data and validation data should be both tf.data.Dataset"
                assert isinstance(validation_data.element_spec, tuple), \
                    "If validation_data is tf.data.Dataset, each element should be " \
                    "(feature tensors, label tensor), where each feature/label tensor can be " \
                    "either a single tensor or a tuple of tensors"

        if isinstance(data, SparkXShards):
            if data._get_class_name() == 'pandas.core.frame.DataFrame':
                assert feature_cols is not None, \
                    "feature columns is None; it should not be None in training"
                assert label_cols is not None, \
                    "label columns is None; it should not be None in training"
                data, validation_data = process_xshards_of_pandas_dataframe(data, feature_cols,
                                                                            label_cols,
                                                                            validation_data,
                                                                            "fit")

        if checkpoint_trigger is not None:
            checkpoint_trigger = Trigger.convert_trigger(checkpoint_trigger)

        if is_tf_data_dataset(data):
            data = data.map(_standardize_keras_target_data)
            validation_data = validation_data.map(_standardize_keras_target_data)

        memory_type = OrcaContext.train_data_store
        dataset = to_dataset(data, batch_size=batch_size, batch_per_thread=-1,
                             validation_data=validation_data,
                             feature_cols=feature_cols, label_cols=label_cols,
                             hard_code_batch_size=False,
                             sequential_order=False, shuffle=True,
                             auto_shard_files=auto_shard_files,
                             memory_type=memory_type)

        self.tf_optimizer = TFOptimizer.from_keras(self.model.model, dataset,
                                                   model_dir=self.model.model_dir,
                                                   session_config=session_config,
                                                   metrics=self.metrics,
                                                   optimizer=self.optimizer)

        if self.clip_norm:
            self.tf_optimizer.set_gradient_clipping_by_l2_norm(clip_norm=self.clip_norm)
        if self.clip_min and self.clip_max:
            self.tf_optimizer.set_constant_gradient_clipping(self.clip_min, self.clip_max)

        if self.load_checkpoint:
            self.tf_optimizer.load_checkpoint(self.checkpoint_path, self.checkpoint_version)

        if self.log_dir and self.app_name:
            self.tf_optimizer.estimator.set_tensorboard(self.log_dir, self.app_name)

        self.tf_optimizer.optimize(MaxEpoch(epochs), checkpoint_trigger=checkpoint_trigger)

        return self

    def predict(self, data, batch_size=4,
                feature_cols=None,
                auto_shard_files=False,
                ):
        """
        Predict input data

        :param data: data to be predicted.
               It can be XShards, Spark DataFrame, or tf.data.Dataset.
               If data is XShards, each partition can be a Pandas DataFrame or a dictionary of
               {'x': feature}, where feature is a numpy array or a tuple of numpy arrays.
               If data is tf.data.Dataset, each element is feature tensor tuple
        :param batch_size: batch size per thread
        :param feature_cols: list of feature column names if input data is Spark DataFrame or
               XShards of Pandas DataFrame.
        :param auto_shard_files: whether to automatically detect if the dataset is file-based and
               and apply sharding on files, otherwise sharding on records. Default is False.
        :return: predicted result.
                 If input data is XShards or tf.data.Dataset, the predict result is also a XShards,
                 and the schema for each result is: {'prediction': predicted numpy array or
                 list of predicted numpy arrays}.
                 If input data is Spark DataFrame, the predict result is a DataFrame which includes
                 original columns plus 'prediction' column. The 'prediction' column can be
                 FloatType, VectorUDT or Array of VectorUDT depending on model outputs shape.
        """

        if isinstance(data, DataFrame):
            assert feature_cols is not None, \
                "feature columns is None; it should not be None in prediction"

        if isinstance(data, SparkXShards):
            if data._get_class_name() == 'pandas.core.frame.DataFrame':
                assert feature_cols is not None, \
                    "feature columns is None; it should not be None in prediction"
                data = process_xshards_of_pandas_dataframe(data, feature_cols)

        assert not is_tf_data_dataset(data), "tf.data.Dataset currently cannot be used for" \
                                             "estimator prediction"

        dataset = to_dataset(data, batch_size=-1, batch_per_thread=batch_size,
                             validation_data=None,
                             feature_cols=feature_cols, label_cols=None,
                             hard_code_batch_size=False,
                             sequential_order=True, shuffle=False,
                             auto_shard_files=auto_shard_files,
                             )

        predicted_rdd = self.model.predict(dataset, batch_size)
        if isinstance(data, DataFrame):
            return convert_predict_rdd_to_dataframe(data, predicted_rdd)
        elif isinstance(data, SparkXShards):
            return convert_predict_rdd_to_xshard(data, predicted_rdd)
        else:
            return predicted_rdd

    def evaluate(self, data, batch_size=32,
                 feature_cols=None,
                 label_cols=None,
                 auto_shard_files=False
                 ):
        """
        Evaluate model.

        :param data: evaluation data. It can be XShards, Spark DataFrame, tf.data.Dataset.
               If data is XShards, each partition can be a Pandas DataFrame or a dictionary of
               {'x': feature, 'y': label}, where feature(label) is a numpy array or a tuple of
               numpy arrays.
               If data is tf.data.Dataset, each element is [feature tensor tuple, label tensor
               tuple]
        :param batch_size: batch size per thread.
        :param feature_cols: feature_cols: feature column names if train data is Spark DataFrame or
               XShards of Pandas DataFrame.
        :param label_cols: label column names if train data is Spark DataFrame or XShards
               of Pandas DataFrame.
        :param auto_shard_files: whether to automatically detect if the dataset is file-based and
               and apply sharding on files, otherwise sharding on records. Default is False.
        :return: evaluation result as a dictionary of {'metric name': metric value}
        """

        if isinstance(data, DataFrame):
            assert feature_cols is not None, \
                "feature columns is None; it should not be None in evaluation"
            assert label_cols is not None, \
                "label columns is None; it should not be None in evaluation"

        if isinstance(data, SparkXShards):
            if data._get_class_name() == 'pandas.core.frame.DataFrame':
                assert feature_cols is not None, \
                    "feature columns is None; it should not be None in evaluation"
                assert label_cols is not None, \
                    "label columns is None; it should not be None in evaluation"
                data = process_xshards_of_pandas_dataframe(data, feature_cols, label_cols)

        dataset = to_dataset(data, batch_size=-1, batch_per_thread=batch_size,
                             validation_data=None,
                             feature_cols=feature_cols, label_cols=label_cols,
                             hard_code_batch_size=False,
                             sequential_order=True, shuffle=False,
                             auto_shard_files=auto_shard_files
                             )

        return self.model.evaluate(dataset, batch_per_thread=batch_size)

    @enable_multi_fs_save
    def save_keras_model(self, path, overwrite=True):
        """
        Save tensorflow keras model in this estimator.

        :param path: keras model save path.
        :param overwrite: Whether to silently overwrite any existing file at the target location.
        """
        self.model.save_model(path, overwrite=overwrite)

    def get_model(self):
        """
        Get the trained Keras model

        :return: The trained Keras model
        """
        return self.model.model

    @enable_multi_fs_save
    def save(self, model_path, overwrite=True):
        """
        Save model to model_path

        :param model_path: path to save the trained model.
        :param overwrite: Whether to silently overwrite any existing file at the target location.

        :return:
        """
        self.save_keras_model(model_path, overwrite=overwrite)

    @enable_multi_fs_load
    def load(self, model_path):
        """
        Load existing keras model

        :param model_path: Path to the existing keras model.
        :return:
        """
        self.model = KerasModel.load_model(model_path)

    def clear_gradient_clipping(self):
        """
        Clear gradient clipping parameters. In this case, gradient clipping will not be applied.
        In order to take effect, it needs to be called before fit.

        :return:
        """
        self.clip_norm = None
        self.clip_min = None
        self.clip_max = None

    def set_constant_gradient_clipping(self, min, max):
        """
        Set constant gradient clipping during the training process.
        In order to take effect, it needs to be called before fit.

        :param min: The minimum value to clip by.
        :param max: The maximum value to clip by.
        :return:
        """
        assert min > 0, "clip value should be larger than 0"
        assert min < max, "clip max should be larger than clip min"
        self.clip_min = min
        self.clip_max = max

    def set_l2_norm_gradient_clipping(self, clip_norm):
        """
        Clip gradient to a maximum L2-Norm during the training process.
        In order to take effect, it needs to be called before fit.

        :param clip_norm: Gradient L2-Norm threshold.
        :return:
        """
        self.clip_norm = clip_norm

    @enable_multi_fs_save
    def save_keras_weights(self, filepath, overwrite=True, save_format=None):
        """
        Save tensorflow keras model weights in this estimator.

        :param filepath: keras model weights save path.
        :param overwrite: Whether to silently overwrite any existing file at the target location.
        :param save_format: Either 'tf' or 'h5'. A `filepath` ending in '.h5' or
               '.keras' will default to HDF5 if `save_format` is `None`. Otherwise
               `None` defaults to 'tf'.
        """
        self.model.save_weights(filepath, overwrite, save_format)

    @enable_multi_fs_load
    def load_keras_weights(self, filepath, by_name=False):
        """
        Save tensorflow keras model in this estimator.

        :param filepath: keras model weights save path.
        :param by_name: Boolean, whether to load weights by name or by topological
               order. Only topological loading is supported for weight files in
               TensorFlow format.
        """
        self.model.load_weights(filepath, by_name)
