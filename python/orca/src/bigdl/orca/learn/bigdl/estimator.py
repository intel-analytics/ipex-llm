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
from bigdl.orca.learn.metrics import Metric
from bigdl.orca.learn.utils import bigdl_metric_results_to_dict
from bigdl.dllib.nnframes import NNEstimator, NNModel
from bigdl.dllib.estimator import Estimator as SparkEstimator
from bigdl.orca.learn.spark_estimator import Estimator as OrcaSparkEstimator
from bigdl.orca.data import SparkXShards
from bigdl.dllib.optim.optimizer import MaxEpoch
from bigdl.dllib.feature.common import FeatureSet
from bigdl.orca.learn.metrics import Accuracy
from pyspark.sql.dataframe import DataFrame


class Estimator(object):
    @staticmethod
    def from_bigdl(*, model, loss=None, optimizer=None, metrics=None,
                   feature_preprocessing=None, label_preprocessing=None,
                   model_dir=None):
        """
        Construct an Estimator with BigDL model, loss function and Preprocessing for feature and
        label data.

        :param model: BigDL Model to be trained.
        :param loss: BigDL criterion.
        :param optimizer: BigDL optimizer.
        :param metrics: A evaluation metric or a list of evaluation metrics
        :param feature_preprocessing: Used when data in `fit` and `predict` is a Spark DataFrame.
               The param converts the data in feature column to a Tensor or to a Sample directly.
               It expects a List of Int as the size of the converted Tensor, or a Preprocessing[F,
               Tensor[T]]

               If a List of Int is set as feature_preprocessing, it can only handle the case that
               feature column contains the following data types:
               Float, Double, Int, Array[Float], Array[Double], Array[Int] and MLlib Vector. The
               feature data are converted to Tensors with the specified sizes before
               sending to the model. Internally, a SeqToTensor is generated according to the
               size, and used as the feature_preprocessing.

               Alternatively, user can set feature_preprocessing as Preprocessing[F, Tensor[T]]
               that transforms the feature data to a Tensor[T]. Some pre-defined Preprocessing are
               provided in package bigdl.dllib.feature. Multiple Preprocessing can be combined as a
               ChainedPreprocessing.

               The feature_preprocessing will also be copied to the generated NNModel and applied
               to feature column during transform.
        :param label_preprocessing: Used when data in `fit` and `predict` is a Spark DataFrame.
               similar to feature_preprocessing, but applies to Label data.
        :param model_dir: The path to save model. During the training, if checkpoint_trigger is
               defined and triggered, the model will be saved to model_dir.
        :return:
        """
        return BigDLEstimator(model=model, loss=loss, optimizer=optimizer,
                              metrics=metrics,
                              feature_preprocessing=feature_preprocessing,
                              label_preprocessing=label_preprocessing, model_dir=model_dir)


class BigDLEstimator(OrcaSparkEstimator):
    def __init__(self, *, model, loss, optimizer=None, metrics=None,
                 feature_preprocessing=None, label_preprocessing=None, model_dir=None):
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = Metric.convert_metrics_list(metrics)
        self.feature_preprocessing = feature_preprocessing
        self.label_preprocessing = label_preprocessing
        self.model_dir = model_dir
        self.model = model
        self.nn_model = NNModel(self.model, feature_preprocessing=self.feature_preprocessing)
        self.nn_estimator = NNEstimator(self.model, self.loss, self.feature_preprocessing,
                                        self.label_preprocessing)
        if self.optimizer is None:
            from bigdl.dllib.optim.optimizer import SGD
            self.optimizer = SGD()
        self.nn_estimator.setOptimMethod(self.optimizer)
        self.estimator = SparkEstimator(self.model, self.optimizer, self.model_dir)
        self.log_dir = None
        self.app_name = None
        self.is_nnframe_fit = False

    def fit(self, data, epochs, batch_size=32, feature_cols="features", label_cols="label",
            caching_sample=True, validation_data=None, validation_trigger=None,
            checkpoint_trigger=None):
        """
        Train this BigDL model with train data.

        :param data: train data. It can be XShards or Spark DataFrame.
               If data is XShards, each partition is a dictionary of  {'x': feature,
               'y': label}, where feature(label) is a numpy array or a list of numpy arrays.
        :param epochs: Number of epochs to train the model.
        :param batch_size: Batch size used for training. Default: 32.
        :param feature_cols: Feature column name(s) of data. Only used when data is a Spark
               DataFrame. Default: "features".
        :param label_cols: Label column name(s) of data. Only used when data is a Spark DataFrame.
               Default: "label".
        :param caching_sample: whether to cache the Samples after preprocessing. Default: True
        :param validation_data: Validation data. XShards and Spark DataFrame are supported.
               If data is XShards, each partition is a dictionary of  {'x': feature,
               'y': label}, where feature(label) is a numpy array or a list of numpy arrays.
        :param validation_trigger: Orca Trigger to trigger validation computation.
        :param checkpoint_trigger: Orca Trigger to set a checkpoint.
        :return:
        """
        from bigdl.orca.learn.trigger import Trigger

        assert batch_size > 0, "batch_size should be greater than 0"

        if validation_data is not None:
            assert self.metrics is not None, \
                "You should provide metrics when creating this estimator if you provide " \
                "validation_data."

        if isinstance(data, DataFrame):
            if isinstance(feature_cols, list):
                data, validation_data, feature_cols = \
                    BigDLEstimator._combine_cols(data, feature_cols, col_name="features",
                                                 val_data=validation_data)

            if isinstance(label_cols, list):
                data, validation_data, label_cols = \
                    BigDLEstimator._combine_cols(data, label_cols, col_name="label",
                                                 val_data=validation_data)

            self.nn_estimator.setBatchSize(batch_size).setMaxEpoch(epochs) \
                .setCachingSample(caching_sample).setFeaturesCol(feature_cols) \
                .setLabelCol(label_cols)

            if validation_data is not None:
                assert isinstance(validation_data, DataFrame), \
                    "validation_data should be a spark DataFrame."
                assert validation_trigger is not None, \
                    "You should provide validation_trigger if you provide validation_data."
                validation_trigger = Trigger.convert_trigger(validation_trigger)
                self.nn_estimator.setValidation(validation_trigger, validation_data,
                                                self.metrics, batch_size)
            if self.log_dir is not None and self.app_name is not None:
                from bigdl.dllib.optim.optimizer import TrainSummary
                from bigdl.dllib.optim.optimizer import ValidationSummary
                train_summary = TrainSummary(log_dir=self.log_dir, app_name=self.app_name)
                self.nn_estimator.setTrainSummary(train_summary)
                val_summary = ValidationSummary(log_dir=self.log_dir, app_name=self.app_name)
                self.nn_estimator.setValidationSummary(val_summary)
            if self.model_dir is not None and checkpoint_trigger is not None:
                checkpoint_trigger = Trigger.convert_trigger(checkpoint_trigger)
                self.nn_estimator.setCheckpoint(self.model_dir, checkpoint_trigger)

            self.nn_model = self.nn_estimator.fit(data)
            self.is_nnframe_fit = True
        elif isinstance(data, SparkXShards):
            from bigdl.orca.data.utils import xshard_to_sample

            end_trigger = MaxEpoch(epochs)
            checkpoint_trigger = Trigger.convert_trigger(checkpoint_trigger)

            if isinstance(data, SparkXShards):
                train_rdd = data.rdd.flatMap(xshard_to_sample)
                train_feature_set = FeatureSet.sample_rdd(train_rdd)
                if validation_data is None:
                    val_feature_set = None
                else:
                    assert isinstance(validation_data, SparkXShards), \
                        "validation_data should be a XShards"
                    val_feature_set = FeatureSet.sample_rdd(
                        validation_data.rdd.flatMap(xshard_to_sample))
                if self.log_dir is not None and self.app_name is not None:
                    self.estimator.set_tensorboard(self.log_dir, self.app_name)
                self.estimator.train(train_feature_set, self.loss, end_trigger, checkpoint_trigger,
                                     val_feature_set, self.metrics, batch_size)
                self.is_nnframe_fit = False
            else:
                raise ValueError("Data and validation data should be XShards, but get " +
                                 data.__class__.__name__)
        else:
            raise ValueError("Data should be XShards or Spark DataFrame, but get " +
                             data.__class__.__name__)
        return self

    def predict(self, data, batch_size=4, feature_cols="features", sample_preprocessing=None):
        """
        Predict input data

        :param data: predict input data. It can be XShards or Spark DataFrame.
               If data is XShards, each partition is a dictionary of  {'x': feature}, where feature
               is a numpy array or a list of numpy arrays.
        :param batch_size: Batch size used for inference. Default: 4.
        :param feature_cols: Feature column name(s) of data. Only used when data is a Spark
               DataFrame. Default: "features".
        :param sample_preprocessing: Used when data is a Spark DataFrame. If the user want change
               the default feature_preprocessing specified in Estimator.from_bigdl, the user can
               pass the new sample_preprocessing methods.
        :return: predicted result.
                 If input data is Spark DataFrame, the predict result is a DataFrame which includes
                 original columns plus 'prediction' column. The 'prediction' column can be
                 FloatType, VectorUDT or Array of VectorUDT depending on model outputs shape.
                 If input data is an XShards, the predict result is a XShards, each partition
                 of the XShards is a dictionary of {'prediction': result}, where result is a numpy
                 array or a list of numpy arrays.
        """
        if isinstance(data, DataFrame):
            if isinstance(feature_cols, list):
                data, _, feature_cols = \
                    BigDLEstimator._combine_cols(data, feature_cols, col_name="features")
            self.nn_model.setBatchSize(batch_size).setFeaturesCol(feature_cols)
            if sample_preprocessing is not None:
                self.nn_model.setSamplePreprocessing(sample_preprocessing)
            return self.nn_model.transform(data)
        elif isinstance(data, SparkXShards):
            from bigdl.orca.data.utils import xshard_to_sample
            from bigdl.orca.learn.utils import convert_predict_rdd_to_xshard
            sample_rdd = data.rdd.flatMap(xshard_to_sample)
            result_rdd = self.model.predict(sample_rdd)
            return convert_predict_rdd_to_xshard(data, result_rdd)
        else:
            raise ValueError("Data should be XShards or Spark DataFrame, but get " +
                             data.__class__.__name__)

    def evaluate(self, data, batch_size=32, feature_cols="features", label_cols="label"):
        """
        Evaluate model.

        :param data: validation data. It can be XShardsor or Spark DataFrame, each partition is
               a dictionary of {'x': feature, 'y': label}, where feature(label) is a numpy array
               or a list of numpy arrays.
        :param batch_size: Batch size used for validation. Default: 32.
        :param feature_cols: (Not supported yet) Feature column name(s) of data. Only used when
               data is a Spark  DataFrame. Default: None.
        :param label_cols: (Not supported yet) Label column name(s) of data. Only used when data
               is a Spark DataFrame. Default: None.
        :return:
        """
        assert data is not None, "validation data shouldn't be None"
        assert self.metrics is not None, "metrics shouldn't be None, please specify the metrics" \
                                         " argument when creating this estimator."

        if isinstance(data, DataFrame):
            if isinstance(feature_cols, list):
                data, _, feature_cols = \
                    BigDLEstimator._combine_cols(data, [feature_cols], col_name="features")

            if isinstance(label_cols, list):
                data, _, label_cols = \
                    BigDLEstimator._combine_cols(data, label_cols, col_name="label")

            self.nn_estimator._setNNBatchSize(batch_size)._setNNFeaturesCol(feature_cols) \
                ._setNNLabelCol(label_cols)

            self.nn_estimator.setValidation(None, None,
                                            self.metrics, batch_size)
            if self.log_dir is not None and self.app_name is not None:
                from bigdl.dllib.optim.optimizer import TrainSummary
                from bigdl.dllib.optim.optimizer import ValidationSummary
                val_summary = ValidationSummary(log_dir=self.log_dir, app_name=self.app_name)
                self.nn_estimator.setValidationSummary(val_summary)

            result = self.nn_estimator._eval(data)

        elif isinstance(data, SparkXShards):
            from bigdl.orca.data.utils import xshard_to_sample
            val_feature_set = FeatureSet.sample_rdd(data.rdd.flatMap(xshard_to_sample))
            result = self.estimator.evaluate(val_feature_set, self.metrics, batch_size)
        else:
            raise ValueError("Data should be XShards or Spark DataFrame, but get " +
                             data.__class__.__name__)

        return bigdl_metric_results_to_dict(result)

    def get_model(self):
        """
        Get the trained BigDL model

        :return: The trained BigDL model
        """
        return self.model

    def save(self, model_path):
        """
        Save the BigDL model to model_path

        :param model_path: path to save the trained model.
        :return:
        """
        try:
            model = self.get_model()
            model.saveModel(model_path + ".bigdl", model_path + ".bin", True)
        except ValueError:
            raise ValueError("You should fit before calling save")

    def load(self, checkpoint, optimizer=None, loss=None, feature_preprocessing=None,
             label_preprocessing=None, model_dir=None, is_checkpoint=False):
        """
        Load existing BigDL model or checkpoint

        :param checkpoint: Path to the existing model or checkpoint.
        :param optimizer: BigDL optimizer.
        :param loss: BigDL criterion.
        :param feature_preprocessing: Used when data in `fit` and `predict` is a Spark DataFrame.
               The param converts the data in feature column to a Tensor or to a Sample directly.
               It expects a List of Int as the size of the converted Tensor, or a Preprocessing[F,
               Tensor[T]]

               If a List of Int is set as feature_preprocessing, it can only handle the case that
               feature column contains the following data types:
               Float, Double, Int, Array[Float], Array[Double], Array[Int] and MLlib Vector. The
               feature data are converted to Tensors with the specified sizes before
               sending to the model. Internally, a SeqToTensor is generated according to the
               size, and used as the feature_preprocessing.

               Alternatively, user can set feature_preprocessing as Preprocessing[F, Tensor[T]]
               that transforms the feature data to a Tensor[T]. Some pre-defined Preprocessing are
               provided in package bigdl.dllib.feature. Multiple Preprocessing can be combined as a
               ChainedPreprocessing.

               The feature_preprocessing will also be copied to the generated NNModel and applied
               to feature column during transform.
        :param label_preprocessing: Used when data in `fit` and `predict` is a Spark DataFrame.
               similar to feature_preprocessing, but applies to Label data.
        :param model_dir: The path to save model. During the training, if checkpoint_trigger is
               defined and triggered, the model will be saved to model_dir.
        :param is_checkpoint: Whether the path is a checkpoint or a saved BigDL model.
               Default: False.
        :return: The loaded estimator object.
        """
        if loss is not None:
            self.loss = loss
        if optimizer is not None:
            self.optimizer = optimizer
        if feature_preprocessing is not None:
            self.feature_preprocessing = feature_preprocessing
        if label_preprocessing is not None:
            self.label_preprocessing = label_preprocessing
        if model_dir is not None:
            self.model_dir = model_dir

        if is_checkpoint:
            self.load_orca_checkpoint(checkpoint)
        else:
            from bigdl.dllib.net import Net
            self.model = Net.load_bigdl(checkpoint + ".bigdl", checkpoint + ".bin")

            self.nn_estimator = NNEstimator(self.model, self.loss, self.feature_preprocessing,
                                            self.label_preprocessing)
            if self.optimizer is None:
                from bigdl.dllib.optim.optimizer import SGD
                self.optimizer = SGD()
            self.nn_estimator.setOptimMethod(self.optimizer)
            self.estimator = SparkEstimator(self.model, self.optimizer, self.model_dir)
            self.nn_model = NNModel(self.model, feature_preprocessing=self.feature_preprocessing)
        return self

    def __load_bigdl_model(self, path, bigdl_type="float"):
        from bigdl.dllib.utils.common import callBigDlFunc
        from bigdl.dllib.nn.layer import Layer
        jmodel = callBigDlFunc(bigdl_type, "loadBigDL", path)
        return Layer.of(jmodel)

    def load_orca_checkpoint(self, path, version=None, prefix=None):
        """
        Load existing checkpoint. To load a specific checkpoint, please provide both `version`
        and `perfix`. If `version` is None, then the latest checkpoint under the specified
        directory will be loaded.

        :param path: Path to the existing checkpoint (or directory containing Orca checkpoint
               files).
        :param version: checkpoint version, which is the suffix of model.* file, i.e., for
               modle.4 file, the version is 4. If it is None, then load the latest checkpoint.
        :param prefix: optimMethod prefix, for example 'optimMethod-Sequentialf53bddcc'
        :return:
        """
        from bigdl.dllib.nn.layer import Model, Container
        from bigdl.dllib.optim.optimizer import OptimMethod
        from bigdl.orca.learn.utils import find_latest_checkpoint
        import os

        if version is None:
            path, prefix, version = find_latest_checkpoint(path, model_type="bigdl")
            if path is None:
                raise ValueError("Cannot find BigDL checkpoint, please check your checkpoint"
                                 " path.")
        else:
            assert prefix is not None, "You should provide optimMethod prefix, " \
                                       "for example 'optimMethod-TorchModelf53bddcc'"

        try:
            self.model = self.__load_bigdl_model(os.path.join(path, "model.{}".format(version)))
            assert isinstance(self.model, Container), \
                "The loaded model should be a Container, please check your checkpoint type."
            self.optimizer = OptimMethod.load(os.path.join(path,
                                                           "{}.{}".format(prefix, version)))
        except Exception:
            raise ValueError("Cannot load BigDL checkpoint, please check your checkpoint path "
                             "and checkpoint type.")
        self.estimator = SparkEstimator(self.model, self.optimizer, self.model_dir)
        self.nn_estimator = NNEstimator(self.model, self.loss, self.feature_preprocessing,
                                        self.label_preprocessing)
        if self.optimizer is not None:
            self.nn_estimator.setOptimMethod(self.optimizer)
        self.nn_model = NNModel(self.model, feature_preprocessing=self.feature_preprocessing)

    def clear_gradient_clipping(self):
        """
        Clear gradient clipping parameters. In this case, gradient clipping will not be applied.
        In order to take effect, it needs to be called before fit.

        :return:
        """
        self.nn_estimator.clearGradientClipping()
        self.estimator.clear_gradient_clipping()

    def set_constant_gradient_clipping(self, min, max):
        """
        Set constant gradient clipping during the training process.
        In order to take effect, it needs to be called before fit.

        :param min: The minimum value to clip by.
        :param max: The maximum value to clip by.
        :return:
        """
        self.nn_estimator.setConstantGradientClipping(min, max)
        self.estimator.set_constant_gradient_clipping(min, max)

    def set_l2_norm_gradient_clipping(self, clip_norm):
        """
        Clip gradient to a maximum L2-Norm during the training process.
        In order to take effect, it needs to be called before fit.

        :param clip_norm: Gradient L2-Norm threshold.
        :return:
        """
        self.nn_estimator.setGradientClippingByL2Norm(clip_norm)
        self.estimator.set_l2_norm_gradient_clipping(clip_norm)

    def get_train_summary(self, tag=None):
        """
        Get the scalar from model train summary.

        This method will return a list of summary data of
        [iteration_number, scalar_value, timestamp].

        :param tag: The string variable represents the scalar wanted
        """
        # Exception handle
        if tag != "Loss" and tag != "LearningRate" and tag != "Throughput":
            raise TypeError('Only "Loss", "LearningRate", "Throughput"'
                            + 'are supported in train summary')
        if self.is_nnframe_fit:
            train_summary = self.nn_estimator.getTrainSummary()
            return train_summary.read_scalar(tag=tag)
        else:
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
        if self.is_nnframe_fit:
            assert tag is not None, "You should provide tag which should match the name of " \
                                    "the ValidationMethod set into the optimizer. " \
                                    "e.g.'MAE', 'Top1AccuracyLoss', 'Top1Accuracy' or " \
                                    "'Top5Accuracy'."
            val_summary = self.nn_estimator.getValidationSummary()
            return val_summary.read_scalar(tag=tag)
        else:
            return self.estimator.get_validation_summary(tag=tag)

    @staticmethod
    def _combine_cols(data, cols, col_name="features", val_data=None):
        if isinstance(cols, list):
            if len(cols) == 1:
                col_name = cols[0]
            else:
                from pyspark.ml.feature import VectorAssembler
                assembler = VectorAssembler(
                    inputCols=cols,
                    outputCol=col_name)
                data = assembler.transform(data)
                if val_data is not None:
                    val_data = assembler.transform(val_data)
        return data, val_data, col_name
