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
from zoo.pipeline.nnframes import NNEstimator, NNModel
from zoo.pipeline.estimator import Estimator as SparkEstimator
from zoo.orca.learn.spark_estimator import Estimator as OrcaSparkEstimator
from zoo.orca.data import SparkXShards
from bigdl.optim.optimizer import MaxEpoch
from zoo.feature.common import FeatureSet
from pyspark.sql.dataframe import DataFrame


class Estimator(object):
    @staticmethod
    def from_bigdl(*, model, loss=None, optimizer=None, feature_preprocessing=None,
                   label_preprocessing=None, model_dir=None):
        """
        Construct an Estimator with BigDL model, loss function and Preprocessing for feature and
        label data.
        :param model: BigDL Model to be trained.
        :param loss: BigDL criterion.
        :param optimizer: BigDL optimizer.
        :param feature_preprocessing: The param converts the data in feature column to a
               Tensor or to a Sample directly. It expects a List of Int as the size of the
               converted Tensor, or a Preprocessing[F, Tensor[T]]

               If a List of Int is set as feature_preprocessing, it can only handle the case that
               feature column contains the following data types:
               Float, Double, Int, Array[Float], Array[Double], Array[Int] and MLlib Vector. The
               feature data are converted to Tensors with the specified sizes before
               sending to the model. Internally, a SeqToTensor is generated according to the
               size, and used as the feature_preprocessing.

               Alternatively, user can set feature_preprocessing as Preprocessing[F, Tensor[T]]
               that transforms the feature data to a Tensor[T]. Some pre-defined Preprocessing are
               provided in package zoo.feature. Multiple Preprocessing can be combined as a
               ChainedPreprocessing.

               The feature_preprocessing will also be copied to the generated NNModel and applied
               to feature column during transform.
        :param label_preprocessing: similar to feature_preprocessing, but applies to Label data.
        :param model_dir: The path to save model. During the training, if checkpoint_trigger is
            defined and triggered, the model will be saved to model_dir.
        :return:
        """
        return BigDLEstimator(model=model, loss=loss, optimizer=optimizer,
                              feature_preprocessing=feature_preprocessing,
                              label_preprocessing=label_preprocessing, model_dir=model_dir)


class BigDLEstimator(OrcaSparkEstimator):
    def __init__(self, *, model, loss, optimizer=None, feature_preprocessing=None,
                 label_preprocessing=None, model_dir=None):
        self.loss = loss
        self.optimizer = optimizer
        self.feature_preprocessing = feature_preprocessing
        self.label_preprocessing = label_preprocessing
        self.model_dir = model_dir
        self.model = model
        self.nn_model = NNModel(self.model, feature_preprocessing=self.feature_preprocessing)
        self.nn_estimator = NNEstimator(self.model, self.loss, self.feature_preprocessing,
                                        self.label_preprocessing)
        if self.optimizer is None:
            from bigdl.optim.optimizer import SGD
            self.optimizer = SGD()
        self.nn_estimator.setOptimMethod(self.optimizer)
        self.estimator = SparkEstimator(self.model, self.optimizer, self.model_dir)
        self.log_dir = None
        self.app_name = None
        self.is_nnframe_fit = False

    def fit(self, data, epochs, batch_size=32, feature_cols="features", labels_cols="label",
            caching_sample=True, validation_data=None, validation_trigger=None,
            validation_metrics=None, checkpoint_trigger=None):
        from zoo.orca.learn.metrics import Metrics
        from zoo.orca.learn.trigger import Trigger

        assert batch_size > 0, "batch_size should be greater than 0"

        if isinstance(data, DataFrame):
            if isinstance(feature_cols, list):
                data, validation_data, feature_cols = \
                    BigDLEstimator._combine_cols(data, feature_cols, col_name="features",
                                                 val_data=validation_data)

            if isinstance(labels_cols, list):
                data, validation_data, labels_cols = \
                    BigDLEstimator._combine_cols(data, labels_cols, col_name="label",
                                                 val_data=validation_data)

            self.nn_estimator.setBatchSize(batch_size).setMaxEpoch(epochs)\
                .setCachingSample(caching_sample).setFeaturesCol(feature_cols)\
                .setLabelCol(labels_cols)

            if validation_data is not None:
                assert isinstance(validation_data, DataFrame), \
                    "validation_data should be a spark DataFrame."
                assert validation_trigger is not None and validation_metrics is not None, \
                    "You should provide validation_trigger and validation_metrics " \
                    "if you provide validation_data."
                validation_trigger = Trigger.convert_trigger(validation_trigger)
                validation_metrics = Metrics.convert_metrics_list(validation_metrics)
                self.nn_estimator.setValidation(validation_trigger, validation_data,
                                                validation_metrics, batch_size)
            if self.log_dir is not None and self.app_name is not None:
                from bigdl.optim.optimizer import TrainSummary
                from bigdl.optim.optimizer import ValidationSummary
                train_summary = TrainSummary(log_dir=self.log_dir, app_name=self.app_name)
                self.nn_estimator.setTrainSummary(train_summary)
                val_summary = ValidationSummary(log_dir=self.log_dir, app_name=self.log_dir)
                self.nn_estimator.setValidationSummary(val_summary)
            if self.model_dir is not None and checkpoint_trigger is not None:
                checkpoint_trigger = Trigger.convert_trigger(checkpoint_trigger)
                self.nn_estimator.setCheckpoint(self.model_dir, checkpoint_trigger)

            self.nn_model = self.nn_estimator.fit(data)
            self.is_nnframe_fit = True
        elif isinstance(data, SparkXShards):
            from zoo.orca.data.utils import to_sample

            end_trigger = MaxEpoch(epochs)
            validation_metrics = Metrics.convert_metrics_list(validation_metrics)
            checkpoint_trigger = Trigger.convert_trigger(checkpoint_trigger)

            if isinstance(data, SparkXShards):
                train_rdd = data.rdd.flatMap(to_sample)
                train_feature_set = FeatureSet.sample_rdd(train_rdd)
                if validation_data is None:
                    val_feature_set = None
                else:
                    assert isinstance(validation_data, SparkXShards), \
                        "validation_data should be a XShards"
                    val_feature_set = FeatureSet.sample_rdd(validation_data.rdd.flatMap(to_sample))
                if self.log_dir is not None and self.app_name is not None:
                    self.estimator.set_tensorboard(self.log_dir, self.app_name)
                self.estimator.train(train_feature_set, self.loss, end_trigger, checkpoint_trigger,
                                     val_feature_set, validation_metrics, batch_size)
                self.is_nnframe_fit = False
            else:
                raise ValueError("Data and validation data should be XShards, but get " +
                                 data.__class__.__name__)
        else:
            raise ValueError("Data should be XShards or Spark DataFrame, but get " +
                             data.__class__.__name__)
        return self

    def predict(self, data, batch_size=4, feature_cols="features", sample_preprocessing=None):
        if isinstance(data, DataFrame):
            if isinstance(feature_cols, list):
                data, _, feature_cols = \
                    BigDLEstimator._combine_cols(data, feature_cols, col_name="features")
            self.nn_model.setBatchSize(batch_size).setFeaturesCol(feature_cols)
            if sample_preprocessing is not None:
                self.nn_model.setSamplePreprocessing(sample_preprocessing)
            return self.nn_model.transform(data)
        elif isinstance(data, SparkXShards):
            from zoo.orca.data.utils import to_sample
            from zoo.orca.learn.utils import convert_predict_to_xshard
            sample_rdd = data.rdd.flatMap(to_sample)
            result_rdd = self.model.predict(sample_rdd)
            return convert_predict_to_xshard(result_rdd)
        else:
            raise ValueError("Data should be XShards or Spark DataFrame, but get " +
                             data.__class__.__name__)

    def evaluate(self, data, batch_size=32, feature_cols=None, labels_cols=None,
                 validation_metrics=None):
        assert data is not None, "validation data shouldn't be None"

        if isinstance(data, DataFrame):
            raise NotImplementedError
        elif isinstance(data, SparkXShards):
            from zoo.orca.data.utils import to_sample
            from zoo.orca.learn.metrics import Metrics

            validation_metrics = Metrics.convert_metrics_list(validation_metrics)
            val_feature_set = FeatureSet.sample_rdd(data.rdd.flatMap(to_sample))
            return self.estimator.evaluate(val_feature_set, validation_metrics, batch_size)
        else:
            raise ValueError("Data should be XShards or Spark DataFrame, but get " +
                             data.__class__.__name__)

    def get_model(self):
        return self.model

    def save(self, model_path):
        try:
            model = self.get_model()
            model.saveModel(model_path + ".bigdl", model_path + ".bin", True)
        except ValueError:
            raise ValueError("You should fit before calling save")

    def load(self, checkpoint, optimizer=None, loss=None, feature_preprocessing=None,
             label_preprocessing=None, model_dir=None, is_checkpoint=False):
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
            self.load_latest_orca_checkpoint(checkpoint)
        else:
            from zoo.pipeline.api.net import Net
            self.model = Net.load_bigdl(checkpoint + ".bigdl", checkpoint + ".bin")

            self.nn_estimator = NNEstimator(self.model, self.loss, self.feature_preprocessing,
                                            self.label_preprocessing)
            if self.optimizer is None:
                from bigdl.optim.optimizer import SGD
                self.optimizer = SGD()
            self.nn_estimator.setOptimMethod(self.optimizer)
            self.estimator = SparkEstimator(self.model, self.optimizer, self.model_dir)
            self.nn_model = NNModel(self.model, feature_preprocessing=self.feature_preprocessing)
        return self

    def load_orca_checkpoint(self, path, version, prefix=None):
        from bigdl.nn.layer import Model, Container
        from bigdl.optim.optimizer import OptimMethod
        import os
        try:
            self.model = Model.load(os.path.join(path, "model.{}".format(version)))
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

    def load_latest_orca_checkpoint(self, path):
        from zoo.orca.learn.utils import find_latest_checkpoint
        path, prefix, version = find_latest_checkpoint(path, model_type="bigdl")
        if path is None:
            raise ValueError("Cannot find BigDL checkpoint, please check your checkpoint path.")
        self.load_orca_checkpoint(path=path, version=version, prefix=prefix)

    def clear_gradient_clipping(self):
        self.nn_estimator.clearGradientClipping()
        self.estimator.clear_gradient_clipping()

    def set_constant_gradient_clipping(self, min, max):
        self.nn_estimator.setConstantGradientClipping(min, max)
        self.estimator.set_constant_gradient_clipping(min, max)

    def set_l2_norm_gradient_clipping(self, clip_norm):
        self.nn_estimator.setGradientClippingByL2Norm(clip_norm)
        self.estimator.set_l2_norm_gradient_clipping(clip_norm)

    def get_train_summary(self, tag=None):
        if self.is_nnframe_fit:
            return self.nn_estimator.getTrainSummary()
        else:
            return self.estimator.get_train_summary(tag=tag)

    def get_validation_summary(self, tag=None):
        if self.is_nnframe_fit:
            return self.nn_estimator.getValidationSummary()
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
