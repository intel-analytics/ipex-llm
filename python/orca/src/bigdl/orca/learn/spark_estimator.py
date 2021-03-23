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

from abc import abstractmethod
from zoo.orca.learn.base_estimator import BaseEstimator


class Estimator(BaseEstimator):
    @abstractmethod
    def fit(self, data, epochs, batch_size=32, feature_cols=None, label_cols=None,
            validation_data=None, checkpoint_trigger=None):
        """
        Train the model with train data.

        :param data: train data.
        :param epochs: number of epochs to train.
        :param batch_size: total batch size for each iteration. Default: 32.
        :param feature_cols: feature column names if train data is Spark DataFrame.
        :param label_cols: label column names if train data is Spark DataFrame.
        :param validation_data: validation data. Validation data type should be the same
        as train data.
        :param checkpoint_trigger: when to trigger checkpoint during training.
        Should be a zoo.orca.learn.trigger, like EveryEpoch(), SeveralIteration(num_iterations),etc.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, data, batch_size=4, feature_cols=None):
        """
        Predict input data

        :param data: data to be predicted.
        :param batch_size: batch size per thread. Default: 4.
        :param feature_cols: list of feature column names if input data is Spark DataFrame.
        :return: predicted result.
         If input data is XShards or tf.data.Dataset, the predict result is a XShards,
         and the schema for each result is: {'prediction': predicted numpy array or
          list of predicted numpy arrays}.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, data, batch_size=32, feature_cols=None, label_cols=None):
        """
        Evaluate model.

        :param data: evaluation data.
        :param batch_size: batch size per thread. Default: 32.
        :param feature_cols: feature column names if train data is Spark DataFrame.
        :param label_cols: label column names if train data is Spark DataFrame.
        :return: evaluation result as a dictionary of {'metric name': metric value}
        """
        raise NotImplementedError

    @abstractmethod
    def get_model(self):
        """
        Get the trained model

        :return: Trained model
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, model_path):
        """
        Save model to model_path

        :param model_path: path to save the trained model.
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, model_path):
        """
        Load existing model from model_path

        :param model_path: Path to the existing model.
        :return:
        """
        raise NotImplementedError

    def set_tensorboard(self, log_dir, app_name):
        """
        Set summary information during the training process for visualization purposes.
        Saved summary can be viewed via TensorBoard.
        In order to take effect, it needs to be called before fit.

        Training summary will be saved to 'log_dir/app_name/train'
        and validation summary (if any) will be saved to 'log_dir/app_name/validation'.

        :param log_dir: The base directory path to store training and validation logs.
        :param app_name: The name of the application.
        """
        self.log_dir = log_dir
        self.app_name = app_name

    @abstractmethod
    def clear_gradient_clipping(self):
        """
        Clear gradient clipping parameters. In this case, gradient clipping will not be applied.
        In order to take effect, it needs to be called before fit.

        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def set_constant_gradient_clipping(self, min, max):
        """
        Set constant gradient clipping during the training process.
        In order to take effect, it needs to be called before fit.

        :param min: The minimum value to clip by.
        :param max: The maximum value to clip by.
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def set_l2_norm_gradient_clipping(self, clip_norm):
        """
        Clip gradient to a maximum L2-Norm during the training process.
        In order to take effect, it needs to be called before fit.

        :param clip_norm: Gradient L2-Norm threshold.
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def get_train_summary(self, tag=None):
        """
        Get the scalar from model train summary
        Return list of summary data of [iteration_number, scalar_value, timestamp]

        :param tag: The string variable represents the scalar wanted
        """
        raise NotImplementedError

    @abstractmethod
    def get_validation_summary(self, tag=None):
        """
        Get the scalar from model validation summary
        Return list of summary data of [iteration_number, scalar_value, timestamp]

        Note: The metric and tag may not be consistent
        Please look up following form to pass tag parameter
        Left side is your metric during compile
        Right side is the tag you should pass
        'Accuracy'                  |   'Top1Accuracy'
        'BinaryAccuracy'            |   'Top1Accuracy'
        'CategoricalAccuracy'       |   'Top1Accuracy'
        'SparseCategoricalAccuracy' |   'Top1Accuracy'
        'AUC'                       |   'AucScore'
        'HitRatio'                  |   'HitRate@k' (k is Top-k)
        'Loss'                      |   'Loss'
        'MAE'                       |   'MAE'
        'NDCG'                      |   'NDCG'
        'TFValidationMethod'        |   '${name + " " + valMethod.toString()}'
        'Top5Accuracy'              |   'Top5Accuracy'
        'TreeNNAccuracy'            |   'TreeNNAccuracy()'
        'MeanAveragePrecision'      |   'MAP@k' (k is Top-k) (BigDL)
        'MeanAveragePrecision'      |   'PascalMeanAveragePrecision' (Zoo)
        'StatelessMetric'           |   '${name}'

        :param tag: The string variable represents the scalar wanted
        """
        raise NotImplementedError

    @abstractmethod
    def load_orca_checkpoint(self, path, version):
        """
        Load specified Orca checkpoint.

        :param path: checkpoint directory which contains model.* and
        optimMethod-TFParkTraining.* files.
        :param version: checkpoint version, which is the suffix of model.* file,
        i.e., for modle.4 file, the version is 4.
        """
        raise NotImplementedError

    def shutdown(self):
        """
        Releases resources.

        :return:
        """
        pass
