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
    def fit(self, data, epochs, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, data, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, data, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_model(self):
        raise NotImplementedError

    @abstractmethod
    def save(self, model_path):
        raise NotImplementedError

    @abstractmethod
    def load(self, checkpoint, **kwargs):
        raise NotImplementedError

    def set_tensorboard(self, log_dir, app_name):
        """
        Set summary information during the training process for visualization purposes.
        Saved summary can be viewed via TensorBoard.
        In order to take effect, it needs to be called before fit.

        Training summary will be saved to 'log_dir/app_name/train'
        and validation summary (if any) will be saved to 'log_dir/app_name/validation'.

        # Arguments
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
        # Arguments
        tag: The string variable represents the scalar wanted
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
        # Arguments
        tag: The string variable represents the scalar wanted
        """
        raise NotImplementedError

    def save_tf_checkpoint(self, path):
        """
        Save tensorflow checkpoint in this estimator.
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
        pass

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

    @abstractmethod
    def load_latest_orca_checkpoint(self, path):
        """
        Load latest Orca checkpoint under specified directory.
        :param path: directory containing Orca checkpoint files.
        """
        raise NotImplementedError
