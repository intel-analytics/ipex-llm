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

from bigdl.util.common import JavaValue

from zoo.common.utils import callZooFunc


class Estimator(JavaValue):
    """
    Estimator class for training and evaluation BigDL models.

    Estimator wraps a model, and provide an uniform training, evaluation or prediction operation on
    both local host and distributed spark environment.
    """

    def __init__(self, model, optim_methods=None, model_dir=None, jvalue=None, bigdl_type="float"):
        self.bigdl_type = bigdl_type
        self.value = jvalue if jvalue else callZooFunc(
            bigdl_type, self.jvm_class_constructor(), model, optim_methods, model_dir)

    def clear_gradient_clipping(self):
        """
        Clear gradient clipping parameters. In this case, gradient clipping will not be applied.
        In order to take effect, it needs to be called before fit.
        :return:
        """
        callZooFunc(self.bigdl_type, "clearGradientClipping", self.value)

    def set_constant_gradient_clipping(self, min, max):
        """
        Set constant gradient clipping during the training process.
        In order to take effect, it needs to be called before fit.
        :param min: The minimum value to clip by.
        :param max: The maximum value to clip by.
        :return:
        """
        callZooFunc(self.bigdl_type, "setConstantGradientClipping", self.value, min, max)

    def set_l2_norm_gradient_clipping(self, clip_norm):
        """
        Clip gradient to a maximum L2-Norm during the training process.
        In order to take effect, it needs to be called before fit.
        :param clip_norm: Gradient L2-Norm threshold.
        :return:
        """
        callZooFunc(self.bigdl_type, "setGradientClippingByL2Norm", self.value, clip_norm)

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
        callZooFunc(self.bigdl_type, "estimatorSetTensorBoard",
                    self.value,
                    log_dir,
                    app_name)

    def get_train_summary(self, tag=None):
        """
        Get the scalar from model train summary
        Return 2-D array like object which could be converted
        by nd.array()
        # Arguments
        tag: The string variable represents the scalar wanted
        """
        # exception handle
        if tag != "Loss" and tag != "LearningRate" and tag != "Throughput":
            raise TypeError('Only "Loss", "LearningRate", "Throughput"'
                            + 'are supported in train summary')

        return callZooFunc("float", "estimatorGetScalarFromSummary",
                           self.value, tag, "Train")

    def get_validation_summary(self, tag=None):
        """
        Get the scalar from model validation summary
        Return 2-D array like object which could be converted
        by np.array()

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
        return callZooFunc("float", "estimatorGetScalarFromSummary",
                           self.value, tag, "Validation")

    def train(self, train_set, criterion, end_trigger=None, checkpoint_trigger=None,
              validation_set=None, validation_method=None, batch_size=32):
        """
        Train model with provided trainSet and criterion.
        The training will end until the endTrigger is triggered.
        During the training, if checkPointTrigger is defined and triggered,
        the model will be saved to modelDir. And if validationSet and validationMethod
        is defined, the model will be evaluated at the checkpoint.
        :param train_set: training FeatureSet, a FeatureSet[Sample[T]]
        :param criterion: Loss function
        :param end_trigger: When to finish the training
        :param checkpoint_trigger: When to save a checkpoint and evaluate model.
        :param validation_set: Validation FeatureSet, a FeatureSet[Sample[T]]
        :param validation_method: Validation Methods.
        :param batch_size:
        :return: Estimator
        """
        callZooFunc(self.bigdl_type, "estimatorTrain", self.value, train_set,
                    criterion, end_trigger, checkpoint_trigger, validation_set,
                    validation_method, batch_size)
        return self

    def train_minibatch(self, train_set, criterion, end_trigger=None, checkpoint_trigger=None,
                        validation_set=None, validation_method=None):
        """
        Train model with provided trainSet and criterion.
        The training will end until the endTrigger is triggered.
        During the training, if checkPointTrigger is defined and triggered,
        the model will be saved to modelDir. And if validationSet and validationMethod
        is defined, the model will be evaluated at the checkpoint.
        :param train_set: training FeatureSet, a FeatureSet[MiniBatch[T]]
        :param criterion: Loss function
        :param end_trigger: When to finish the training
        :param checkpoint_trigger: When to save a checkpoint and evaluate model.
        :param validation_set: Validation FeatureSet, a FeatureSet[MiniBatch[T]]
        :param validation_method: Validation Methods.
        :return: Estimator
        """

        callZooFunc(self.bigdl_type, "estimatorTrainMiniBatch", self.value, train_set,
                    criterion, end_trigger, checkpoint_trigger, validation_set,
                    validation_method)
        return self

    def train_imagefeature(self, train_set, criterion, end_trigger=None, checkpoint_trigger=None,
                           validation_set=None, validation_method=None, batch_size=32):
        """
        Train model with provided imageFeature trainSet and criterion.
        The training will end until the endTrigger is triggered.
        During the training, if checkPointTrigger is defined and triggered,
        the model will be saved to modelDir. And if validationSet and validationMethod
        is defined, the model will be evaluated at the checkpoint.
        :param train_set: training FeatureSet, a FeatureSet[ImageFeature]
        :param criterion: Loss function
        :param end_trigger: When to finish the training
        :param checkpoint_trigger: When to save a checkpoint and evaluate model.
        :param validation_set: Validation FeatureSet, a FeatureSet[Sample[T]]
        :param validation_method: Validation Methods.
        :param batch_size: Batch size
        :return:
        """
        callZooFunc(self.bigdl_type, "estimatorTrainImageFeature", self.value, train_set,
                    criterion, end_trigger, checkpoint_trigger, validation_set,
                    validation_method, batch_size)
        return self

    def evaluate(self, validation_set, validation_method, batch_size=32):
        """
        Evaluate the model on the validationSet with the validationMethods.
        :param validation_set: validation FeatureSet, a FeatureSet[Sample[T]]
        :param validation_method: validation methods
        :param batch_size: batch size
        :return: validation results
        """
        return callZooFunc(self.bigdl_type, "estimatorEvaluate", self.value,
                           validation_set, validation_method, batch_size)

    def evaluate_imagefeature(self, validation_set, validation_method, batch_size=32):
        """
        Evaluate the model on the validationSet with the validationMethods.
        :param validation_set: validation FeatureSet, a FeatureSet[Sample[T]]
        :param validation_method: validation methods
        :param batch_size: batch size
        :return: validation results
        """
        return callZooFunc(self.bigdl_type, "estimatorEvaluateImageFeature", self.value,
                           validation_set, validation_method, batch_size)

    def evaluate_minibatch(self, validation_set, validation_method):
        """
        Evaluate the model on the validationSet with the validationMethods.
        :param validation_set: validation FeatureSet, a FeatureSet[MiniBatch[T]]
        :param validation_method: validation methods
        :return: validation results
        """
        return callZooFunc(self.bigdl_type, "estimatorEvaluateMiniBatch", self.value,
                           validation_set, validation_method)
