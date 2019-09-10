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

from bigdl.util.common import JavaValue, callBigDlFunc


class Estimator(JavaValue):
    """
    Estimator class for training and evaluation BigDL models.

    Estimator wraps a model, and provide an uniform training, evaluation or prediction operation on
    both local host and distributed spark environment.
    """
    def __init__(self, model, optim_methods=None, model_dir=None, jvalue=None, bigdl_type="float"):
        self.bigdl_type = bigdl_type
        self.value = jvalue if jvalue else callBigDlFunc(
            bigdl_type, self.jvm_class_constructor(), model, optim_methods, model_dir)

    def clear_gradient_clipping(self):
        """
        Clear gradient clipping parameters. In this case, gradient clipping will not be applied.
        In order to take effect, it needs to be called before fit.
        :return:
        """
        callBigDlFunc(self.bigdl_type, "clearGradientClipping")

    def set_constant_gradient_clipping(self, min, max):
        """
        Set constant gradient clipping during the training process.
        In order to take effect, it needs to be called before fit.
        :param min: The minimum value to clip by.
        :param max: The maximum value to clip by.
        :return:
        """
        callBigDlFunc(self.bigdl_type, "setConstantGradientClipping", self.value, min, max)

    def set_l2_norm_gradient_clipping(self, clip_norm):
        """
        Clip gradient to a maximum L2-Norm during the training process.
        In order to take effect, it needs to be called before fit.
        :param clip_norm: Gradient L2-Norm threshold.
        :return:
        """
        callBigDlFunc(self.bigdl_type, "setGradientClippingByL2Norm", self.value, clip_norm)

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
        callBigDlFunc(self.bigdl_type, "estimatorTrain", self.value, train_set,
                      criterion, end_trigger, checkpoint_trigger, validation_set,
                      validation_method, batch_size)

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
        callBigDlFunc(self.bigdl_type, "estimatorTrainImageFeature", self.value, train_set,
                      criterion, end_trigger, checkpoint_trigger, validation_set,
                      validation_method, batch_size)

    def evaluate(self, validation_set, validation_method, batch_size=32):
        """
        Evaluate the model on the validationSet with the validationMethods.
        :param validation_set: validation FeatureSet, a FeatureSet[Sample[T]]
        :param validation_method: validation methods
        :param batch_size: batch size
        :return: validation results
        """
        callBigDlFunc(self.bigdl_type, "estimatorEvaluate", self.value,
                      validation_set, validation_method, batch_size)

    def evaluate_imagefeature(self, validation_set, validation_method, batch_size=32):
        """
        Evaluate the model on the validationSet with the validationMethods.
        :param validation_set: validation FeatureSet, a FeatureSet[Sample[T]]
        :param validation_method: validation methods
        :param batch_size: batch size
        :return: validation results
        """
        callBigDlFunc(self.bigdl_type, "estimatorEvaluateImageFeature", self.value,
                      validation_set, validation_method, batch_size)
