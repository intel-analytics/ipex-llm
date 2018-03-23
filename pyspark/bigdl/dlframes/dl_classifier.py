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

from pyspark.ml.param.shared import *
from pyspark.ml.wrapper import JavaModel, JavaEstimator, JavaTransformer
from bigdl.util.common import *


if sys.version >= '3':
    long = int
    unicode = str


class HasBatchSize(Params):
    """
    Mixin for param batchSize: batch size.
    """

    # a placeholder to make it appear in the generated doc
    batchSize = Param(Params._dummy(), "batchSize", "batchSize")

    def __init__(self):
        super(HasBatchSize, self).__init__()
        #: param for batch size.
        self.batchSize = Param(self, "batchSize", "batchSize")
        self._setDefault(batchSize=1)

    def setBatchSize(self, val):
        """
        Sets the value of :py:attr:`batchSize`.
        """
        self._paramMap[self.batchSize] = val
        pythonBigDL_method_name = "setBatchSize" + self.__class__.__name__
        callBigDlFunc(self.bigdl_type, pythonBigDL_method_name, self.value, val)
        return self

    def getBatchSize(self):
        """
        Gets the value of batchSize or its default value.
        """
        return self.getOrDefault(self.batchSize)

class HasMaxEpoch(Params):
    maxEpoch = Param(Params._dummy(), "maxEpoch", "number of max Epoch")

    def __init__(self):
        super(HasMaxEpoch, self).__init__()
        self.maxEpoch = Param(self, "maxEpoch", "maxEpoch")
        self._setDefault(maxEpoch=100)

    def setMaxEpoch(self, val):
        self._paramMap[self.maxEpoch] = val
        pythonBigDL_method_name = "setMaxEpoch" + self.__class__.__name__
        callBigDlFunc(self.bigdl_type, pythonBigDL_method_name, self.value, val)
        return self

    def getMaxEpoch(self):
        """
        Gets the value of maxEpoch or its default value.
        """
        return self.getOrDefault(self.maxEpoch)

class HasLearningRate(Params):
    learningRate = Param(Params._dummy(), "learningRate", "learning rate")

    def __init__(self):
        super(HasLearningRate, self).__init__()
        self.learningRate = Param(self, "learningRate", "learning rate")
        self._setDefault(learningRate=100)

    def setLearningRate(self, val):
        self._paramMap[self.learningRate] = val
        pythonBigDL_method_name = "setLearningRate" + self.__class__.__name__
        callBigDlFunc(self.bigdl_type, pythonBigDL_method_name, self.value, val)
        return self

    def getLearningRate(self):
        """
        Gets the value of maxEpoch or its default value.
        """
        return self.getOrDefault(self.learningRate)

class DLEstimator(JavaEstimator, HasFeaturesCol, HasLabelCol, HasPredictionCol, HasBatchSize, HasMaxEpoch, HasLearningRate, JavaValue):

    def __init__(self,  model, criterion, feature_size, label_size, jvalue=None, bigdl_type="float"):
        super(DLEstimator, self).__init__()
        self.value = jvalue if jvalue else callBigDlFunc(
            bigdl_type, self.jvm_class_constructor(), model, criterion, feature_size, label_size)
        self.bigdl_type = bigdl_type
        self._java_obj = self.value
        self.featureSize = feature_size

    def _create_model(self, java_model):
        dlModel = DLModel.of(java_model, self.featureSize, self.bigdl_type)
        dlModel.setFeaturesCol(self.getFeaturesCol()).setPredictionCol(self.getPredictionCol()) \
            .setBatchSize(self.getBatchSize())
        return dlModel

class DLModel(JavaTransformer, HasFeaturesCol, HasPredictionCol, HasBatchSize, JavaValue):
    def __init__(self,  model, featureSize, jvalue=None, bigdl_type="float"):
        super(DLModel, self).__init__()
        self.value = jvalue if jvalue else callBigDlFunc(
            bigdl_type, self.jvm_class_constructor(), model, featureSize)
        self._java_obj = self.value
        self.bigdl_type = bigdl_type
        self.setFeatureSize(featureSize)

    @classmethod
    def of(self, jvalue, feature_size=None, bigdl_type="float"):
        model = DLModel(model=None, featureSize=feature_size, jvalue=jvalue, bigdl_type=bigdl_type)
        return model

    def setFeatureSize(self, val):
        pythonBigDL_method_name = "setFeatureSize" + self.__class__.__name__
        self.__featuresize = val
        callBigDlFunc(self.bigdl_type, pythonBigDL_method_name, self.value, val)
        return self

    def getFeatureSize(self):
        return self.__featuresize


class DLClassifier(DLEstimator):
    def __init__(self,  model, criterion, feature_size,  bigdl_type="float"):
        super(DLClassifier, self).__init__(model, criterion, feature_size, [1], None, bigdl_type)


class DLClassifierModel(DLModel):
    def __init__(self,  model, featureSize, jvalue=None, bigdl_type="float"):
        super(DLClassifierModel, self).__init__(model, featureSize, jvalue, bigdl_type)

    @classmethod
    def of(self, jvalue, feature_size=None, bigdl_type="float"):
        model = DLClassifierModel(model=None, featureSize=feature_size, jvalue=jvalue, bigdl_type=bigdl_type)
        return model
