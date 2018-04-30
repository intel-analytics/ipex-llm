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

from pyspark.ml.param.shared import *
from pyspark.ml.wrapper import JavaModel, JavaEstimator, JavaTransformer
from bigdl.optim.optimizer import SGD
from bigdl.util.common import *
from zoo.pipeline.nnframes.nn_transformers import *

if sys.version >= '3':
    long = int
    unicode = str


class HasBatchSize(Params):
    """
    Mixin for param batchSize: batch size.
    """

    # a placeholder to make it appear in the generated doc
    batchSize = Param(Params._dummy(), "batchSize", "batchSize (>= 0).")

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
        return self

    def getBatchSize(self):
        """
        Gets the value of batchSize or its default value.
        """
        return self.getOrDefault(self.batchSize)


class HasOptimMethod:

    optimMethod = SGD()

    def __init__(self):
        super(HasOptimMethod, self).__init__()

    def setOptimMethod(self, val):
        """
        Sets optimization method. E.g. SGD, Adam, LBFGS etc. from bigdl.optim.optimizer.
        default: SGD()
        """
        pythonBigDL_method_name = "setOptimMethod"
        callBigDlFunc(self.bigdl_type, pythonBigDL_method_name, self.value, val)
        self.optimMethod = val
        return self

    def getOptimMethod(self):
        """
        Gets the optimization method
        """
        return self.optimMethod


class NNEstimator(JavaEstimator, HasFeaturesCol, HasLabelCol, HasPredictionCol, HasBatchSize, HasOptimMethod, JavaValue):
    """
    NNEstimator provides DataFrame-based API that allows users to train a BigDL
    Model with the Spark ML Estimator/Transfomer pattern, thus Spark users can conveniently fit
    BigDL into Spark ML pipeline.

    NNEstimator supports feature and label data in the format of Array[Double], Array[Float],
    org.apache.spark.mllib.linalg.{Vector, VectorUDT}, org.apache.spark.ml.linalg.{Vector, VectorUDT},
    Double, Float and ImageSchema (refer to NNImageReader).
    """

    def __init__(self,  model, criterion, sample_transformer, jvalue=None, bigdl_type="float"):
        """
        create a NNEstimator with BigDL Model.
        :param model: BigDL Model to be trained.
        :param criterion: BigDL criterion.
        :param sample_transformer: Expert param. A transformer that transforms the (feature, label) tuple
               to a BigDL Sample[T], where T is decided by the BigDL model.

               Note that sampleTransformer should be able to handle the case that label = null.
               During fit, NNEstimator will extract (feature, label) tuple from input DataFrame and use
               sampleTransformer to transform the tuple into BigDL Sample to be ingested by the model.
               If Label column is not available, (feature, null) will be sent to sampleTransformer.
 
               The sampleTransformer will also be copied to the generated NNModel and applied to feature
               column during transform, where (feature, null) will be passed to the sampleTransformer.
               sampleTransformer should be a subClass of com.intel.analytics.bigdl.dataset.Transformer.
        :param label_size: The size (Tensor dimensions) of the feature data.
        :param jvalue: Java object create by Py4j
        :param bigdl_type: optional parameter. data type of model, "float"(default) or "double".
        """
        super(NNEstimator, self).__init__()
        self.value = jvalue if jvalue else callBigDlFunc(
            bigdl_type, self.jvm_class_constructor(), model, criterion, sample_transformer)
        self.bigdl_type = bigdl_type
        self._java_obj = self.value
        self.maxEpoch = Param(self, "maxEpoch", "number of max Epoch")
        self.learningRate = Param(self, "learningRate", "learning rate")
        self._setDefault(maxEpoch=50, learningRate=1e-3, batchSize=1)
        self.sample_transformer = sample_transformer

    @classmethod
    def withTensorTransformer(cls, model, criterion, feature_transformer, label_transformer, jvalue=None,
                              bigdl_type="float"):
        """
        create a NNEstimator with BigDL Model.
        :param model: BigDL Model to be trained.
        :param criterion: BigDL criterion.
        :param feature_transformer: A transformer that transforms the feature data to a Tensor[T].
               featureTransformer should be a subClass of com.intel.analytics.bigdl.dataset.Transformer.
               Some common transformers have been defined in package
               pipeline.nnframes.nn_transformers. E.g. SeqToTensor is used
               to transform Array[_] to Tensor, and NumToTensor transform a number to a Tensor. Multiple
               Transformer can be combined as a ChainedTransformer.
               E.g. For a feature column that contains 28 * 28 floats in an Array, Users can set
               SeqToTensor(Array(28, 28)) as featureTransformer, which will convert the feature data into
               Tensors with dimension 28 * 28 to be processed by Model.
        :param label_size: The size (Tensor dimensions) of the feature data.
        :param jvalue: Java object create by Py4j
        :param bigdl_type: optional parameter. data type of model, "float"(default) or "double".
        """
        return cls(model, criterion, FeatureLabelTransformer(feature_transformer, label_transformer), jvalue,
                   bigdl_type)

    def setMaxEpoch(self, val):
        """
        Sets the value of :py:attr:`maxEpoch`.
        """
        self._paramMap[self.maxEpoch] = val
        return self

    def getMaxEpoch(self):
        """
        Gets the value of maxEpoch or its default value.
        """
        return self.getOrDefault(self.maxEpoch)

    def setLearningRate(self, val):
        """
        Sets the value of :py:attr:`learningRate`.
        """
        self._paramMap[self.learningRate] = val
        return self

    def getLearningRate(self):
        """
        Gets the value of learningRate or its default value.
        """
        return self.getOrDefault(self.learningRate)

    def _create_model(self, java_model):
        nnModel = NNModel.of(java_model, FeatureToTupleAdapter(self.sample_transformer), self.bigdl_type)
        nnModel.setFeaturesCol(self.getFeaturesCol()) \
            .setPredictionCol(self.getPredictionCol()) \
            .setBatchSize(self.getBatchSize())
        return nnModel

class NNModel(JavaTransformer, HasFeaturesCol, HasPredictionCol, HasBatchSize, JavaValue):
    """
    NNModel helps embed a BigDL model into a Spark Transformer, thus Spark users can
    conveniently merge BigDL into Spark ML pipeline. NNModel supports feature data in
    the format of Array[Double], Array[Float], org.apache.spark.mllib.linalg.{Vector, VectorUDT},
    org.apache.spark.ml.linalg.{Vector, VectorUDT}, Double, Float and image schema.
    Internally NNModel use features column as storage of the feature data, and create
    Tensors according to the constructor parameter featureSize.
    """
    def __init__(self,  model, sample_transformer, jvalue=None, bigdl_type="float"):
        """
        create a NNModel with a BigDL model
        :param model: trained BigDL model to use in prediction.
        :param jvalue: Java object create by Py4j
        :param bigdl_type: optional parameter. data type of model, "float"(default) or "double".
        """
        super(NNModel, self).__init__()
        self.value = jvalue if jvalue else callBigDlFunc(
            bigdl_type, self.jvm_class_constructor(), model, sample_transformer)
        self._java_obj = self.value
        self.bigdl_type = bigdl_type

    @classmethod
    def withTensorTransformer(cls, model, feature_transformer, jvalue=None, bigdl_type="float"):
        chained_transformer = ChainedTransformer([feature_transformer, TensorToSample()])
        return NNModel(model, chained_transformer, jvalue, bigdl_type)

    @classmethod
    def of(self, jvalue, sample_transformer=None, bigdl_type="float"):
        model = NNModel(model=None, sample_transformer=sample_transformer, jvalue=jvalue, bigdl_type=bigdl_type)
        return model


class NNClassifier(NNEstimator):
    """
    NNClassifier is a specialized NNEstimator that simplifies the data format for
    classification tasks. It only supports label column of DoubleType, and the fitted
    NNClassifierModel will have the prediction column of DoubleType.
    """
    def __init__(self,  model, criterion, feature_transformer, bigdl_type="float"):
        """
        :param model: BigDL module to be optimized
        :param criterion: BigDL criterion method
        :param feature_size: The size (Tensor dimensions) of the feature data. (e.g. an
                             image may be with featureSize = 28 * 28).
        :param bigdl_type(optional): Data type of BigDL model, "float"(default) or "double".
        """
        super(NNClassifier, self).__init__(model, criterion, feature_transformer, None, bigdl_type)


class NNClassifierModel(NNModel):
    """
    NNClassifierModel is a specialized [[NNModel]] for classification tasks. The prediction
    column will have the datatype of Double.
    """
    def __init__(self,  model, feature_transformer, jvalue=None, bigdl_type="float"):
        """
        :param model: trained BigDL model to use in prediction.
        :param featureSize: The size (Tensor dimensions) of the feature data. (e.g. an
                            image may be with featureSize = 28 * 28).
        :param jvalue: Java object create by Py4j
        :param bigdl_type(optional): Data type of BigDL model, "float"(default) or "double".
        """
        super(NNClassifierModel, self).__init__(model, feature_transformer, jvalue, bigdl_type)

    @classmethod
    def of(self, jvalue, feaTran=None, bigdl_type="float"):
        model = NNClassifierModel(model=None, feature_transformer=feaTran, jvalue=jvalue, bigdl_type=bigdl_type)
        return model
