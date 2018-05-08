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
from zoo.feature.common import *

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


class NNEstimator(JavaEstimator, HasFeaturesCol, HasLabelCol, HasPredictionCol, HasBatchSize,
                  HasOptimMethod, JavaValue):
    """
    NNEstimator extends org.apache.spark.ml.Estimator and supports training a BigDL model with
    Spark DataFrame data. It can be integrated into a standard Spark ML Pipeline to enable
    users for combined usage with Spark MLlib.

    NNEstimator supports different feature and label data type through operation defined in 
    Preprocessing. We provide pre-defined Preprocessing for popular data types like Array
    or Vector in package zoo.feature, while user can also develop customized Preprocess
    which extends from feature.common.Preprocessing. During fit, NNEstimator
    will extract feature and label data from input DataFrame and use the Preprocessing to prepare
    data for the model.
    Using the Preprocessing allows NNEstimator to cache only the raw data and decrease the
    memory consumption during feature conversion and training.

    More concrete examples are available in package com.intel.analytics.zoo.examples.nnframes
    """

    def __init__(self, model, criterion, sample_preprocessing, jvalue=None, bigdl_type="float"):
        """
        Construct a NNEstimator with BigDL model, criterion and a sample_preprocessing that
        transform a (feature, Option[label]) tuple to a BigDL Sample. This constructor is only 
        recommended for the expert users. Most users should use class method
        NNEstimator.create.
        :param model: BigDL Model to be trained.
        :param criterion: BigDL criterion.
        :param sample_preprocessing: Expert param. A Preprocessing that transforms the (feature, 
               Option[label]) tuple to a BigDL Sample[T], where T is decided by the BigDL model.

               Note that sample_preprocessing should be able to handle the case that label = None.
               During fit, NNEstimator will extract (feature, Option[label]) tuple from input 
               DataFrame and use sample_preprocessing to transform the tuple into BigDL Sample  
               to be ingested by the model. If Label column is not available, (feature, None)
               will be sent to sample_preprocessing.
 
               The sample_preprocessing will also be copied to the generated NNModel and applied
               to feature column during transform, where (feature, None) will be passed to the
               sample_preprocessing.
        :param jvalue: Java object create by Py4j
        :param bigdl_type: optional parameter. data type of model, "float"(default) or "double".
        """
        super(NNEstimator, self).__init__()
        self.value = jvalue if jvalue else callBigDlFunc(
            bigdl_type, self.jvm_class_constructor(), model, criterion, sample_preprocessing)
        self.bigdl_type = bigdl_type
        self._java_obj = self.value
        self.maxEpoch = Param(self, "maxEpoch", "number of max Epoch")
        self.learningRate = Param(self, "learningRate", "learning rate")
        self._setDefault(maxEpoch=50, learningRate=1e-3, batchSize=1)
        self.sample_preprocessing = sample_preprocessing
        self.train_summary = None
        self.validation_config = None
        self.validation_summary = None

    @classmethod
    def create(cls, model, criterion, feature_preprocessing, label_preprocessing,
               jvalue=None, bigdl_type="float"):
        """
        Construct a NNEstimator with a feature_preprocessing and a label_Preprocessing, which
        convert the data in feature column and label column to Tensors (Multi-dimension array)
        for model. This is the the recommended constructor for most users.

        The feature_preprocessing will be copied to the fitted NNModel, and apply to feature
        column data during transform.

        :param model: BigDL Model to be trained.
        :param criterion: BigDL criterion.
        :param feature_preprocessing: A Preprocessing that transforms the feature data to a
               Tensor[T]. Some pre-defined Preprocessing are provided in package
               zoo.feature. E.g.
               ArrayToTensor is used to transform Array[_] in DataFrame to Tensor. For a feature
               column that contains 576 floats in an Array, Users can set
               ArrayToTensor(Array(28, 28)) as feature_preprocessing, which will convert the feature
               data into Tensors with dimension 28 * 28 to be processed by a convolution Model.
               For a simple linear model, user may just use ArrayToTensor(Array(576)), which will
               convert the data into Tensors with single dimension (576).
               MLlibVectorToTensor is used to transform org.apache.spark.mllib.linalg.Vector
               to a Tensor.
               ScalarToTensor transform a number to a Tensor with single dimension of length 1.
               Multiple Preprocessing can be combined as a ChainedPreprocessing.
        :param label_preprocessing: similar to feature_preprocessing, but applies to Label data.
        :param jvalue: Java object create by Py4j
        :param bigdl_type: optional parameter. data type of model, "float"(default) or "double".
        """
        return cls(model, criterion,
                   FeatureLabelPreprocessing(feature_preprocessing, label_preprocessing),
                   jvalue, bigdl_type)
    
    @classmethod
    def createWithSize(cls, model, criterion, feature_size, label_size,
               jvalue=None, bigdl_type="float"):
        """
        Construct a NNEstimator with a feature size and label size. The constructor is useful
        when the feature column and label column contains the following data types:
        Float, Double, Int, Array[Float], Array[Double], Array[Int] and MLlib Vector. The
        feature and label data are converted to Tensors with the specified sizes before sending
        to the model.

        :param model: BigDL Model to be trained.
        :param criterion: BigDL criterion.
        :param feature_size: The size (Tensor dimensions) of the feature data. e.g. an image
                            may be with width * height = 28 * 28, featureSize = Array(28, 28).
        :param label_size: The size (Tensor dimensions) of the label data.
        :param jvalue: Java object create by Py4j
        :param bigdl_type: optional parameter. data type of model, "float"(default) or "double".
        """
        return cls(model, criterion,
                   FeatureLabelPreprocessing(SeqToTensor(feature_size), SeqToTensor(label_size)),
                   jvalue, bigdl_type)

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

    def setTrainSummary(self, val):
        """
        Statistics (LearningRate, Loss, Throughput, Parameters) collected during training for the
        training data, which can be used for visualization via Tensorboard.
        Use setTrainSummary to enable train logger. Then the log will be saved to
        logDir/appName/train as specified by the parameters of TrainSummary.
        Default: Not enabled

        :param summary: a TrainSummary object
        """
        pythonBigDL_method_name = "setTrainSummary"
        callBigDlFunc(self.bigdl_type, pythonBigDL_method_name, self.value, val)
        self.train_summary = val
        return self

    def getTrainSummary(self):
        """
        Gets the train summary
        """
        return self.train_summary

    def setValidationSummary(self, val):
        """
        Statistics (LearningRate, Loss, Throughput, Parameters) collected during training for the
        validation data if validation data is set, which can be used for visualization via
        Tensorboard. Use setValidationSummary to enable validation logger. Then the log will be
        saved to logDir/appName/ as specified by the parameters of validationSummary.
        Default: None
        """
        pythonBigDL_method_name = "setValidationSummary"
        callBigDlFunc(self.bigdl_type, pythonBigDL_method_name, self.value, val)
        self.validation_summary = val
        return self

    def getValidationSummary(self):
        """
        Gets the Validation summary
        """
        return self.validation_summary

    def setValidation(self, trigger, val_df, val_method, batch_size):
        """
        Set a validate evaluation during training

        :param trigger: validation interval
        :param val_df: validation dataset
        :param val_method: the ValidationMethod to use,e.g. "Top1Accuracy", "Top5Accuracy", "Loss"
        :param batch_size: validation batch size
        """
        pythonBigDL_method_name = "setValidation"
        callBigDlFunc(self.bigdl_type, pythonBigDL_method_name, self.value,
                      trigger, val_df, val_method, batch_size)
        self.validation_config = [trigger, val_df, val_method, batch_size]
        return self

    def getValidation(self):
        """
        Gets the validate configuration. If validation config has been set, getValidation will
        return a List of [ValidationTrigger, Validation data, Array[ValidationMethod[T]],
        batchsize]
        """
        return self.validation_config

    def _create_model(self, java_model):
        nnModel = NNModel.of(java_model, FeatureToTupleAdapter(self.sample_preprocessing),
                             self.bigdl_type)
        nnModel.setFeaturesCol(self.getFeaturesCol()) \
            .setPredictionCol(self.getPredictionCol()) \
            .setBatchSize(self.getBatchSize())
        return nnModel

class NNModel(JavaTransformer, HasFeaturesCol, HasPredictionCol, HasBatchSize, JavaValue):
    """
    NNModel extends Spark ML Transformer and supports BigDL model with Spark DataFrame.

    NNModel supports different feature data type through Preprocessing. Some common
    Preprocessing have been defined in com.intel.analytics.zoo.feature.

    After transform, the prediction column contains the output of the model as Array[T], where
    T (Double or Float) is decided by the model type.
    """
    def __init__(self, model, sample_preprocessing, jvalue=None, bigdl_type="float"):
        """
        create a NNModel with a BigDL model
        :param model: trained BigDL model to use in prediction.
        :param sample_preprocessing: A Preprocessing that transforms the feature data to a
               Sample[T].
        :param jvalue: Java object create by Py4j
        :param bigdl_type: optional parameter. data type of model, "float"(default) or "double".
        """
        super(NNModel, self).__init__()
        self.value = jvalue if jvalue else callBigDlFunc(
            bigdl_type, self.jvm_class_constructor(), model, sample_preprocessing)
        self._java_obj = self.value
        self.bigdl_type = bigdl_type

    @classmethod
    def create(cls, model, feature_preprocessing, jvalue=None, bigdl_type="float"):
        """
        Construct NNModel with a BigDL model and a feature-to-tensor Preprocessing
        :param model: trainned BigDL models to use in prediction.
        :param feature_preprocessing: A Preprocessing that transforms the feature data to a
               Tensor[T]. Some pre-defined Preprocessing are provided in package
               zoo.feature. E.g.
               ArrayToTensor is used to transform Array[_] in DataFrame to Tensor. For a feature
               column that contains 576 floats in an Array, Users can set
               ArrayToTensor(Array(28, 28)) as feature_preprocessing, which will convert the feature
               data into Tensors with dimension 28 * 28 to be processed by a convolution Model.
               For a simple linear model, user may just use ArrayToTensor(Array(576)), which will
               convert the data into Tensors with single dimension (576).
               MLlibVectorToTensor is used to transform org.apache.spark.mllib.linalg.Vector
               to a Tensor.
               ScalarToTensor transform a number to a Tensor with single dimension of length 1.
               Multiple Preprocessing can be combined as a ChainedPreprocessing.
        :param jvalue: Java object create by Py4j
        :param bigdl_type(optional): Data type of BigDL model, "float"(default) or "double".
        :return:
        """
        chained_transformer = ChainedPreprocessing([feature_preprocessing, TensorToSample()])
        return NNModel(model, chained_transformer, jvalue, bigdl_type)

    @classmethod
    def of(self, jvalue, sample_preprocessing=None, bigdl_type="float"):
        model = NNModel(model=None, sample_preprocessing=sample_preprocessing, jvalue=jvalue,
                        bigdl_type=bigdl_type)
        return model


class NNClassifier(NNEstimator):
    """
    NNClassifier is a specialized NNEstimator that simplifies the data format for
    classification tasks. It only supports label column of DoubleType, and the fitted
    NNClassifierModel will have the prediction column of DoubleType.
    """
    def __init__(self, model, criterion, sample_preprocessing, jvalue=None, bigdl_type="float"):
        """
        :param model: BigDL module to be optimized
        :param criterion: BigDL criterion method
        :param sample_preprocessing: Expert param. A Preprocessing that transforms the (feature, 
               Option[label]) tuple to a BigDL Sample[T], where T is decided by the BigDL model.

               Note that sample_preprocessing should be able to handle the case that label = None.
               During fit, NNEstimator will extract (feature, Option[label]) tuple from input 
               DataFrame and use sample_preprocessing to transform the tuple into BigDL Sample  
               to be ingested by the model. If Label column is not available, (feature, None)
               will be sent to sample_preprocessing.
 
               The sample_preprocessing will also be copied to the generated NNModel and applied
               to feature column during transform, where (feature, None) will be passed to the
               sample_preprocessing.
               Multiple Preprocessing can be combined as a ChainedPreprocessing.
        :param bigdl_type(optional): Data type of BigDL model, "float"(default) or "double".
        """
        super(NNClassifier, self).__init__(
            model, criterion, sample_preprocessing, jvalue, bigdl_type)

    @classmethod
    def create(cls, model, criterion, feature_preprocessing, jvalue=None, bigdl_type="float"):
        """
        Construct a NNEstimator with a feature_preprocessing and a label_Preprocessing, which
        convert the data in feature column and label column to Tensors (Multi-dimension array)
        for model. This is the the recommended constructor for most users.

        :param model: BigDL module to be optimized
        :param criterion: BigDL criterion method
        :param feature_Preprocessing: A Preprocessing that transforms the feature data to a
               Tensor[T]. Some pre-defined Preprocessing are provided in package
               zoo.feature. E.g.
               ArrayToTensor is used to transform Array[_] in DataFrame to Tensor. For a feature
               column that contains 576 floats in an Array, Users can set
               ArrayToTensor(Array(28, 28)) as feature_Preprocessing, which will convert the feature
               data into Tensors with dimension 28 * 28 to be processed by a convolution Model.
               For a simple linear model, user may just use ArrayToTensor(Array(576)), which will
               convert the data into Tensors with single dimension (576).
               MLlibVectorToTensor is used to transform org.apache.spark.mllib.linalg.Vector
               to a Tensor.
               ScalarToTensor transform a number to a Tensor with single dimension of length 1.
               Multiple Preprocessing can be combined as a ChainedPreprocessing.
        """
        return NNClassifier(model, criterion,
                            FeatureLabelPreprocessing(feature_preprocessing, ScalarToTensor()),
                            jvalue, bigdl_type)

    @classmethod
    def createWithSize(cls, model, criterion, feature_size, jvalue=None, bigdl_type="float"):
        """
        Construct a NNClassifier with a feature size. The constructor is useful
        when the feature column contains the following data types:
        Float, Double, Int, Array[Float], Array[Double], Array[Int] and MLlib Vector. The
        feature data are converted to Tensors with the specified sizes before sending
        to the model.
        :param model: BigDL Model to be trained.
        :param criterion: BigDL criterion.
        :param feature_size: The size (Tensor dimensions) of the feature data. e.g. an image
                            may be with width * height = 28 * 28, featureSize = Array(28, 28).
        :param label_size: The size (Tensor dimensions) of the label data.
        :param jvalue: Java object create by Py4j
        :param bigdl_type: optional parameter. data type of model, "float"(default) or "double".
        """
        return cls(model, criterion,
                   FeatureLabelPreprocessing(SeqToTensor(feature_size), ScalarToTensor()),
                   jvalue, bigdl_type)


class NNClassifierModel(NNModel):
    """
    NNClassifierModel is a specialized [[NNModel]] for classification tasks. The prediction
    column will have the datatype of Double.
    """
    def __init__(self,  model, feature_preprocessing, jvalue=None, bigdl_type="float"):
        """
        :param model: trained BigDL model to use in prediction.
        :param feature_Preprocessing: A Preprocessing that transforms the feature data to a
               Tensor[T]. Some pre-defined Preprocessing are provided in package
               zoo.feature. E.g.
               ArrayToTensor is used to transform Array[_] in DataFrame to Tensor. For a feature
               column that contains 576 floats in an Array, Users can set
               ArrayToTensor(Array(28, 28)) as feature_Preprocessing, which will convert the feature
               data into Tensors with dimension 28 * 28 to be processed by a convolution Model.
               For a simple linear model, user may just use ArrayToTensor(Array(576)), which will
               convert the data into Tensors with single dimension (576).
               MLlibVectorToTensor is used to transform org.apache.spark.mllib.linalg.Vector
               to a Tensor.
               ScalarToTensor transform a number to a Tensor with single dimension of length 1.
               Multiple Preprocessing can be combined as a ChainedPreprocessing.
        :param jvalue: Java object create by Py4j
        :param bigdl_type(optional): Data type of BigDL model, "float"(default) or "double".
        """
        super(NNClassifierModel, self).__init__(model, feature_preprocessing, jvalue, bigdl_type)

    @classmethod
    def of(self, jvalue, feaTran=None, bigdl_type="float"):
        model = NNClassifierModel(
            model=None, feature_preprocessing=feaTran, jvalue=jvalue, bigdl_type=bigdl_type)
        return model
