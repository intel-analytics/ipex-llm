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

import json
from bigdl.dllib.nn.layer import Layer
from pyspark.ml.param.shared import *
from pyspark.ml.wrapper import JavaModel, JavaEstimator, JavaTransformer
from pyspark.ml.util import MLWritable, MLReadable, JavaMLWriter
from bigdl.dllib.optim.optimizer import SGD
from bigdl.dllib.utils.file_utils import callZooFunc, put_local_file_to_remote
from bigdl.dllib.utils.common import *
from bigdl.dllib.feature.common import *
from bigdl.dllib.nncontext import init_nncontext
from bigdl.dllib.utils.log4Error import *


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


class HasSamplePreprocessing:
    """
    Mixin for param samplePreprocessing
    """
    samplePreprocessing = None

    def __init__(self):
        super(HasSamplePreprocessing, self).__init__()

    def setSamplePreprocessing(self, val):
        """
        Sets samplePreprocessing
        """
        pythonBigDL_method_name = "setSamplePreprocessing"
        callZooFunc(self.bigdl_type, pythonBigDL_method_name, self.value, val)
        self.samplePreprocessing = val
        return self

    def getSamplePreprocessing(self):
        return self.samplePreprocessing


class HasOptimMethod:

    def __init__(self):
        super(HasOptimMethod, self).__init__()
        self.optimMethod = SGD()

    def setOptimMethod(self, val):
        """
        Sets optimization method. E.g. SGD, Adam, LBFGS etc. from bigdl.optim.optimizer.
        default: SGD()
        """
        pythonBigDL_method_name = "setOptimMethod"
        callZooFunc(self.bigdl_type, pythonBigDL_method_name, self.value, val)
        self.optimMethod = val
        return self

    def getOptimMethod(self):
        """
        Gets the optimization method
        """
        return self.optimMethod


class HasThreshold(Params):
    """
    Mixin for param Threshold in binary classification.

    The threshold applies to the raw output of the model. If the output is greater than
    threshold, then predict 1, else 0. A high threshold encourages the model to predict 0
    more often; a low threshold encourages the model to predict 1 more often.

    Note: the param is different from the one in Spark ProbabilisticClassifier which is compared
    against estimated probability.

    Default is 0.5.
    """

    def __init__(self):
        super(HasThreshold, self).__init__()
        self.threshold = Param(self, "threshold", "threshold")
        self._setDefault(threshold=0.5)

    def setThreshold(self, val):
        """
        Sets the value of :py:attr:`threshold`.
        """
        self._paramMap[self.threshold] = val
        return self

    def getThreshold(self):
        """
        Gets the value of threshold or its default value.
        """
        return self.getOrDefault(self.threshold)


class NNEstimator(JavaEstimator, HasFeaturesCol, HasLabelCol, HasPredictionCol, HasBatchSize,
                  HasOptimMethod, HasSamplePreprocessing, JavaValue):
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

    More concrete examples are available in package com.intel.analytics.bigdl.dllib.example.nnframes
    """

    def __init__(self, model, criterion,
                 feature_preprocessing=None,
                 label_preprocessing=None,
                 jvalue=None, bigdl_type="float"):
        """
        Construct a NNEstimator with BigDL model, criterion and Preprocessing for feature and label
        data.
        :param model: BigDL Model to be trained.
        :param criterion: BigDL criterion.
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
        :param jvalue: Java object create by Py4j
        :param bigdl_type: optional parameter. data type of model, "float"(default) or "double".
        """
        super(NNEstimator, self).__init__()

        # avoid initialization during import.
        if not feature_preprocessing:
            feature_preprocessing = SeqToTensor()
        if not label_preprocessing:
            label_preprocessing = SeqToTensor()

        if type(feature_preprocessing) is list:
            if type(feature_preprocessing[0]) is list:
                feature_preprocessing = SeqToMultipleTensors(feature_preprocessing)
            elif isinstance(feature_preprocessing[0], int):
                feature_preprocessing = SeqToTensor(feature_preprocessing)

        if type(label_preprocessing) is list:
            invalidInputError((all(isinstance(x, int) for x in label_preprocessing)),
                              "some elements in label_preprocessing is not integer")
            label_preprocessing = SeqToTensor(label_preprocessing)

        sample_preprocessing = FeatureLabelPreprocessing(feature_preprocessing, label_preprocessing)

        self.value = jvalue if jvalue else callZooFunc(
            bigdl_type, self.jvm_class_constructor(), model, criterion, sample_preprocessing)
        self.model = model
        self.samplePreprocessing = sample_preprocessing
        self.bigdl_type = bigdl_type
        self._java_obj = self.value

        self.maxEpoch = Param(self, "maxEpoch", "number of max Epoch")
        self.learningRate = Param(self, "learningRate", "learning rate")
        self.learningRateDecay = Param(self, "learningRateDecay", "learning rate decay")
        self.cachingSample = Param(self, "cachingSample", "cachingSample")

        self.train_summary = None
        self.validation_config = None
        self.checkpoint_config = None
        self.validation_summary = None
        self.endWhen = None
        self.dataCacheLevel = "DRAM"

    def setSamplePreprocessing(self, val):
        """
        Sets the value of sample_preprocessing
        :param val: a Preprocesing[(Feature, Option(Label), Sample]
        """
        super(NNEstimator, self).setSamplePreprocessing(val)
        return self

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

    def setEndWhen(self, trigger):
        """
        When to stop the training, passed in a Trigger. E.g. maxIterations(100)
        """
        pythonBigDL_method_name = "setEndWhen"
        callZooFunc(self.bigdl_type, pythonBigDL_method_name, self.value, trigger)
        self.endWhen = trigger
        return self

    def getEndWhen(self):
        """
        Gets the value of endWhen or its default value.
        """
        return self.endWhen

    def setDataCacheLevel(self, level, numSlice=None):
        """
        :param level: string, "DRAM" or "DISK_AND_DRAM".
                If it's DRAM, will cache dataset into dynamic random-access memory
                If it's DISK_AND_DRAM, will cache dataset into disk, and only hold 1/numSlice
                  of the data into memory during the training. After going through the
                  1/numSlice, we will release the current cache, and load another slice into
                  memory.
        """
        pythonBigDL_method_name = "setDataCacheLevel"
        callZooFunc(self.bigdl_type, pythonBigDL_method_name, self.value, level, numSlice)
        self.dataCacheLevel = level if numSlice is None else (level, numSlice)
        return self

    def getDataCacheLevel(self):
        return self.dataCacheLevel

    def setLearningRate(self, val):
        """
        Sets the value of :py:attr:`learningRate`.
        .. note:: Deprecated in 0.4.0. Please set learning rate with optimMethod directly.
        """
        self._paramMap[self.learningRate] = val
        return self

    def getLearningRate(self):
        """
        Gets the value of learningRate or its default value.
        """
        return self.getOrDefault(self.learningRate)

    def setLearningRateDecay(self, val):
        """
        Sets the value of :py:attr:`learningRateDecay`.
        .. note:: Deprecated in 0.4.0. Please set learning rate decay with optimMethod directly.
        """
        self._paramMap[self.learningRateDecay] = val
        return self

    def getLearningRateDecay(self):
        """
        Gets the value of learningRateDecay or its default value.
        """
        return self.getOrDefault(self.learningRateDecay)

    def setCachingSample(self, val):
        """
        whether to cache the Samples after preprocessing. Default: True
        """
        self._paramMap[self.cachingSample] = val
        return self

    def isCachingSample(self):
        """
        Gets the value of cachingSample or its default value.
        """
        return self.getOrDefault(self.cachingSample)

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
        callZooFunc(self.bigdl_type, pythonBigDL_method_name, self.value, val)
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
        callZooFunc(self.bigdl_type, pythonBigDL_method_name, self.value, val)
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
        callZooFunc(self.bigdl_type, pythonBigDL_method_name, self.value,
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

    def _setNNBatchSize(self, batch_size):
        """
        Set BatchSize in NNEstimator directly instead of Deserialized from python object
        For evaluating use ONLY
        """
        pythonBigDL_method_name = "setNNBatchSize"
        callZooFunc(self.bigdl_type, pythonBigDL_method_name, self.value,
                    batch_size)
        return self

    def _setNNFeaturesCol(self, feature_cols):
        """
        Set FeaturesCol in NNEstimator directly instead of Deserialized from python object
        For evaluating use ONLY
        """
        pythonBigDL_method_name = "setNNFeaturesCol"
        callZooFunc(self.bigdl_type, pythonBigDL_method_name, self.value,
                    feature_cols)
        return self

    def _setNNLabelCol(self, label_cols):
        """
        Set LabelCol in NNEstimator directly instead of Deserialized from python object
        For evaluating use ONLY
        """
        pythonBigDL_method_name = "setNNLabelCol"
        callZooFunc(self.bigdl_type, pythonBigDL_method_name, self.value,
                    label_cols)
        return self

    def clearGradientClipping(self):
        """
        Clear clipping params, in this case, clipping will not be applied.
        In order to take effect, it needs to be called before fit.
        """
        callZooFunc(self.bigdl_type, "nnEstimatorClearGradientClipping",
                    self.value)
        return self

    def setConstantGradientClipping(self, min, max):
        """
        Set constant gradient clipping during the training process.
        In order to take effect, it needs to be called before fit.

        # Arguments
        min: The minimum value to clip by. Float.
        max: The maximum value to clip by. Float.
        """
        callZooFunc(self.bigdl_type, "nnEstimatorSetConstantGradientClipping",
                    self.value,
                    float(min),
                    float(max))
        return self

    def setGradientClippingByL2Norm(self, clip_norm):
        """
        Clip gradient to a maximum L2-Norm during the training process.
        In order to take effect, it needs to be called before fit.

        # Arguments
        clip_norm: Gradient L2-Norm threshold. Float.
        """
        callZooFunc(self.bigdl_type, "nnEstimatorSetGradientClippingByL2Norm",
                    self.value,
                    float(clip_norm))
        return self

    def setCheckpoint(self, path, trigger, isOverWrite=True):
        """
        Set check points during training. Not enabled by default
        :param path: the directory to save the model
        :param trigger: how often to save the check point
        :param isOverWrite: whether to overwrite existing snapshots in path. Default is True
        :return: self
        """
        pythonBigDL_method_name = "setCheckpoint"
        callZooFunc(self.bigdl_type, pythonBigDL_method_name, self.value,
                    path, trigger, isOverWrite)
        self.checkpoint_config = [path, trigger, isOverWrite]
        return self

    def getCheckpoint(self):
        """
        :return: a tuple containing (checkpointPath, checkpointTrigger, checkpointOverwrite)
        """
        return self.checkpoint_config

    def _create_model(self, java_model):
        # explicity reset SamplePreprocessing even though java_model already has the preprocessing,
        # so that python NNModel also has sample_preprocessing
        estPreprocessing = self.getSamplePreprocessing()
        model = Layer.from_jvalue(java_model.getModel(), bigdl_type=self.bigdl_type)
        nnModel = NNModel(model=model, feature_preprocessing=None, jvalue=java_model,
                          bigdl_type=self.bigdl_type) \
            .setSamplePreprocessing(ChainedPreprocessing([ToTuple(), estPreprocessing]))

        nnModel.setFeaturesCol(self.getFeaturesCol()) \
            .setPredictionCol(self.getPredictionCol()) \
            .setBatchSize(java_model.getBatchSize())
        return nnModel

    def setFeaturesCol(self, value):
        """
        Sets the value of :py:attr:`featuresCol`.
        """
        return self._set(featuresCol=value)

    def setPredictionCol(self, value):
        """
        Sets the value of :py:attr:`predictionCol`.
        """
        return self._set(predictionCol=value)

    def setLabelCol(self, value):
        """
        Sets the value of :py:attr:`labelCol`.
        """
        return self._set(labelCol=value)

    def _eval(self, val_data):
        """
        Call Evaluating process.

        :param val_data: validation data. Spark DataFrame
        """
        pythonBigDL_method_name = "internalEval"
        result = callZooFunc(self.bigdl_type, pythonBigDL_method_name, self.value,
                             val_data)
        return result


class NNModel(JavaTransformer, MLWritable, MLReadable, HasFeaturesCol, HasPredictionCol,
              HasBatchSize, HasSamplePreprocessing, JavaValue):
    """
    NNModel extends Spark ML Transformer and supports BigDL model with Spark DataFrame.

    NNModel supports different feature data type through Preprocessing. Some common
    Preprocessing have been defined in com.intel.analytics.bigdl.dllib.feature.

    After transform, the prediction column contains the output of the model as Array[T], where
    T (Double or Float) is decided by the model type.
    """

    def __init__(self, model, feature_preprocessing=None, jvalue=None, bigdl_type="float"):
        """
        create a NNModel with a BigDL model
        :param model: trained BigDL model to use in prediction.
        :param feature_preprocessing: The param converts the data in feature column to a
                                      Tensor. It expects a List of Int as
                                      the size of the converted Tensor, or a
                                      Preprocessing[F, Tensor[T]]
        :param jvalue: Java object create by Py4j
        :param bigdl_type: optional parameter. data type of model, "float"(default) or "double".
        """
        super(NNModel, self).__init__()
        # initialize with Java NNModel
        if jvalue:
            invalidInputError(feature_preprocessing is None,
                              "feature_preprocessing cannot be None")
            self.value = jvalue
        # initialize with Python Model and preprocessing
        else:
            if not feature_preprocessing:
                feature_preprocessing = SeqToTensor()

            if type(feature_preprocessing) is list:
                if type(feature_preprocessing[0]) is list:
                    feature_preprocessing = SeqToMultipleTensors(feature_preprocessing)
                elif isinstance(feature_preprocessing[0], int):
                    feature_preprocessing = SeqToTensor(feature_preprocessing)

            sample_preprocessing = ChainedPreprocessing([feature_preprocessing, TensorToSample()])
            self.value = callZooFunc(
                bigdl_type, self.jvm_class_constructor(), model, sample_preprocessing)
            self.samplePreprocessing = sample_preprocessing

        self.model = model
        self._java_obj = self.value
        self.bigdl_type = bigdl_type
        self.setBatchSize(self.value.getBatchSize())

    def write(self):
        return NNModelWriter(self)

    @staticmethod
    def load(path):
        jvalue = callZooFunc("float", "loadNNModel", path)
        return NNModel(model=None, feature_preprocessing=None, jvalue=jvalue)

    def setFeaturesCol(self, value):
        """
        Sets the value of :py:attr:`featuresCol`.
        """
        return self._set(featuresCol=value)

    def setPredictionCol(self, value):
        """
        Sets the value of :py:attr:`predictionCol`.
        """
        return self._set(predictionCol=value)

    def getModel(self):
        return self.model


class NNModelWriter(JavaMLWriter):
    def __init__(self, instance):
        super(NNModelWriter, self).__init__(instance)

    def save(self, path):
        """Save the ML instance to the input path."""
        super(NNModelWriter, self).save(path)
        sc = init_nncontext()
        # change class name in metadata to python class name
        metadata_path = os.path.join(path, "metadata")
        metadataStr = sc.textFile(metadata_path, 1).first()
        metadata = json.loads(metadataStr)
        py_type = metadata['class'].replace("com.intel.analytics.zoo", "zoo")
        metadata['class'] = py_type
        metadata_json = json.dumps(metadata, separators=[',', ':'])
        # replace old metadata
        temp_dir = tempfile.mkdtemp()
        temp_meta_path = os.path.join(temp_dir, "metadata")
        sc.parallelize([metadata_json], 1).saveAsTextFile(temp_meta_path)
        for file in os.listdir(temp_meta_path):
            put_local_file_to_remote(os.path.join(temp_meta_path, file),
                                     os.path.join(metadata_path, file), True)
        import shutil
        shutil.rmtree(temp_dir)


class NNClassifier(NNEstimator):
    """
    NNClassifier is a specialized NNEstimator that simplifies the data format for
    classification tasks. It only supports label column of DoubleType, and the fitted
    NNClassifierModel will have the prediction column of DoubleType.
    """

    def __init__(self, model, criterion, feature_preprocessing=None,
                 jvalue=None, bigdl_type="float"):
        """
        :param model: BigDL module to be optimized
        :param criterion: BigDL criterion method
        :param feature_preprocessing: The param converts the data in feature column to a
                                      Tensor. It expects a List of Int as
                                      the size of the converted Tensor, or a
                                      Preprocessing[F, Tensor[T]]
        :param bigdl_type(optional): Data type of BigDL model, "float"(default) or "double".
        """
        if not feature_preprocessing:
            feature_preprocessing = SeqToTensor()

        super(NNClassifier, self).__init__(
            model, criterion, feature_preprocessing, ScalarToTensor(), jvalue, bigdl_type)

    def setSamplePreprocessing(self, val):
        """
        Sets the value of sample_preprocessing
        :param val: a Preprocesing[(Feature, Option(Label), Sample]
        """
        super(NNClassifier, self).setSamplePreprocessing(val)
        return self

    def _create_model(self, java_model):
        # explicity reset SamplePreprocessing even though java_model already has the preprocessing,
        # so that python NNClassifierModel also has sample_preprocessing
        estPreprocessing = self.getSamplePreprocessing()
        model = Layer.from_jvalue(java_model.getModel(), bigdl_type=self.bigdl_type)
        classifierModel = NNClassifierModel(model=model, feature_preprocessing=None,
                                            jvalue=java_model, bigdl_type=self.bigdl_type) \
            .setSamplePreprocessing(ChainedPreprocessing([ToTuple(), estPreprocessing]))

        classifierModel.setFeaturesCol(self.getFeaturesCol()) \
            .setPredictionCol(self.getPredictionCol()) \
            .setBatchSize(java_model.getBatchSize())
        return classifierModel


class NNClassifierModel(NNModel, HasThreshold):
    """
    NNClassifierModel is a specialized [[NNModel]] for classification tasks. The prediction
    column will have the datatype of Double.
    """

    def __init__(self, model, feature_preprocessing=None, jvalue=None,
                 bigdl_type="float"):
        """
        :param model: trained BigDL model to use in prediction.
        :param feature_preprocessing: The param converts the data in feature column to a
                                      Tensor. It expects a List of Int as
                                      the size of the converted Tensor, or a
                                      Preprocessing[F, Tensor[T]]
        :param jvalue: Java object create by Py4j
        :param bigdl_type(optional): Data type of BigDL model, "float"(default) or "double".
        """
        super(NNClassifierModel, self).__init__(model, feature_preprocessing, jvalue, bigdl_type)

    @staticmethod
    def load(path):
        jvalue = callZooFunc("float", "loadNNClassifierModel", path)
        return NNClassifierModel(model=None, feature_preprocessing=None, jvalue=jvalue)
