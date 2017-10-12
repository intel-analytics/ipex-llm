from pyspark.ml.pipeline import Estimator, Model
from pyspark.ml.param.shared import *
from bigdl.util.common import *
from pyspark.ml.wrapper import JavaParams

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

    def setBatchSize(self, value):
        """
        Sets the value of :py:attr:`batchSize`.
        """
        self._paramMap[self.batchSize] = value
        return self

    def getBatchSize(self):
        """
        Gets the value of batchSize or its default value.
        """
        return self.getOrDefault(self.batchSize)


class DLEstimator(JavaParams, Estimator, HasFeaturesCol, HasLabelCol, HasPredictionCol, HasBatchSize):
    maxEpoch = Param(Params._dummy(), "maxEpoch", "number of max Epoch")
    def jvm_class_constructor(self):
        name = "create" + self.__class__.__name__
        print("creating: " + name)
        return name

    def __init__(self,  model, criterion, featureSize, labelSize, bigdl_type="float"):
        super(DLEstimator, self).__init__()
        self.value = callBigDlFunc(
            bigdl_type, self.jvm_class_constructor(), model, criterion, featureSize, labelSize)
        self.bigdl_type = bigdl_type
        self.maxEpoch = Param(self, "maxEpoch", "number of max Epoch")
        self._setDefault(maxEpoch=100)

    def setMaxEpoch(self, value):
        self._set(maxEpoch=value)
        return self

    def getMaxEpoch(self):
        return self.getOrDefault(self.maxEpoch)

    def _fit(self, dataset):
        #self._transfer_params_to_java()
        jmodel = callBigDlFunc(self.bigdl_type, "fitEstimator", self.value, dataset)
        model = DLModel.of(jmodel, self.bigdl_type)
        return model


class DLModel(JavaParams, Model, HasFeaturesCol, HasPredictionCol, HasBatchSize):
    def jvm_class_constructor(self):
        name = "create" + self.__class__.__name__
        print("creating: " + name)
        return name

    def __init__(self,  model, featureSize, jvalue=None, bigdl_type="float"):
        super(DLModel, self).__init__()
        self.value = jvalue if jvalue else callBigDlFunc(
            bigdl_type, self.jvm_class_constructor(), model, featureSize)
        self.bigdl_type = bigdl_type

    def setFeatureSize(self, value):
        callBigDlFunc(self.bigdl_type, "dlmodelSetFeatureSize", self.value, value)
        return self

    def getFeatureSize(self):
        return callBigDlFunc(self.bigdl_type, "dlmodelGetFeatureSize", self.value)

    def _transform(self, dataset):
        #self._transfer_params_to_java()
        #return callJavaFunc(get_spark_context(), self.value.transform, dataset)
        return callBigDlFunc(self.bigdl_type, "dlmodelTransform", self.value, dataset)

    @classmethod
    def of(cls, jvalue, bigdl_type="float"):
        model = DLModel(model=None, featureSize=None, jvalue=jvalue, bigdl_type=bigdl_type)
        #model._transfer_params_from_java()
        return model


class DLClassifier(DLEstimator):
    def __init__(self,  model, criterion, featureSize,  bigdl_type="float"):
        super(DLClassifier, self).__init__(model, criterion, featureSize, [1], bigdl_type)

    def _fit(self, dataset):
        #self._transfer_params_to_java()
        jmodel = callBigDlFunc(self.bigdl_type, "fitClassifier", self.value, dataset)
        model = DLClassifierModel.of(jmodel, self.bigdl_type)
        return model


class DLClassifierModel(DLModel):
    def __init__(self,  model, featureSize, jvalue=None, bigdl_type="float"):
        super(DLClassifierModel, self).__init__(model, featureSize, jvalue, bigdl_type)

    @classmethod
    def of(cls, jvalue, bigdl_type="float"):
        model = DLClassifierModel(model=None, featureSize=None, jvalue=jvalue, bigdl_type=bigdl_type)
        #model._transfer_params_from_java()
        return model