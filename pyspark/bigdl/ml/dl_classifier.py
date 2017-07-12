import sys
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.param import *
from pyspark.sql.functions import *

from bigdl.util.common import *
import numpy as np

if sys.version >= '3':
    long = int
    unicode = str


class DLEstimator(JavaValue):
    def __init__(self,  model, criterion, featureSize, labelSize, bigdl_type="float"):
        super(DLEstimator, self).__init__(None, bigdl_type, model, criterion, featureSize, labelSize)

    def setFeaturesCol(self, featuresColName):
        callJavaFunc(get_spark_context(), self.value.setFeaturesCol, featuresColName)
        return self

    def setLabelCol(self, labelColName):
        callJavaFunc(get_spark_context(), self.value.setLabelCol, labelColName)
        return self

    def setPredictionCol(self, value):
        callJavaFunc(get_spark_context(), self.value.setPredictionCol, value)
        return self

    def setBatchSize(self, value):
        callJavaFunc(get_spark_context(), self.value.setBatchSize, value)
        return self

    def setMaxEpoch(self, value):
        callJavaFunc(get_spark_context(), self.value.setMaxEpoch, value)
        return self

    def getMaxEpoch(self):
        return callJavaFunc(get_spark_context(), self.value.getMaxEpoch)

    def fit(self, dataset):
        jmodel = callBigDlFunc(self.bigdl_type, "fit", self.value, dataset)
        return DLModel(jmodel, self.bigdl_type)


class DLModel(JavaValue):
    def __init__(self,  model, featureSize, bigdl_type="float"):
        super(DLModel, self).__init__(None, bigdl_type, model, featureSize)

    def setFeaturesCol(self, featuresColName):
        callJavaFunc(get_spark_context(), self.value.setFeaturesCol, featuresColName)
        return self

    def setPredictionCol(self, value):
        callJavaFunc(get_spark_context(), self.value.setPredictionCol, value)
        return self

    def setFeatureSize(self, value):
        callBigDlFunc(self.bigdl_type, "dlmodelSetFeatureSize", self.value, value)
        return self

    def setBatchSize(self, value):
        callJavaFunc(get_spark_context(), self.value.setBatchSize, value)
        return self

    def transform(self, dataset):
        return callJavaFunc(get_spark_context(), self.value.transform, dataset)


class DLClassifier(DLEstimator):
    def __init__(self,  model, criterion, featureSize, bigdl_type="float"):
        super(DLClassifier, self).__init__(None, bigdl_type, model, criterion, featureSize)

    def fit(self, dataset):
        jmodel = callBigDlFunc(self.bigdl_type, "fit", self.value, dataset)
        return DLClassifierModel(jmodel, self.bigdl_type)


class DLClassifierModel(DLModel):
    def __init__(self,  model, featureSize, bigdl_type="float"):
        super(DLClassifierModel, self).__init__(None, bigdl_type, model, featureSize)


# class DLClassifier(Transformer, HasInputCol, HasOutputCol, Param):
#     """
#     A general Classifier to classify the input data in inputCol, and write the results to outputCol.
#     Need to create classifier with BigDL model and input column shape.
#     Use setInputCol to set inputCol name, and use setOutputCol to set outputCol name.
#     """
#
#     def __init__(self, model, sample_shape):
#         super(DLClassifier, self).__init__()
#         self.model = Param(self, "model", "model trained.")
#         self.setModel(model)
#         self.sample_shape = Param(self, "sample_shape", "shape of the sample.")
#         self.setSampleShape(sample_shape)
#
#     def setModel(self, value):
#         """
#         Sets the value of Model.
#         """
#         self._paramMap[self.model] = value
#         return self
#
#     def getModel(self):
#         """
#         Gets the value of model or its default value.
#         """
#         return self.getOrDefault(self.model)
#
#     def setSampleShape(self, value):
#         """
#         Sets the value of input column shape.
#         """
#         self._paramMap[self.sample_shape] = value
#         return self
#
#     def getSampleShape(self):
#         """
#         Gets the value of input column shape or its default value.
#         """
#         return self.getOrDefault(self.sample_shape)
#
#     def setInputCol(self, input_col_name):
#         """
#         Sets the name of input column.
#         """
#         super(DLClassifier, self).setInputCol(input_col_name)
#         return self
#
#     def setOutputCol(self, output_col_name):
#         """
#         Sets the name of output column.
#         """
#         super(DLClassifier, self).setOutputCol(output_col_name)
#         return self
#
#     def _transform(self, df):
#         # get parameters
#         model = self.getModel()
#         sample_shape = self.getSampleShape()
#         input_col = self.getInputCol()
#         output_col = self.getOutputCol()
#         df.cache()
#
#         feature_rdd = df.rdd.map(lambda row: row[input_col])
#         sample_rdd = df.rdd.map(
#             lambda row: Sample(row[input_col].toArray(), [-1], sample_shape, [1]))
#         result_rdd = model.predict(sample_rdd).map(lambda result: np.asscalar(np.argmax(result)+1))
#         result_feature_rdd = result_rdd.zip(feature_rdd)
#         result_df = df.sql_ctx.createDataFrame(result_feature_rdd, [output_col, input_col])
#         out_df = df.join(result_df, [input_col])
#         return out_df
