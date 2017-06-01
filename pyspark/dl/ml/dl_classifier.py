import sys
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.param import *
from pyspark.sql.functions import *

from util.common import Sample
import numpy as np

if sys.version >= '3':
    long = int
    unicode = str


class DLClassifier(Transformer, HasInputCol, HasOutputCol, Param):
    """
    A general Classifier to classify the input data in inputCol, and write the results to outputCol.
    Need to create classifier with BigDL model and input column shape.
    Use setInputCol to set inputCol name, and use setOutputCol to set outputCol name.
    """

    def __init__(self, model, sample_shape):
        super(DLClassifier, self).__init__()
        self.model = Param(self, "model", "model trained.")
        self.setModel(model)
        self.sample_shape = Param(self, "sample_shape", "shape of the sample.")
        self.setSampleShape(sample_shape)

    def setModel(self, value):
        """
        Sets the value of Model.
        """
        self._paramMap[self.model] = value
        return self

    def getModel(self):
        """
        Gets the value of model or its default value.
        """
        return self.getOrDefault(self.model)

    def setSampleShape(self, value):
        """
        Sets the value of input column shape.
        """
        self._paramMap[self.sample_shape] = value
        return self

    def getSampleShape(self):
        """
        Gets the value of input column shape or its default value.
        """
        return self.getOrDefault(self.sample_shape)

    def setInputCol(self, input_col_name):
        """
        Sets the name of input column.
        """
        super(DLClassifier, self).setInputCol(input_col_name)
        return self

    def setOutputCol(self, output_col_name):
        """
        Sets the name of output column.
        """
        super(DLClassifier, self).setOutputCol(output_col_name)
        return self

    def _transform(self, df):
        # get parameters
        model = self.getModel()
        sample_shape = self.getSampleShape()
        input_col = self.getInputCol()
        output_col = self.getOutputCol()
        df.cache()

        feature_rdd = df.rdd.map(lambda row: row[input_col])
        sample_rdd = df.rdd.map(
            lambda row: Sample(row[input_col].toArray(), [-1], sample_shape, [1]))
        result_rdd = model.predict(sample_rdd).map(lambda result: np.asscalar(np.argmax(result)+1))
        result_feature_rdd = result_rdd.zip(feature_rdd)
        result_df = df.sql_ctx.createDataFrame(result_feature_rdd, [output_col, input_col])
        out_df = df.join(result_df, [input_col])
        return out_df
