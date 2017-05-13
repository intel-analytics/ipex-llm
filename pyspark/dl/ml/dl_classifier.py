import sys

from util.common import JavaValue
from util.common import callBigDlFunc
from util.common import callJavaFunc
from pyspark import SparkContext
from pyspark.ml import Transformer
from pyspark.ml.param.shared import Param, HasInputCol, HasOutputCol
from pyspark.ml.classification import LogisticRegressionModel

from util.common import Sample
import numpy as np

if sys.version >= '3':
    long = int
    unicode = str

class DLClassifier(Transformer, HasInputCol, HasOutputCol):
    """
    Criterion is helpful to train a neural network.
    Given an input and a target, they compute a gradient according to a given loss function.
    """

    def set_input_col(self, input_col_name):
        self.setInputCol(input_col_name)
        return self

    def set_output_col(self, output_col_name):
        self.setOutputCol(output_col_name)
        return self

    def _transform(self, dataframe):
        # get model
        model = self.getOrDefault(self.getParam("model_train"))
        batch_shape = self.getOrDefault(self.getParam("batch_shape"))
        input_col = self.getInputCol()
        output_col = self.getOutputCol()
        sample_rdd = dataframe.select(input_col).rdd.map(
            lambda features: Sample(features, [-1], (batch_shape(1), batch_shape(2), batch_shape(3)), [1]))
        result_rdd = model.predict(sample_rdd)
        out_df = dataframe.sql_ctx.createDataFrame(dataframe.rdd.zip(result_rdd), dataframe.columns.append(output_col))
        return out_df



