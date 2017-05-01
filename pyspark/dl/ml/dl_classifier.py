import sys

from util.common import JavaValue
from util.common import callBigDlFunc
from util.common import callJavaFunc
from pyspark import SparkContext
from util.common import JTensor
import numpy as np

if sys.version >= '3':
    long = int
    unicode = str

class DLClassifier(JavaValue):
    """
    Criterion is helpful to train a neural network.
    Given an input and a target, they compute a gradient according to a given loss function.
    """
    def __init__(self, bigdl_type='float'):
        JavaValue.__init__(self, None, bigdl_type)

    def set_input_col(self, input_col_name):
        callJavaFunc(SparkContext.getOrCreate(), self.value.setInputCol, input_col_name)
        return self

    def set_output_col(self, output_col_name):
        callJavaFunc(SparkContext.getOrCreate(), self.value.setOutputCol, output_col_name)
        return self

    def transform(self, dataframe, param_map):
        return callBigDlFunc(self.bigdl_type,
                      "transform",
                      self.value,
                      dataframe, param_map)


