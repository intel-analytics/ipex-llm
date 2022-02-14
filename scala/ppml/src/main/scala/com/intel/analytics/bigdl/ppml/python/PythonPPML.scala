package com.intel.analytics.bigdl.ppml.python

import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.python.api.PythonBigDL

import scala.reflect.ClassTag

class PythonPPML[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonBigDL {

}
