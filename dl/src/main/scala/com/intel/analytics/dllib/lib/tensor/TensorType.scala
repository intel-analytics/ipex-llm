package com.intel.analytics.dllib.lib.tensor

object TensorType {
  sealed trait DataType

  type DoubleReal = Double
  type FloatReal = Float

  object DoubleType extends DataType
  object FloatType extends DataType
}
