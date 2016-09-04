package com.intel.analytics.dllib.lib.nn

import com.intel.analytics.dllib.lib.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class SpatialBatchNormalization[@specialized(Float, Double) T:ClassTag](nOutput: Int,eps: Double = 1e-5,momentum: Double = 0.1,affine: Boolean = true)(implicit ev: TensorNumeric[T])
  extends BatchNormalization[T](nOutput, eps, momentum,affine) {
  override val nDim = 4

  override def toString(): String ={
    s"nn.SpatialBatchNormalization[${ev.getType()}]($nOutput, $eps, $momentum, $affine)"
  }
}
