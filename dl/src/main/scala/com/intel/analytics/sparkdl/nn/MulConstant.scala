package com.intel.analytics.sparkdl.nn

import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
  * Created by yao on 9/21/16.
  */
class MulConstant[@specialized(Float, Double) T: ClassTag](
  constantScalar:T,
  ip: Boolean = false)
  (implicit ev: TensorNumeric[T]) extends Module[T] {

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    if (ip) {
      input.mul(constantScalar)
      output.set(input)
    } else {
      output.resizeAs(input)
            .copy(input)
            .mul(constantScalar)
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    if (ip) {
      gradOutput.mul(constantScalar)
      gradInput.set(gradOutput)
      input.div(constantScalar)
    } else {
      gradInput = gradInput.resizeAs(gradOutput)
        .copy(gradOutput)
        .mul(constantScalar)
    }
    gradInput
  }
}
