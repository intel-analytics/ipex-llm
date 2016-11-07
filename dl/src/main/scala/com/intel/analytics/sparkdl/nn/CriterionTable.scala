package com.intel.analytics.sparkdl.nn

import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class CriterionTable[T: ClassTag](val criterion: TensorCriterion[T])
 (implicit ev: TensorNumeric[T]) extends  TensorCriterion[T] {

  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    output  = criterion.updateOutput(input, target)
    output
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    var gradInput: Tensor[T] = Tensor[T]()
    gradInput = criterion.updateGradInput(input, target)
    gradInput
  }

  override def toString(): String = {
    s"nn.CriterionTable"
  }
}
