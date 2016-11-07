package com.intel.analytics.sparkdl.nn

import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class CrossEntropyCriterion[T: ClassTag](
   val weights: Tensor[T] = null )(implicit ev: TensorNumeric[T]) extends TensorCriterion[T]{
  private val gradInput: Tensor[T] = Tensor[T]()
  val lsm = new LogSoftMax[T]()
  val nll = new ClassNLLCriterion[T](weights)
  var _gradInput = Tensor[T]()

  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    input.squeeze()
    lsm.updateOutput(input)
    nll.updateOutput(lsm.output, target.asInstanceOf[Tensor[T]])
    output = nll.output
    output
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    val size = input.size()
    input.squeeze()

    _gradInput = nll.updateGradInput(lsm.output, target)
    lsm.updateGradInput(input, _gradInput)
    gradInput.resizeAs(lsm.gradInput).copy(lsm.gradInput).view(size)
    gradInput
  }

  override def toString(): String = {
    s"nn.CrossEntropyCriterion"
  }
}
