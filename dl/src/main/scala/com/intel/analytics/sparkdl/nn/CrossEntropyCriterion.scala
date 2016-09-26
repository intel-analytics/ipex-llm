package com.intel.analytics.sparkdl.nn

import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.tensor.Tensor

import scala.reflect.ClassTag


/**
  * Created by ywan on 16-9-21.
  */
class CrossEntropyCriterion[T: ClassTag](var weights: Tensor[T] = null)
                                        (implicit ev: TensorNumeric[T]) extends Criterion[T] {
  var gradInput: Tensor[T] = Tensor[T]()
  var total_weight = ev.fromType[Int](0)
  //val eps = ev.fromType[Double](1e-12)
  if (weights != null) require(weights.dim() == 1, "weights input should be 1-D Tensor")

  var nll = new ClassNLLCriterion(weights)
  var lsm = new LogSoftMax()

  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {

    var lsmOutput = lsm.updateOutput(input)
    return nll.updateOutput(lsmOutput, target)
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    val size = input.size
    input.squeeze()
    target.squeeze()
    var lsmOutput = lsm.updateOutput(input)
    var nllGrad = nll.updateGradInput(lsmOutput, target)
    var lsmGrad = lsm.updateGradInput(input, nllGrad)
    this.gradInput = lsmGrad.view(size)
    this.gradInput

  }
  override def toString(): String = {
    s"nn.CrossEntropy"
  }
}
