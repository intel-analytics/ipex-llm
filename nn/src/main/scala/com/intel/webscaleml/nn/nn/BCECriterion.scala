package com.intel.webscaleml.nn.nn

import com.intel.webscaleml.nn.tensor.TensorNumericMath.TensorNumeric
import com.intel.webscaleml.nn.tensor.{DenseTensorApply, torch, Tensor}

import scala.reflect.ClassTag

class BCECriterion[@specialized(Float, Double) T:ClassTag](var weights: Tensor[T] = null, sizeAverage:Boolean = true)
                                                          (implicit ev: TensorNumeric[T]) extends Criterion[T]{
  var gradInput:Tensor[T] = torch.Tensor[T]()
  var total_weight = ev.fromType[Int](0)
  val eps = ev.fromType[Double](1e-12)
  if(weights != null) require(weights.dim() == 1, "weights input should be 1-D Tensor")

  var buffer = torch.Tensor[T]()
  override  def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    // - log(input) * target - log(1 - input) * (1 - target)

    require(input.nElement() == target.nElement())
    buffer.resizeAs(input).zero()

    if(null != weights && target.dim() != 1){
      weights = weights.view(1, target.size(2)).expandAs(target)
    }

    buffer.add(input).add(eps)
    buffer.apply1(ev.log(_))

    if(null != weights) buffer.cmul(weights)

    output = target.dot(buffer)

    buffer.mul(input, ev.fromType[Int](-1)).add(ev.fromType[Int](1)).add(eps).apply1(ev.log(_))
    if(null != weights) buffer.cmul(weights)

    output = ev.plus(output, buffer.sum())
    output = ev.minus(output, target.dot(buffer))

    if (sizeAverage) output = ev.divide(output, ev.fromType[Int](input.nElement()))

    output = ev.negative(output)

    output
  }

  override  def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    /*
       - (target - input) / ( input (1 - input) )
       The gradient is slightly incorrect:
       It should have be divided by (input + eps) (1 - input + eps)
       but it is divided by input (1 - input + eps) + eps
       This modification requires less memory to be computed.
     */
    require(input.nElement() == target.nElement())

    if(null != weights && target.dim() != 1){
      weights = weights.view(1, target.size(2)).expandAs(target)
    }

    buffer.resizeAs(input)
    buffer.zero()
    // -x ( 1 + eps - x) + eps
    buffer.add(ev.fromType[Int](-1)).add(input).add(ev.negative(eps)).cmul(input).add(ev.negative(eps))

    gradInput.resizeAs(input)
    // y - x
    gradInput.add(target, ev.fromType[Int](-1), input)
    //- (y - x) / ( x ( 1 + eps -x ) + eps )
    gradInput = gradInput / buffer

    if (null != weights) gradInput.cmul(weights)

    if(sizeAverage) gradInput.div(ev.fromType[Int](target.nElement()))

    gradInput
  }
}
