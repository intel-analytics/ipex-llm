package com.intel.analytics.sparkdl.nn

import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
  * Created by xianyan on 10/26/16.
  */
class SmoothL1Criterion[T: ClassTag](sizeAverage: Boolean = true)
                                    (implicit ev: TensorNumeric[T]) extends Criterion[T] {
  var gradInput: Tensor[T] = Tensor[T]()

  var buffer = Tensor[T]()

  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    require(input.nElement() == target.nElement())
    buffer.resizeAs(input).zero()
    buffer.copy(input)
    var sum = (buffer - target).abs().apply1(z =>
      if (ev.toType[Double](z) < 1)
        ev.times(ev.fromType[Double](0.5), ev.times(z, z))
      else
        ev.minus(z, ev.fromType[Double](0.5)))
      .sum()
    if (sizeAverage) {
      sum = ev.divide(sum, ev.fromType(input.nElement()))
    }
    sum
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    require(input.nElement() == target.nElement())
    val norm = ev.fromType(if (sizeAverage) 1.0 / input.nElement() else 1.0)
    gradInput.resizeAs(input)
    gradInput.copy(input)
    (gradInput - target).apply1(x => {
      if (ev.isGreater(ev.negative(x), ev.fromType(1))) ev.negative(norm)
      else if (ev.isGreater(x, ev.fromType(1))) norm
      else ev.times(norm, x)
    })
  }
}
