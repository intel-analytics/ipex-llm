package com.intel.analytics.dllib.lib.nn

import com.intel.analytics.dllib.lib.tensor.TensorNumericMath.TensorNumeric

import scala.math.tanh
import com.intel.analytics.dllib.lib.tensor._

import scala.reflect.ClassTag

class Tanh[@specialized(Float, Double) T:ClassTag](implicit ev: TensorNumeric[T]) extends Module[T] {
  override def updateOutput(input:Tensor[T]):Tensor[T] ={
    output.resizeAs(input)
    output.map(input, (_, inputVal)=>ev.fromType[Double](tanh(ev.toType[Double](inputVal)))) //todo, better to support apply_2
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] ={
    gradInput.resizeAs(gradOutput)
    gradInput.copy(gradOutput)  //todo, better to support apply_3
    gradInput.map(output,(gradValue,outputValue)=>ev.times(gradValue, ev.minus(ev.fromType[Int](1), ev.times(outputValue,outputValue))))
    gradInput
  }

  override def toString() : String = {
    s"nn.Tanh"
  }
}


