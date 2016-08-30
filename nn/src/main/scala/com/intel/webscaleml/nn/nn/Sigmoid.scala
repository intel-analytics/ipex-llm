package com.intel.webscaleml.nn.nn

import com.intel.webscaleml.nn.tensor.TensorNumericMath.TensorNumeric

import scala.math.exp
import com.intel.webscaleml.nn.tensor.Tensor

import scala.reflect.ClassTag

class Sigmoid[@specialized(Float, Double) T:ClassTag](implicit ev: TensorNumeric[T]) extends Module[T]{

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output.resizeAs(input)
    output.map(input,(_,i)=>ev.divide(ev.fromType[Int](1), ev.plus(ev.fromType[Int](1), ev.exp(ev.negative(i)))))
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] ={
    gradInput.resizeAs(input)
    gradInput.copy(gradOutput)
    gradInput.map(output,(g,z)=>ev.times(ev.times(g,ev.minus(ev.fromType[Int](1), z)), z))
    gradInput
  }

  override def toString() : String = {
    s"nn.Sigmoid"
  }
}
