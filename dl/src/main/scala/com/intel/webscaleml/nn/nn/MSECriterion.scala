package com.intel.webscaleml.nn.nn

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import com.intel.webscaleml.nn.tensor.TensorNumericMath.TensorNumeric

import com.intel.webscaleml.nn.tensor.{torch, Tensor}

import scala.reflect.ClassTag

class MSECriterion[@specialized(Float, Double) T: ClassTag](implicit ev: TensorNumeric[T]) extends Criterion[T]{
  var gradInput:Tensor[T] = torch.Tensor[T]()
  var sizeAverage = true
  override def updateOutput(input:Tensor[T], target:Tensor[T]):T ={
    output = ev.fromType[Int](0)

    input.map(target,(a,b)=>{output = ev.plus(output,ev.times(ev.minus(a,b),ev.minus(a,b)));a})
    if(sizeAverage) output = ev.divide(output, ev.fromType[Int](input.nElement()))
    output
  }

  override  def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(input)
    var norm = ev.fromType[Int](2)
    if (sizeAverage) {
      norm = ev.fromType[Double](2.0/input.nElement())
    }
    gradInput.copy(input)
    gradInput.map(target, (a,b)=>ev.times(norm, ev.minus(a,b)))
    gradInput
  }

}
