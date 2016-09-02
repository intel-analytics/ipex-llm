package com.intel.analytics.dllib.lib.nn

import com.intel.analytics.dllib.lib.tensor.Tensor
import com.intel.analytics.dllib.lib.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class Transpose[@specialized(Float, Double) T:ClassTag](val permutations : Array[(Int, Int)])(implicit ev: TensorNumeric[T]) extends Module[T] {

  override def updateOutput(input: Tensor[T]): Tensor[T] ={
    output.resizeAs(input).copy(input)
    var i = 0
    while(i < permutations.length) {
      output.transpose(permutations(i)._1, permutations(i)._2)
      i += 1
    }
    output
  }
  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(gradOutput).copy(gradOutput)
    var i = permutations.length - 1
    while(i >= 0) {
      gradInput.transpose(permutations(i)._1, permutations(i)._2)
      i -= 1
    }
    gradInput
  }

  override def toString() : String = {
    s"nn.Transpose(${permutations.map{case (from : Int, to : Int) => s"$from -> $to"}.mkString(", ")})"
  }
}
