package com.intel.webscaleml.nn.nn

import com.intel.webscaleml.nn.tensor.Tensor
import com.intel.webscaleml.nn.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class Sequential[@specialized(Float, Double) T: ClassTag](implicit ev: TensorNumeric[T]) extends Container[T] {

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    var i = 0
    var result = input
    while (i < modules.length) {
      result = modules(i).forward(result)
      i += 1
    }
    this.output = result
    result
  }

  override def updateGradInput(input: Tensor[T], nextError: Tensor[T]): Tensor[T] = {
    var i = modules.length - 1
    var error = nextError
    while (i > 0) {
      val input = modules(i - 1).output
      error = modules(i).backward(input, error)
      i -= 1
    }
    error = modules(0).backward(input, error)
    this.gradInput = error
    error
  }

  override def equals(obj : Any) : Boolean = {
    if(!super.equals(obj)) {
      return false
    }

    if(!obj.isInstanceOf[Sequential[T]])
      return false
    val other = obj.asInstanceOf[Sequential[T]]
    if(this.eq(other))
      return true

    if(this.modules.length != other.modules.length)
      return false

    val moduleLength = modules.length
    var i = 0
    while(i < moduleLength) {
      if(modules(i) != other.modules(i)) {
        return false
      }
      i += 1
    }

    true
  }

  override def toString() : String = {
    val tab = "  "

    s"nn.Sequential {${line + tab}[input -> ${modules.zipWithIndex.map{case (m : Module[T], i : Int) => "(" + (i + 1) + ")"}.mkString(" -> ")} -> output]${line + tab}" +
      s"${modules.zipWithIndex.map{ case (model : Module[T], index : Int) => s"(${index + 1}): ${model.setLine(line + tab)}"}.mkString(line + tab)}$line}"
  }
}


