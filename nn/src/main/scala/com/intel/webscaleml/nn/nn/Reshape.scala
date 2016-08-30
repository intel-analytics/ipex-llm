package com.intel.webscaleml.nn.nn

import com.intel.webscaleml.nn.tensor.Tensor
import com.intel.webscaleml.nn.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class Reshape[@specialized(Float, Double) T: ClassTag](size: Array[Int], var batchMode:Option[Boolean] = None)(implicit ev: TensorNumeric[T]) extends Module[T]{
  val batchSize = new Array[Int](size.length+1)
  var nElement:Int = 1
  for(i<-1 to size.length) {
    batchSize(i) = size(i - 1)
    nElement *= size(i - 1)
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] ={

    if((batchMode.nonEmpty && batchMode.get == false) || (input.nElement()==nElement && batchMode.isEmpty && input.size(1) != 1)){
      require(input.nElement()==nElement, "element number must match Reshape size")
      if(input.isContiguous()) output = input.view(size) else output = input.contiguous().view(size)
    }
    else {
      require(input.nElement()==nElement*input.size(1), "element number must match Reshape size")
      batchSize(0) = input.size(1)
      if(input.isContiguous()) output = input.view(batchSize) else output = input.contiguous().view(batchSize)
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] ={
    if(gradOutput.isContiguous()) gradInput = gradOutput.view(input.size()) else gradInput = gradOutput.contiguous().view(input.size())
    gradInput
  }

  override def equals(obj : Any) : Boolean = {

    if(!super.equals(obj)) {
      return false
    }

    if(!obj.isInstanceOf[Reshape[T]])
      return false
    val other = obj.asInstanceOf[Reshape[T]]
    if(this.eq(other))
      return true

    var i = 0
    while(i < batchSize.length) {
      if(batchSize(i) != other.batchSize(i)) {
        return false
      }
      i += 1
    }
    nElement == other.nElement &&
      batchMode == other.batchMode
  }

  override def toString() : String = {
    s"nn.Reshape(${size.mkString("x")})"
  }
}
