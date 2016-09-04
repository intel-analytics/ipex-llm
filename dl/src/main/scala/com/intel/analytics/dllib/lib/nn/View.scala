package com.intel.analytics.dllib.lib.nn

import com.intel.analytics.dllib.lib.tensor.Tensor
import com.intel.analytics.dllib.lib.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

class View[@specialized(Float, Double) T:ClassTag](sizes : Array[Int])(implicit ev: TensorNumeric[T]) extends Module[T] {

  def getSize(): Array[Int] ={
    return sizes
  }

  val numElements = {
    var init = 1
    var inferDim = false
    var i = 0
    while(i < sizes.length) {
      if(sizes(i) >= 0) {
        init *= sizes(i)
      } else {
        require(sizes(i) == -1, "size should be positive or -1")
        require(!inferDim, "only one dimension should be -1")
        inferDim = true
      }
      i += 1
    }

    init
  }

  private var numInputDims = 0

  def setNumInputDims(numInputDims : Int) = {
    this.numInputDims = numInputDims
    this
  }

  private def batchSize(input : Tensor[T], size : Array[Int], numberInputDims : Int, numElements : Int) : Int = {
    val ind = input.nDimension()
    val isz = input.size()
    val maxDim = if(numberInputDims == 0) ind else numberInputDims

    var ine = 1
    var i = ind - 1
    while(i >= ind - maxDim) {
      ine *= isz(i)
      i -= 1
    }

    require(ine % numElements == 0, "input view doesn't match desired view")

    var bse = ine / numElements

    i = 0
    var break = false
    while(i < sizes.length && ! break) {
      if(sizes(i) == -1) {
        bse = 1
        break = true
      }
      i += 1
    }

    i = ind - maxDim - 1
    while(i >= 0) {
      bse *= isz(i)
      i -= 1
    }

    if(bse == 1 && (numberInputDims == 0 || input.nDimension() <= numberInputDims)) {
      -1
    } else {
      bse
    }
  }

  override def updateOutput(input : Tensor[T]) : Tensor[T] = {
    val bse = batchSize(input, this.sizes, this.numInputDims, this.numElements)
    if(bse != -1) {
      val newSizes = new Array[Int](this.sizes.length + 1)
      newSizes(0) = bse
      System.arraycopy(this.sizes, 0, newSizes, 1, this.sizes.length)
      this.output = input.view(newSizes)
    } else {
      this.output = input.view(this.sizes)
    }

    this.output
  }

  def this(s : Int*)(implicit ev: TensorNumeric[T]) = this(s.toArray)

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    this.gradInput = gradOutput.view(input.size)
    this.gradInput
  }

  override def toString() : String = {
    s"nn.View(${sizes.mkString("x")})"
  }
}
