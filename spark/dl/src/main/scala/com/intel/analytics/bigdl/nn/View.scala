/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

/**
 * This module creates a new view of the input tensor using the sizes passed to the constructor.
 * The method setNumInputDims() allows to specify the expected number of dimensions of the inputs
 * of the modules. This makes it possible to use minibatch inputs
 * when using a size -1 for one of the dimensions.
 * @param sizes sizes use for creates a new view
 */
@SerialVersionUID(1238814703013238333L)
class View[T: ClassTag](val sizes: Array[Int])(
  implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  def getSize(): Array[Int] = {
    return sizes
  }

  val numElements = {
    var init = 1
    var inferDim = false
    var i = 0
    while (i < sizes.length) {
      if (sizes(i) >= 0) {
        init *= sizes(i)
      } else {
        require(sizes(i) == -1, "size should be positive or -1" +
          s"size ${sizes(i)}")
        require(!inferDim, "only one dimension should be -1")
        inferDim = true
      }
      i += 1
    }

    init
  }

  private var numInputDims = 0

  def setNumInputDims(numInputDims: Int): this.type = {
    this.numInputDims = numInputDims
    this
  }

  def getNumInputDims(): Int = numInputDims

  private def batchSize(ind: Int, isz: Array[Int],
                    size: Array[Int], numberInputDims: Int, numElements: Int) : Int = {
    val maxDim = if (numberInputDims == 0) ind else numberInputDims

    var ine = 1
    var i = ind - 1
    while (i >= ind - maxDim) {
      ine *= isz(i)
      i -= 1
    }

    require(ine % numElements == 0, "input view doesn't match desired view" +
      s"inputElemnts $ine numeberofElements $numElements")

    var bse = ine / numElements

    i = 0
    var break = false
    while (i < sizes.length && !break) {
      if (sizes(i) == -1) {
        bse = 1
        break = true
      }
      i += 1
    }

    i = ind - maxDim - 1
    while (i >= 0) {
      bse *= isz(i)
      i -= 1
    }

    if (bse == 1 && (numberInputDims == 0 || ind <= numberInputDims)) {
      -1
    } else {
      bse
    }
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val bse = batchSize(input.nDimension(), input.size(), this.sizes,
      this.numInputDims, this.numElements)
    if (bse != -1) {
      val newSizes = new Array[Int](this.sizes.length + 1)
      newSizes(0) = bse
      System.arraycopy(this.sizes, 0, newSizes, 1, this.sizes.length)
      this.output = input.view(newSizes)
    } else {
      this.output = input.view(this.sizes)
    }

    this.output
  }

  def this(s: Int*)(implicit ev: TensorNumeric[T]) = this(s.toArray)

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    this.gradInput = gradOutput.view(input.size)
    this.gradInput
  }

  override def toString(): String = {
    s"${getPrintName}(${sizes.mkString("x")})"
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray
    val bse = batchSize(input.length, input, this.sizes, this.numInputDims, this.numElements)

    if (bse != -1) {
      val newSizes = new Array[Int](this.sizes.length + 1)
      newSizes(0) = bse
      System.arraycopy(this.sizes, 0, newSizes, 1, this.sizes.length)
      Shape(newSizes)
    } else {
      Shape(this.sizes)
    }
  }
}

object View {
  def apply[@specialized(Float, Double) T: ClassTag](
      sizes: Int*)(implicit ev: TensorNumeric[T]) : View[T] = {
    new View[T](sizes.toArray)
  }

  def apply[@specialized(Float, Double) T: ClassTag](
      sizes: Array[Int])(implicit ev: TensorNumeric[T]) : View[T] = {
    new View[T](sizes)
  }
}
