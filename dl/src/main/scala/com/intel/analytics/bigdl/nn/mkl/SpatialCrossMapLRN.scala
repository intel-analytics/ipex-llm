/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.nn.mkl

import com.intel.analytics.bigdl.mkl.MKL
import com.intel.analytics.bigdl.nn.{Module, TensorModule}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.RandomGenerator._

import scala.reflect.ClassTag
import scala.language.implicitConversions

class SpatialCrossMapLRN[@specialized(Float, Double) T: ClassTag](
    val size: Int = 5,
    val alpha: Double = 1.0,
    val beta: Double = 0.75,
    val k: Double = 1.0)(implicit ev: TensorNumeric[T])
    extends TensorModule[T] {

  private val scale = Tensor[T]()
  private val paddedSquare = Tensor[T]()
  private val paddedRatio = Tensor[T]()
  private val accumRatio = Tensor[T]()
  private val accumRatioTimeInput = Tensor[T]()

  require(size % 2 == 1, "LRN only supports odd values for size")
  val prePad = (size - 1) / 2

  var classPtr = 0L
  private var firstPass = true

  override def getClassPtr(): Long = classPtr

  override def equals(obj: Any): Boolean = {
    if (!super.equals(obj)) {
      return false
    }

    if (!obj.isInstanceOf[SpatialCrossMapLRN[T]]) { return false }
    val other = obj.asInstanceOf[SpatialCrossMapLRN[T]]
    if (this.eq(other)) { return true }

    size == other.size &&
    alpha == other.alpha && beta == other.beta && k == other.k
  }

  override def hashCode(): Int = {
    val seed = 37
    var hash = super.hashCode()
    hash = hash * seed + size.hashCode()
    hash = hash * seed + alpha.hashCode()
    hash = hash * seed + beta.hashCode()
    hash = hash * seed + k.hashCode()

    hash
  }

  override def toString(): String = {
    s"mkl.SpatialCrossMapLRN($size, $alpha, $beta, $k)"
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.nDimension() == 4,
            "Input must have 4 dimensions, corresponding to (batch, channels, height, width)")
    require(input.isContiguous(), "Input is not contiguous")

    output.resizeAs(input)

    val inputOffset = input.storageOffset() - 1;
    val outputOffset = output.storageOffset() - 1;

    val inputNumber = input.size(1)
    val inputChannel = input.size(2)
    val inputHeight = if (input.dim() <= 2) 1 else input.size(3)
    val inputWidth = if (input.dim() <= 3) 1 else input.size(4)
    // TODO we may set input.size(input.dim() - 3) == 1 if input.dim() == 3

    if (firstPass) {
      ev.getType() match {
        case "Float" =>
          classPtr = MKL.LRNInitFloat(inputNumber,
                                      inputChannel,
                                      inputHeight,
                                      inputWidth,
                                      size,
                                      alpha.toFloat,
                                      beta.toFloat,
                                      k.toFloat,
                                      4)
        case "Double" =>
          classPtr = MKL.LRNInitDouble(inputNumber,
                                       inputChannel,
                                       inputHeight,
                                       inputWidth,
                                       size,
                                       alpha.toDouble,
                                       beta.toDouble,
                                       k.toDouble,
                                       4)
        case _ =>
          throw new UnsupportedOperationException(s"Only Float/Double supported")
      }
      firstPass = false
    }

    if (initForward) {
      this.updateMklOut()
      this.initForward = false
    }

    implicit def bool2int(b: Boolean) = if (b) 1 else 0
    ev.getType() match {
      case "Float" =>
        MKL.LRNForwardFloat(
          input.storage().array().asInstanceOf[Array[Float]],
          inputOffset,
          output.storage().array().asInstanceOf[Array[Float]],
          outputOffset,
          classPtr
        )
      case "Double" =>
        MKL.LRNForwardDouble(
          input.storage().array().asInstanceOf[Array[Double]],
          inputOffset,
          output.storage().array().asInstanceOf[Array[Double]],
          outputOffset,
          classPtr
        )
      case _ =>
        throw new UnsupportedOperationException(s"Only Float/Double supported")
    }

    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    require(input.nDimension() == 4,
            "Input must have 4 dimensions, corresponding to (batch, channels, height, width)")
    require(gradOutput.isContiguous(), "gradOutput is not contiguous")

    gradInput.resizeAs(input)

    val inputOffset = input.storageOffset() - 1;
    val outputOffset = output.storageOffset() - 1;

    val inputNumber = input.size(1)
    val inputChannel = input.size(2)
    val inputHeight = if (input.dim() <= 2) 1 else input.size(3)
    val inputWidth = if (input.dim() <= 3) 1 else input.size(4)
    // TODO we may set input.size(input.dim() - 3) == 1 if input.dim() == 3

    val gradOutputOffset = gradOutput.storageOffset() - 1
    val gradInputOffset = gradInput.storageOffset() - 1

    ev.getType() match {
      case "Float" =>
        MKL.LRNBackwardFloat(input.storage().array().asInstanceOf[Array[Float]],
                             inputOffset,
                             gradOutput.storage().array().asInstanceOf[Array[Float]],
                             gradOutputOffset,
                             gradInput.storage().array().asInstanceOf[Array[Float]],
                             gradInputOffset,
                             classPtr)
      case "Double" =>
        MKL.LRNBackwardDouble(input.storage().array().asInstanceOf[Array[Double]],
                              inputOffset,
                              gradOutput.storage().array().asInstanceOf[Array[Double]],
                              gradOutputOffset,
                              gradInput.storage().array().asInstanceOf[Array[Double]],
                              gradInputOffset,
                              classPtr)
      case _ =>
        throw new UnsupportedOperationException(s"Only Float/Double supported")
    }
    if (initBackward) {
      updateMklGradInput()
      initBackward = false
    }

    gradInput
  }
}
