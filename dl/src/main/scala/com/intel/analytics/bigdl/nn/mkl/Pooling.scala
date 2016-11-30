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
import com.intel.analytics.bigdl.utils.RandomGenerator
import com.intel.analytics.bigdl.tensor.Tensor

import scala.language.implicitConversions
import scala.reflect.ClassTag

class SpatialPooling[@specialized(Float, Double) T: ClassTag](
    val kernelWidth: Int,
    val kernelHeight: Int,
    val strideWidth: Int,
    val strideHeight: Int,
    val padWidth: Int = 0,
    val padHeight: Int = 0)(implicit ev: TensorNumeric[T])
  extends TensorModule[T] {
  implicit def bool2int(b: Boolean): Int = if (b) 1 else 0

  var classPtr: Long = 0L
  private var firstPass = true

  override def getClassPtr(): Long = classPtr

  // algorithm = 0 -> max
  // algorithm = 0 -> avg
  val algorithm : Int = 0

  // TODO just for adopt to the testcase
  var ceil_mode = false
  def ceil(): SpatialPooling[T] = {
    ceil_mode = true
    this
  }

  def floor(): SpatialPooling[T] = {
    ceil_mode = false
    this
  }

  def this(kernelWidth: Int, kernelHeight: Int)(implicit ev: TensorNumeric[T]) {
    this(kernelWidth, kernelHeight, kernelWidth, kernelHeight)
  }

  // compute the output height and width
  def computeOut(input: Int, pad: Int, kernel: Int, stride: Int): Int = {
    if (ceil_mode) {
      math.ceil(1.0 * (input + 2 * pad - kernel) / stride).toInt + 1
    } else {
      math.floor(1.0 * (input + 2 * pad - kernel) / stride).toInt + 1
    }
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(input)

    val inputOffset = input.storageOffset() - 1;
    val outputOffset = output.storageOffset() - 1;
    val gradInputOffset = gradInput.storageOffset() - 1;
    val gradOutputOffset = gradOutput.storageOffset() - 1;

    val inputWidth = input.size(input.dim())
    val inputHeight = input.size(input.dim() - 1)
    val inputChannel = input.size(input.dim() - 2)
    val inputNumber = if (input.dim() == 3) 1 else input.size(input.dim() - 3)
    // TODO we may set input.size(input.dim() - 3) == 1 if input.dim() == 3

    val outputHeight =
      computeOut(inputHeight, padHeight, kernelHeight, strideHeight)
    val outputWidth =
      computeOut(inputWidth, padHeight, kernelWidth, strideWidth)
    val outputChannel = inputChannel
    val outputNumber = inputNumber

    ev.getType() match {
      case "Float" =>
        MKL.PoolingBackwardFloat(input.storage().array().asInstanceOf[Array[Float]],
                                 inputOffset,
                                 gradOutput.storage().array().asInstanceOf[Array[Float]],
                                 gradOutputOffset,
                                 gradInput.storage().array().asInstanceOf[Array[Float]],
                                 gradInputOffset,
                                 classPtr)
      case "Double" =>
        MKL.PoolingBackwardDouble(input.storage().array().asInstanceOf[Array[Double]],
                                  inputOffset,
                                  gradOutput.storage().array().asInstanceOf[Array[Double]],
                                  gradOutputOffset,
                                  gradInput.storage().array().asInstanceOf[Array[Double]],
                                  gradOutputOffset,
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

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val inputWidth = input.size(input.dim())
    val inputHeight = input.size(input.dim() - 1)
    val inputChannel = input.size(input.dim() - 2)
    val inputNumber = if (input.dim() == 3) 1 else input.size(input.dim() - 3)
    // TODO we may set input.size(input.dim() - 3) == 1 if input.dim() == 3

    val outputHeight =
      computeOut(inputHeight, padHeight, kernelHeight, strideHeight)
    val outputWidth =
      computeOut(inputWidth, padWidth, kernelWidth, strideWidth)
    val outputChannel = inputChannel
    val outputNumber = inputNumber

    val inputOffset = input.storageOffset() - 1;
    val outputOffset = output.storageOffset() - 1;

    if (input.dim() == 3) {
      output.resize(Array(outputChannel, outputHeight, outputWidth))
    } else {
      output.resize(Array(outputNumber, outputChannel, outputHeight, outputWidth))
    }

    // TODO algorithm = 0 means using MAX
    if (firstPass) {
      ev.getType() match {
        case "Float" =>
          classPtr = MKL.PoolingInitFloat(inputNumber,
                                          inputChannel,
                                          inputHeight,
                                          inputWidth,
                                          kernelHeight,
                                          kernelWidth,
                                          strideHeight,
                                          strideWidth,
                                          padHeight,
                                          padWidth,
                                          4,
                                          ceil_mode,
                                          algorithm,
                                          this.getName())
        case "Double" =>
          classPtr = MKL.PoolingInitDouble(inputNumber,
                                           inputChannel,
                                           inputHeight,
                                           inputWidth,
                                           kernelHeight,
                                           kernelWidth,
                                           strideHeight,
                                           strideWidth,
                                           padHeight,
                                           padWidth,
                                           4,
                                           ceil_mode,
                                           algorithm,
                                           this.getName())
        case _ =>
          throw new UnsupportedOperationException(s"Only Float/Double supported")
      }

      firstPass = false
    }

    if (initForward) {
      this.updateMklOut()
      this.initForward = false
    }

    ev.getType() match {
      case "Float" =>
        MKL.PoolingForwardFloat(input.storage().array.asInstanceOf[Array[Float]],
                                inputOffset,
                                output.storage().array.asInstanceOf[Array[Float]],
                                outputOffset,
                                classPtr)
      case "Double" =>
        MKL.PoolingForwardDouble(input.storage().array.asInstanceOf[Array[Double]],
                                 inputOffset,
                                 output.storage().array.asInstanceOf[Array[Double]],
                                 outputOffset,
                                 classPtr)
      case _ =>
        throw new UnsupportedOperationException(s"Only Float/Double supported")
    }
    output
  }

  override def toString(): String = {
    s"mkl.Pooling"
  }

}

class SpatialMaxPooling[T: ClassTag](kernelWidth: Int,
                                     kernelHeight: Int,
                                     strideWidth: Int,
                                     strideHeight: Int,
                                     padWidth: Int = 0,
                                     padHeight: Int = 0)(implicit ev: TensorNumeric[T])
    extends SpatialPooling[T](kernelWidth,
                              kernelHeight,
                              strideWidth,
                              strideHeight,
                              padWidth,
                              padHeight) {
  override val algorithm: Int = 0
  def this(kernelWidth: Int, kernelHeight: Int)(implicit ev: TensorNumeric[T]) {
    this(kernelWidth, kernelHeight, kernelWidth, kernelHeight)
  }
  override def toString(): String = {
    s"""mkl.SpatialMaxPooling($kernelWidth, $kernelHeight, $strideWidth, $strideHeight,
       |$padWidth, $padHeight)""".stripMargin.replaceAll("\n", " ")
  }
}

class SpatialAveragePooling[T: ClassTag](kernelWidth: Int,
                                         kernelHeight: Int,
                                         strideWidth: Int,
                                         strideHeight: Int,
                                         padWidth: Int = 0,
                                         padHeight: Int = 0)(implicit ev: TensorNumeric[T])
    extends SpatialPooling[T](kernelWidth,
                              kernelHeight,
                              strideWidth,
                              strideHeight,
                              padWidth,
                              padHeight) {
  override val algorithm: Int = 1
  def this(kernelWidth: Int, kernelHeight: Int)(implicit ev: TensorNumeric[T]) {
    this(kernelWidth, kernelHeight, kernelWidth, kernelHeight)
  }
  override def toString(): String = {
    s"""mkl.SpatialAveragePooling($kernelWidth, $kernelHeight,$strideWidth, $strideHeight,
       |$padWidth, $padHeight)""".stripMargin.replaceAll("\n", " ")
  }
}
