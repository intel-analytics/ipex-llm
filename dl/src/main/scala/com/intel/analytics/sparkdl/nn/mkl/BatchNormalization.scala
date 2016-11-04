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

package com.intel.analytics.sparkdl.nn.mkl

import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.utils.RandomGenerator._
import com.intel.analytics.sparkdl.nn.{Module, TensorModule}
import com.intel.analytics.sparkdl.mkl.MKL

import scala.language.implicitConversions
import scala.reflect.ClassTag

class SpatialBatchNormalization[@specialized(Float, Double) T: ClassTag](
    val nOutput: Int,
    val eps: Double = 1e-5,
    val momentum: Double = 0.1,
    val affine: Boolean = true)(implicit ev: TensorNumeric[T])
  extends TensorModule[T] {
  require(nOutput > 0,
          "To set affine=false call SpatialBatchNormalization(nFeature,  eps, momentum, false)")

  val nDim = 2
  val runningMean = Tensor[T](nOutput)
  val runningVar = Tensor[T](nOutput).fill(ev.fromType[Int](1))
  val saveMean = Tensor[T](nOutput)
  val saveStd = Tensor[T](nOutput).fill(ev.fromType[Int](1))

  private var classPtr = 0L
  private var firstPass = true

  override def getClassPtr(): Long = classPtr

  val weight: Tensor[T] = if (affine) Tensor[T](nOutput) else null
  val bias: Tensor[T] = if (affine) Tensor[T](nOutput) else null
  gradWeight = if (affine) Tensor[T](nOutput) else null
  gradBias = if (affine) Tensor[T](nOutput) else null

  val useWeight: Boolean = if (weight != null) true else false
  val useBias: Boolean = if (bias != null) true else false

  if (affine) {
    reset()
  }

  override def reset(): Unit = {
    if (null != weight) {
      weight.apply1(_ => ev.fromType[Double](RNG.uniform(0, 1)))
    }

    if (null != bias) {
      bias.fill(ev.fromType[Int](0))
    }

    runningMean.zero()
    runningVar.fill(ev.fromType[Int](1))
  }

  def checkInputDim(input: Tensor[T]): Unit = {
    require(input.dim() == nDim,
            s"only mini-batch supported (${nDim}D tensor), got ${input.dim()}D tensor instead")
    require(input.size(2) == runningMean.nElement(),
            s"got ${input.size(2)}-feature tensor, expected ${runningMean.nElement()}")
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output.resizeAs(input)

    val inputOffset = input.storageOffset() - 1;
    val outputOffset = output.storageOffset() - 1;

    val inputNumber = input.size(1)
    val inputChannel = input.size(2)
    val inputHeight = if (input.dim() <= 2) 1 else input.size(3)
    val inputWidth = if (input.dim() <= 3) 1 else input.size(4)
    // TODO we may set input.size(input.dim() - 3) == 1 if input.dim() == 3

    val kernelOffset = weight.storageOffset() - 1
    val biasOffset = bias.storageOffset() - 1

    implicit def bool2int(b: Boolean) = if (b) 1 else 0
    if (firstPass) {
      ev.getType() match {
        case "Float" =>
          classPtr = MKL.BatchNormInitFloat(inputNumber,
                                            inputChannel,
                                            inputHeight,
                                            inputWidth,
                                            eps.toFloat,
                                            useWeight,
                                            useBias,
                                            4,
                                            this.getName())
        case "Double" =>
          classPtr = MKL.BatchNormInitDouble(inputNumber,
                                             inputChannel,
                                             inputHeight,
                                             inputWidth,
                                             eps,
                                             useWeight,
                                             useBias,
                                             4,
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
        MKL.BatchNormForwardFloat(input.storage().array().asInstanceOf[Array[Float]],
                                  inputOffset,
                                  output.storage().array().asInstanceOf[Array[Float]],
                                  outputOffset,
                                  weight.storage().array().asInstanceOf[Array[Float]],
                                  kernelOffset,
                                  bias.storage().array().asInstanceOf[Array[Float]],
                                  biasOffset,
                                  classPtr)
      case "Double" =>
        MKL.BatchNormForwardDouble(input.storage().array().asInstanceOf[Array[Double]],
                                   inputOffset,
                                   output.storage().array().asInstanceOf[Array[Double]],
                                   outputOffset,
                                   weight.storage().array().asInstanceOf[Array[Double]],
                                   kernelOffset,
                                   bias.storage().array().asInstanceOf[Array[Double]],
                                   biasOffset,
                                   classPtr)
      case _ =>
        throw new UnsupportedOperationException(s"Only Float/Double supported")
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(input)

    val inputOffset = input.storageOffset() - 1;
    val outputOffset = output.storageOffset() - 1;

    val inputNumber = input.size(1)
    val inputChannel = input.size(2)
    val inputHeight = if (input.dim() <= 2) 1 else input.size(3)
    val inputWidth = if (input.dim() <= 3) 1 else input.size(4)
    // TODO we may set input.size(input.dim() - 3) == 1 if input.dim() == 3

    val kernelOffset = weight.storageOffset() - 1
    val biasOffset = bias.storageOffset() - 1

    val kernelDiffOffset = gradWeight.storageOffset() - 1
    val biasDiffOffset = gradBias.storageOffset() - 1

    val gradOutputOffset = gradOutput.storageOffset() - 1
    val gradInputOffset = gradInput.storageOffset() - 1

    implicit def bool2int(b: Boolean) = if (b) 1 else 0
    ev.getType() match {
      case "Float" =>
        MKL.BatchNormBackwardFloat(input.storage().array().asInstanceOf[Array[Float]],
                                   inputOffset,
                                   gradOutput.storage().array().asInstanceOf[Array[Float]],
                                   gradOutputOffset,
                                   gradInput.storage().array().asInstanceOf[Array[Float]],
                                   gradInputOffset,
                                   gradWeight.storage().array().asInstanceOf[Array[Float]],
                                   kernelDiffOffset,
                                   gradBias.storage().array().asInstanceOf[Array[Float]],
                                   biasDiffOffset,
                                   classPtr)
      case "Double" =>
        MKL.BatchNormBackwardDouble(input.storage().array().asInstanceOf[Array[Double]],
                                    inputOffset,
                                    gradOutput.storage().array().asInstanceOf[Array[Double]],
                                    gradOutputOffset,
                                    gradInput.storage().array().asInstanceOf[Array[Double]],
                                    gradInputOffset,
                                    gradWeight.storage().array().asInstanceOf[Array[Double]],
                                    kernelDiffOffset,
                                    gradBias.storage().array().asInstanceOf[Array[Double]],
                                    biasDiffOffset,
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

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T], scale: Double): Unit = {}

  override def updateParameters(learningRate: T): Unit = {
    weight.map(gradWeight, (a, b) => ev.minus(a, ev.times(learningRate, b)))
    bias.map(gradBias, (a, b) => ev.minus(a, ev.times(learningRate, b)))
  }

  override def zeroGradParameters(): Unit = {
    gradWeight.zero()
    gradBias.zero()
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(this.weight, this.bias), Array(this.gradWeight, this.gradBias))
  }

  override def toString(): String = {
    s"mkl.SpatialBatchNormalization[${ev.getType()}]($nOutput, $eps, $momentum, $affine)"
  }
}

class BatchNormalization[@specialized(Float, Double) T: ClassTag](
    nOutput: Int,
    eps: Double = 1e-5,
    momentum: Double = 0.1,
    affine: Boolean = true)(implicit ev: TensorNumeric[T])
    extends SpatialBatchNormalization[T](nOutput, eps, momentum, affine) {
  override def toString(): String = {
    s"mkl.BatchNormalization[${ev.getType()}]($nOutput, $eps, $momentum, $affine)"
  }
}
