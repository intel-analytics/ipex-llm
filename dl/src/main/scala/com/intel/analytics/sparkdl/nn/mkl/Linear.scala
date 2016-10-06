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

import com.intel.analytics.sparkdl.mkl.MKL
import com.intel.analytics.sparkdl.nn.{Default, InitializationMethod, Module, Xavier}
import com.intel.analytics.sparkdl.utils.RandomGenerator._
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.tensor.Tensor

import scala.reflect.ClassTag

class Linear[@specialized(Float, Double) T: ClassTag](
    inputSize: Int,
    outputSize: Int,
    val needCompute: Boolean = true,
    private var initMethod: InitializationMethod = Default
)(implicit ev: TensorNumeric[T])
    extends Module[T] {
  val weight: Tensor[T] = Tensor[T](outputSize, inputSize)
  val bias: Tensor[T] = Tensor[T](outputSize)
  val addBuffer: Tensor[T] = Tensor[T]()
  this.gradWeight = Tensor[T](outputSize, inputSize)
  this.gradBias = Tensor[T](outputSize)

  private var classPtr = 0L
  private var firstPass = true

  override def getClassPtr(): Long = classPtr

  reset()

  def setInitMethod(initMethod: InitializationMethod): this.type = {
    this.initMethod = initMethod
    this
  }

  override def reset(): Unit = {
    initMethod match {
      case Default =>
        val stdv = 1.0 / math.sqrt(weight.size(2)) // todo, better to support uniform
        weight.apply1(_ => ev.fromType[Double](RNG.uniform(0, 1) * 2 * stdv - stdv))
        bias.apply1(_ => ev.fromType[Double](RNG.uniform(0, 1) * 2 * stdv - stdv))
      case Xavier =>
        val fanIn = weight.size(2)
        val fanOut = weight.size(1)
        val stdv = math.sqrt(3 / (fanIn + fanOut)) // todo, better to support uniform
        weight.apply1(_ => ev.fromType[Double](RNG.uniform(0, 1) * 2 * stdv - stdv))
        bias.fill(ev.fromType(0))
      case _ =>
        throw new UnsupportedOperationException(s"Only Default / Xavier supported")
    }
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim() == 2, "only batch mode supported")

    val inputWidth = input.size(input.dim())
    val inputHeight = input.size(input.dim() - 1)

    val nFrame = input.size(1)
    val nElement = output.nElement
    output.resize(Array(nFrame, bias.size(1)))
    if (output.nElement() != nElement) { output.zero() }

    val inputOffset = input.storageOffset() - 1
    val outputOffset = output.storageOffset() - 1
    val biasOffset = bias.storageOffset() - 1
    val kernelOffset = weight.storageOffset() - 1

    val kernelHeight = outputSize
    val kernelWidth = inputSize
    val outputChannels = outputSize

    if (firstPass) {
      ev.getType() match {
        case "Double" =>
          classPtr = MKL.LinearInitDouble(inputHeight,
                                          inputWidth,
                                          outputChannels,
                                          kernelHeight,
                                          kernelWidth,
                                          this.getName())
        case "Float" =>
          classPtr = MKL.LinearInitFloat(inputHeight,
                                         inputWidth,
                                         outputChannels,
                                         kernelHeight,
                                         kernelWidth,
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
      case "Double" =>
        MKL.LinearForwardDouble(input.storage().array().asInstanceOf[Array[Double]],
                                inputOffset,
                                output.storage().array().asInstanceOf[Array[Double]],
                                outputOffset,
                                weight.storage().array().asInstanceOf[Array[Double]],
                                kernelOffset,
                                bias.storage().array().asInstanceOf[Array[Double]],
                                biasOffset,
                                classPtr)
      case "Float" =>
        MKL.LinearForwardFloat(input.storage().array().asInstanceOf[Array[Float]],
                               inputOffset,
                               output.storage().array().asInstanceOf[Array[Float]],
                               outputOffset,
                               weight.storage().array().asInstanceOf[Array[Float]],
                               kernelOffset,
                               bias.storage().array().asInstanceOf[Array[Float]],
                               biasOffset,
                               classPtr)
      case _ =>
        throw new UnsupportedOperationException(s"Only Float supported")
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    require(input.dim() == 2, "only batch mode supported")
    val nElement = gradInput.nElement()
    gradInput.resizeAs(input)
    if (nElement != gradInput.nElement()) {
      gradInput.zero()
    }

    val inputWidth = input.size(input.dim())
    val inputHeight = input.size(input.dim() - 1)

    val inputOffset = input.storageOffset() - 1
    val kernelOffset = weight.storageOffset() - 1
    val biasOffset = bias.storageOffset() - 1
    val gradOutputOffset = gradOutput.storageOffset() - 1
    val gradInputOffset = gradInput.storageOffset() - 1
    val gradWeightOffset = gradWeight.storageOffset() - 1
    val gradBiasOffset = gradBias.storageOffset() - 1

    val kernelHeight = outputSize
    val kernelWidth = inputSize
    val outputChannels = outputSize

    if (initBackward) {
      updateMklGradInput()
      initBackward = false
    }

    if (needCompute) {
      ev.getType() match {
        case "Double" =>
          MKL.LinearBackwardDataDouble(input.storage().array().asInstanceOf[Array[Double]],
                                       inputOffset,
                                       gradOutput.storage().array().asInstanceOf[Array[Double]],
                                       gradOutputOffset,
                                       gradInput.storage().array().asInstanceOf[Array[Double]],
                                       gradInputOffset,
                                       weight.storage().array().asInstanceOf[Array[Double]],
                                       kernelOffset,
                                       bias.storage().array().asInstanceOf[Array[Double]],
                                       biasOffset,
                                       classPtr)
        case "Float" =>
          MKL.LinearBackwardDataFloat(input.storage().array().asInstanceOf[Array[Float]],
                                      inputOffset,
                                      gradOutput.storage().array().asInstanceOf[Array[Float]],
                                      gradOutputOffset,
                                      gradInput.storage().array().asInstanceOf[Array[Float]],
                                      gradInputOffset,
                                      weight.storage().array().asInstanceOf[Array[Float]],
                                      kernelOffset,
                                      bias.storage().array().asInstanceOf[Array[Float]],
                                      biasOffset,
                                      classPtr)
        case _ =>
          throw new UnsupportedOperationException(s"Only Float supported")
      }
    }

    ev.getType() match {
      case "Double" =>
        MKL.LinearBackwardKernelDouble(input.storage().array().asInstanceOf[Array[Double]],
                                       inputOffset,
                                       gradOutput.storage().array().asInstanceOf[Array[Double]],
                                       gradOutputOffset,
                                       gradWeight.storage().array().asInstanceOf[Array[Double]],
                                       gradWeightOffset,
                                       weight.storage().array().asInstanceOf[Array[Double]],
                                       kernelOffset,
                                       bias.storage().array().asInstanceOf[Array[Double]],
                                       biasOffset,
                                       classPtr)

      case "Float" =>
        MKL.LinearBackwardKernelFloat(input.storage().array().asInstanceOf[Array[Float]],
                                      inputOffset,
                                      gradOutput.storage().array().asInstanceOf[Array[Float]],
                                      gradOutputOffset,
                                      gradWeight.storage().array().asInstanceOf[Array[Float]],
                                      gradWeightOffset,
                                      weight.storage().array().asInstanceOf[Array[Float]],
                                      kernelOffset,
                                      bias.storage().array().asInstanceOf[Array[Float]],
                                      biasOffset,
                                      classPtr)

      case _ =>
        throw new UnsupportedOperationException(s"Only Float/Double supported")
    }

    ev.getType() match {
      case "Double" =>
        MKL.LinearBackwardBiasDouble(input.storage().array().asInstanceOf[Array[Double]],
                                     inputOffset,
                                     gradOutput.storage().array().asInstanceOf[Array[Double]],
                                     gradOutputOffset,
                                     gradBias.storage().array().asInstanceOf[Array[Double]],
                                     gradBiasOffset,
                                     weight.storage().array().asInstanceOf[Array[Double]],
                                     kernelOffset,
                                     bias.storage().array().asInstanceOf[Array[Double]],
                                     biasOffset,
                                     classPtr)

      case "Float" =>
        MKL.LinearBackwardBiasFloat(input.storage().array().asInstanceOf[Array[Float]],
                                    inputOffset,
                                    gradOutput.storage().array().asInstanceOf[Array[Float]],
                                    gradOutputOffset,
                                    gradBias.storage().array().asInstanceOf[Array[Float]],
                                    gradBiasOffset,
                                    weight.storage().array().asInstanceOf[Array[Float]],
                                    kernelOffset,
                                    bias.storage().array().asInstanceOf[Array[Float]],
                                    biasOffset,
                                    classPtr)

      case _ =>
        throw new UnsupportedOperationException(s"Only Float/Double supported")
    }

    gradInput
  }

//  override def accGradParameters(input: Tensor[T],
//                                 gradOutput: Tensor[T],
//                                 scale: Double = 1.0): Unit = {
//    require(input.dim() == 2, "only batch mode supported")
//    require(input.dim() == 1 || input.dim() == 2, "input must be vector or matrix")
//    val value = ev.fromType[Double](scale)
//    if (input.dim() == 1) {
//      gradWeight.addr(value, gradOutput, input)
//      gradBias.add(value, gradOutput)
//    } else if (input.dim() == 2) {
//      gradWeight.addmm(value, gradOutput.t, input)
//      gradBias.addmv(value, gradOutput.t, addBuffer)
//    }
//  }

  override def updateParameters(learningRate: T): Unit = {
    // weight.map(gradWeight,(a,b)=>a - learningRate*b)
    weight.add(ev.negative(learningRate), gradWeight)
    // bias.map(gradBias,(a,b)=>a - learningRate*b)
    bias.add(ev.negative(learningRate), gradBias)
  }

  override def zeroGradParameters(): Unit = {
    gradWeight.zero()
    gradBias.zero()
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(this.weight, this.bias), Array(this.gradWeight, this.gradBias))
  }

  override def equals(obj: Any): Boolean = {

    if (!super.equals(obj)) {
      return false
    }

    if (!obj.isInstanceOf[Linear[T]]) { return false }
    val other = obj.asInstanceOf[Linear[T]]
    if (this.eq(other)) { return true }

    gradWeight == other.gradWeight &&
    gradBias == other.gradBias &&
    weight == other.weight &&
    bias == other.bias
  }

  override def hashCode(): Int = {
    val seed = 37
    var hash = super.hashCode()
    hash = hash * seed + gradWeight.hashCode()
    hash = hash * seed + gradBias.hashCode()
    hash = hash * seed + weight.hashCode()
    hash = hash * seed + bias.hashCode()

    hash
  }

  override def toString(): String = {
    s"nn.mkl.Linear($inputSize -> $outputSize)"
  }

  override def findModel(paramOffset: Int, indexes: Array[Int]): (Module[T], Int, Array[Int]) = {
    (this, paramOffset - outputSize * inputSize - outputSize, indexes)
  }

}
