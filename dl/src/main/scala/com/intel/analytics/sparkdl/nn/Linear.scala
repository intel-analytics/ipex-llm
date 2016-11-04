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

package com.intel.analytics.sparkdl.nn

import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.utils.RandomGenerator
import scala.reflect.ClassTag
import RandomGenerator._

class Linear[@specialized(Float, Double) T: ClassTag](
  inputSize: Int,
  outputSize: Int,
  private var initMethod: InitializationMethod = Default
)(implicit ev: TensorNumeric[T]) extends TensorModule[T] {
  val weight: Tensor[T] = Tensor[T](outputSize, inputSize)
  val bias: Tensor[T] = Tensor[T](outputSize)
  val addBuffer: Tensor[T] = Tensor[T]()
  this.gradWeight = Tensor[T](outputSize, inputSize)
  this.gradBias = Tensor[T](outputSize)
  reset()

  def setInitMethod(initMethod: InitializationMethod): this.type = {
    this.initMethod = initMethod
    this
  }

  override def reset(): Unit = {
    initMethod match {
      case Default =>
        val stdv = 1.0 / math.sqrt(weight.size(2))
        weight.apply1(_ => ev.fromType[Double](RNG.uniform(0, 1) * 2 * stdv - stdv))
        bias.apply1(_ => ev.fromType[Double](RNG.uniform(0, 1) * 2 * stdv - stdv))
      case Xavier =>
        val fanIn = weight.size(2)
        val fanOut = weight.size(1)
        val stdv = math.sqrt(6.0 / (fanIn + fanOut))
        weight.apply1(_ => ev.fromType[Double](RNG.uniform(-stdv, stdv)))
        bias.fill(ev.fromType(0))
      case Constant =>
        weight.apply1(_ => ev.fromType[Double](0.1))
        bias.fill(ev.fromType(0))
    }
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim() == 1 || input.dim() == 2, "input must be vector or matrix")
    if (input.dim() == 1) {
      output.resize(Array(outputSize))
      output.copy(bias)
      output.addmv(ev.fromType[Int](1), weight, input)
    }
    else if (input.dim() == 2) {
      val nFrame = input.size(1)
      val nElement = output.nElement
      val t = Array(nFrame, bias.size(1))
      output.resize(t)
      if (output.nElement() != nElement) {
        output.zero()
      }

      if (addBuffer.nElement() != nFrame) {
        addBuffer.resize(Array(nFrame)).fill(ev.fromType[Int](1))
      }

      output.addmm(ev.fromType[Int](0), output, ev.fromType[Int](1), input, weight.t)
      output.addr(ev.fromType[Int](1), addBuffer, bias)
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    require(input.dim() == 1 || input.dim() == 2, "input must be vector or matrix")
    val nElement = gradInput.nElement()
    gradInput.resizeAs(input)
    if (nElement != gradInput.nElement()) {
      gradInput.zero()
    }

    if (input.dim() == 1) {
      gradInput.addmv(ev.fromType[Int](0), ev.fromType[Int](1), weight.t(), gradOutput)
    } else if (input.dim() == 2) {
      gradInput.addmm(ev.fromType[Int](0), ev.fromType[Int](1), gradOutput, weight)
    }
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T],
    scale: Double = 1.0): Unit = {
    require(input.dim() == 1 || input.dim() == 2, "input must be vector or matrix")
    val value = ev.fromType[Double](scale)
    if (input.dim() == 1) {
      gradWeight.addr(value, gradOutput, input)
      gradBias.add(value, gradOutput)
    }
    else if (input.dim() == 2) {
      gradWeight.addmm(value, gradOutput.t, input)
      gradBias.addmv(value, gradOutput.t, addBuffer)
    }
  }

  override def updateParameters(learningRate: T): Unit = {
    weight.add(ev.negative(learningRate), gradWeight)
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

    if (!obj.isInstanceOf[Linear[T]]) {
      return false
    }
    val other = obj.asInstanceOf[Linear[T]]
    if (this.eq(other)) {
      return true
    }

    gradWeight == other.gradWeight &&
      gradBias == other.gradBias &&
      weight == other.weight &&
      bias == other.bias
  }

  override def hashCode() : Int = {
    val seed = 37
    var hash = super.hashCode()
    hash = hash * seed + gradWeight.hashCode()
    hash = hash * seed + gradBias.hashCode()
    hash = hash * seed + weight.hashCode()
    hash = hash * seed + bias.hashCode()

    hash
  }

  override def toString(): String = {
    s"nn.Linear($inputSize -> $outputSize)"
  }

  override def findModel(paramOffset: Int,
    indexes: Array[Int]): (Module[Tensor[T], Tensor[T], T], Int, Array[Int]) = {
    (this, paramOffset - outputSize * inputSize - outputSize, indexes)
  }
}
