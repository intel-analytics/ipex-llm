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
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

/**
 * Outputs the Euclidean distance of the input to outputSize centers
 * @param inputSize inputSize
 * @param outputSize outputSize
 * @tparam T Numeric type. Only support float/double now
 */

@SerialVersionUID(1438188993718795033L)
class Euclidean[T: ClassTag](val inputSize: Int, val outputSize: Int,
  val fastBackward: Boolean = true)(implicit ev: TensorNumeric[T]) extends TensorModule[T]{

  val weight = Tensor(inputSize, outputSize)
  val gradWeight = Tensor(inputSize, outputSize)

  // buffer
  var inputBuffer = Tensor[T]()
  var weightBuffer = Tensor[T]()
  var repeatBuffer = Tensor[T]()
  var divBuffer = Tensor[T]()
  var sumBuffer = Tensor[T]()

  reset()

  override def reset(): Unit = {
    val stdv = 1 / math.sqrt(weight.size(1))
    weight.apply1(_ => ev.fromType[Double](RNG.uniform(-stdv, stdv)))
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {

    require(input.dim() == 1 || input.dim() == 2,
      "Euclidean: " + ErrorInfo.constrainInputAsVectorOrBatch)

    if (input.dim() == 1) {
      if (input.isContiguous()) {
        inputBuffer = input.view(inputSize, 1)
      } else {
        inputBuffer = input.reshape(Array(inputSize, 1))
      }
      inputBuffer.expandAs(weight)
      repeatBuffer.resizeAs(inputBuffer).copy(inputBuffer)

      repeatBuffer.add(ev.fromType(-1), weight)
      repeatBuffer.norm(output, 2, 1)

      output.resize(outputSize)
    } else if (input.dim() == 2) {
      val batchSize = input.size(1)

      if (input.isContiguous()) {
        inputBuffer = input.view(batchSize, inputSize, 1)
      } else {
        inputBuffer = input.reshape(Array(batchSize, inputSize, 1))
      }
      inputBuffer.expand(Array(batchSize, inputSize, outputSize))
      repeatBuffer.resizeAs(inputBuffer).copy(inputBuffer)

      weightBuffer = weight.view(1, inputSize, outputSize)
      weightBuffer.expandAs(repeatBuffer)

      repeatBuffer.add(ev.fromType(-1), weightBuffer)

      repeatBuffer.norm(output, 2, 2)
      output.resize(batchSize, outputSize)
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    require(input.dim() == 1 || input.dim() == 2,
      "Euclidean: " + ErrorInfo.constrainInputAsVectorOrBatch)

    if (!fastBackward) {
      updateOutput(input)
    }
    // to prevent div by zero (NaN) bugs
    inputBuffer.resizeAs(output).copy(output).add(ev.fromType(0.0000001))
    divBuffer.resizeAs(gradOutput).cdiv(gradOutput, inputBuffer)
    if (input.dim() == 1) {
      divBuffer.resize(1, outputSize)
      divBuffer.expandAs(weight)

      repeatBuffer.cmul(divBuffer)
      gradInput.sum(repeatBuffer, 2)
      gradInput.resizeAs(input)
    } else if (input.dim() == 2) {
      val batchSize = input.size(1)

      divBuffer.resize(batchSize, 1, outputSize)
      divBuffer.expand(Array(batchSize, inputSize, outputSize))

      repeatBuffer.cmul(divBuffer)
      gradInput.sum(repeatBuffer, 3)
      gradInput.resizeAs(input)
    }
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T],
    scale: Double = 1.0): Unit = {

    require(input.dim() == 1 || input.dim() == 2,
      "Euclidean: " + ErrorInfo.constrainInputAsVectorOrBatch)
    if (input.dim() == 1) {
      gradWeight.add(ev.fromType(-scale), repeatBuffer)
    } else if (input.dim() == 2) {
      sumBuffer.sum(repeatBuffer, 1)
      sumBuffer.resizeAs(weight)
      gradWeight.add(ev.fromType(-scale), sumBuffer)
    }
  }

  override def toString(): String = {
    s"nn.Euclidean($inputSize, $outputSize)"
  }

  override def zeroGradParameters(): Unit = {
    gradWeight.zero()
  }

  override def clearState() : this.type = {
    super.clearState()
    inputBuffer.set()
    weightBuffer.set()
    repeatBuffer.set()
    divBuffer.set()
    sumBuffer.set()
    this
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(this.weight), Array(this.gradWeight))
  }

  override def getParametersTable(): Table = {
    T(getName() -> T("weight" -> weight, "gradWeight" -> gradWeight))
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[Euclidean[T]]

  override def equals(other: Any): Boolean = other match {
    case that: Euclidean[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        weight == that.weight &&
        gradWeight == that.gradWeight &&
        inputSize == that.inputSize &&
        outputSize == that.outputSize &&
        fastBackward == that.fastBackward
    case _ => false
  }

  override def hashCode(): Int = {
    def getHashCode(a: Any): Int = if (a == null) 0 else a.hashCode()
    val state = Seq(super.hashCode(), weight, gradWeight, inputSize, outputSize, fastBackward)
    state.map(getHashCode).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object Euclidean {
  def apply[@specialized(Float, Double) T: ClassTag](
    inputSize: Int,
    outputSize: Int,
    fastBackward: Boolean = true)(implicit ev: TensorNumeric[T]) : Euclidean[T] = {
    new Euclidean[T](inputSize, outputSize, fastBackward)
  }
}
