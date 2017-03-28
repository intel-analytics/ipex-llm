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

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * This layer is intended to apply contained layer to each temporal time slice
 * of input tensor.
 *
 * For instance, The TimeDistributed Layer can feed each time slice of input tensor
 * to the Linear layer.
 *
 * @tparam T data type, which can be [[Double]] or [[Float]]
 */

class TimeDistributed[T : ClassTag] (layer: TensorModule[T])
(implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  private val fInput: Tensor[T] = Tensor[T]()
  private val fGradOutput: Tensor[T] = Tensor[T]()
  private var inputSize: Array[Int] = _
  private var gradOutputSize: Array[Int] = _
  private var outputSize: Array[Int] = _

  private def combine(src: Array[Int], target: Array[Int]): Unit = {
    require(src.length == target.length + 1,
      "TimeDistributed: combine method requires src.length == target.length + 1" +
        s" Current src.length = ${src.length}" +
        s" Current target.length = ${target.length}")

    target(0) = src(0) * src(1)
    var j = 1
    while (j < target.length) {
      target(j) = src(j + 1)
      j += 1
    }
  }

  private def split(src: Array[Int], target: Array[Int], dim1: Int, dim2: Int): Unit = {
    require(src.length == target.length - 1,
      "TimeDistributed: split method requires src.length == target.length - 1" +
        s" Current src.length = ${src.length}" +
        s" Current target.length = ${target.length}")
    require(dim1 * dim2 == src(0),
    "TimeDistributed: split method requires dim1 * dim2 == src(0), " +
      s"Current dim1 = ${dim1}, dim2 = ${dim2}, src(0) = ${src(0)}")

    target(0) = dim1
    target(1) = dim2
    var j = 1
    while (j < src.length) {
      target(j + 1) = src(j)
      j += 1
    }
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim >= 3,
      "TimeDistributed: input should be at least a 3D Tensor, e.g [batch, time, inputDim]. " +
        s"Current input.dim = ${input.dim}")

    if (inputSize == null) {
      inputSize = new Array[Int](input.size.length - 1)
    }
    if (outputSize == null) {
      outputSize = new Array[Int](input.size.length)
    }

    /**
     * combine: [B, T, D] => [B * T, D]
     * split:   [B * T, D] => [B, T, D]
     */
    combine(input.size, inputSize)
    fInput.set(input).resize(inputSize)
    val _output = layer.updateOutput(fInput).toTensor[T]
    split(_output.size, outputSize, input.size(1), input.size(2))
    output.set(_output).resize(outputSize)
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradOutputSize = inputSize
    combine(gradOutput.size, gradOutputSize)
    fGradOutput.set(gradOutput).resize(gradOutputSize)
    val _gradInput = layer.updateGradInput(fInput, fGradOutput).toTensor[T]
    gradInput.set(_gradInput).resize(input.size)
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T],
                                 scale: Double = 1.0): Unit = {
    layer.accGradParameters(fInput, fGradOutput)
  }

  /**
   * If the module has parameters, this will zero the accumulation of the gradients with respect
   * to these parameters. Otherwise, it does nothing.
   */
  override def zeroGradParameters(): Unit = {
    layer.zeroGradParameters()
  }

  override def updateParameters(learningRate: T): Unit = layer.updateParameters(learningRate)

  override def reset(): Unit = layer.reset()

  override def training(): TimeDistributed.this.type = {
    layer.training()
    super.training()
  }

  /**
   * get execution engine type
   */
  override def checkEngineType(): TimeDistributed.this.type = {
    layer.checkEngineType()
    super.checkEngineType()
  }

  override def resetTimes(): Unit = layer.resetTimes()

  override def getTimes(): Array[(AbstractModule[_ <: Activity, _ <: Activity, T], Long, Long)] = {
    layer.getTimes()
  }

  override def evaluate(): TimeDistributed.this.type = {
    layer.evaluate()
    super.evaluate()
  }

  /**
   * This function returns two arrays. One for the weights and the other the gradients
   * Custom modules should override this function if they have parameters
   *
   * @return (Array of weights, Array of grad)
   */
  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = layer.parameters()

  /**
   * This method compact all parameters and gradients of the model into two tensors. So it's easier
   * to use optim method
   *
   * @return
   */
  override def getParameters(): (Tensor[T], Tensor[T]) = layer.getParameters()

  /**
   * Copy the useful running status from src to this.
   *
   * The subclass should override this method if it has some parameters besides weight and bias.
   * Such as runningMean and runningVar of BatchNormalization.
   *
   * @param src source Module
   * @return this
   */
  override def copyStatus(src: Module[T]): TimeDistributed.this.type = {
    layer.copyStatus(src)
    this
  }

  override def clearState(): TimeDistributed.this.type = {
    super.clearState()
    layer.clearState()
    fInput.set()
    fGradOutput.set()
    inputSize = null
    gradOutputSize = null
    outputSize = null
    this
  }

  override def toString(): String = {
    val str = "nn.TimeDistributed"
    str
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[TimeDistributed[T]]

  override def equals(other: Any): Boolean = other match {
    case that: TimeDistributed[T] =>
      super.equals(that) &&
        (that canEqual this) &&
      layer.equals(layer) &&
        fInput == that.fInput &&
        fGradOutput == that.fGradOutput &&
        inputSize == that.inputSize &&
        gradOutputSize == that.gradOutputSize &&
        outputSize == that.outputSize
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(super.hashCode(),
      layer, fInput, fGradOutput, inputSize, gradOutputSize, outputSize)
    state.map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object TimeDistributed {
  def apply[@specialized(Float, Double) T: ClassTag](layer: TensorModule[T])
  (implicit ev: TensorNumeric[T]): TimeDistributed[T] = {
    new TimeDistributed[T](layer)
  }
}
