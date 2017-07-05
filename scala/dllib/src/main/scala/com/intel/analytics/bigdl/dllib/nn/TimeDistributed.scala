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
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

/**
 * This layer is intended to apply contained layer to each temporal time slice
 * of input tensor.
 *
 * For instance, The TimeDistributed Layer can feed each time slice of input tensor
 * to the Linear layer.
 *
 * The input data format is [Batch, Time, Other dims]. For the contained layer, it must not change
 * the Other dims length.
 *
 * @tparam T data type, which can be [[Double]] or [[Float]]
 */

class TimeDistributed[T : ClassTag] (layer: TensorModule[T])
  (implicit ev: TensorNumeric[T]) extends TensorModule[T] {

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

    val _inputSize = input.size
    combine(_inputSize, inputSize)
    input.resize(inputSize)
    val _output = layer.updateOutput(input).toTensor[T]
    split(_output.size, outputSize, _inputSize(0), _inputSize(1))
    input.resize(_inputSize)
    output.set(_output).resize(outputSize)
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    if (gradOutputSize == null) {
      gradOutputSize = new Array[Int](gradOutput.size.length - 1)
    }
    val _inputSize = input.size
    val _gradOutputSize = gradOutput.size
    combine(_gradOutputSize, gradOutputSize)
    input.resize(inputSize)
    gradOutput.resize(gradOutputSize)
    val _gradInput = layer.updateGradInput(input, gradOutput).toTensor[T]
    gradInput.set(_gradInput).resize(_inputSize)
    input.resize(_inputSize)
    gradOutput.resize(_gradOutputSize)
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    val _inputSize = input.size
    val _gradOutputSize = gradOutput.size
    input.resize(inputSize)
    gradOutput.resize(gradOutputSize)
    layer.accGradParameters(input, gradOutput)
    input.resize(_inputSize)
    gradOutput.resize(_gradOutputSize)
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
   * This method will return a table indicating the name and corresponding parameters.
   * @return Table
   */
  override def getParametersTable(): Table = layer.getParametersTable()

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
    inputSize = null
    gradOutputSize = null
    outputSize = null
    this
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[TimeDistributed[T]]

  override def equals(other: Any): Boolean = other match {
    case that: TimeDistributed[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        layer.equals(layer) &&
        inputSize == that.inputSize &&
        gradOutputSize == that.gradOutputSize &&
        outputSize == that.outputSize
    case _ => false
  }

  override def hashCode(): Int = {
    val state = Seq(super.hashCode(),
      layer, inputSize, gradOutputSize, outputSize)
    state.filter(_ != null).map(_.hashCode()).foldLeft(0)((a, b) => 31 * a + b)
  }
}

object TimeDistributed {
  def apply[@specialized(Float, Double) T: ClassTag](layer: TensorModule[T])
    (implicit ev: TensorNumeric[T]): TimeDistributed[T] = {
    new TimeDistributed[T](layer)
  }
}
