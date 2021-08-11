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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.abstractnn.{Activity, TensorModule}
import com.intel.analytics.bigdl.nn.{Module => _}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * Applies layer normalization.
 * @param hiddenSize
 * @param ev$1
 * @param ev
 * @tparam T
 */
class LayerNormalization[T: ClassTag](hiddenSize: Int)
  (implicit ev: TensorNumeric[T]) extends BaseModule[T] {
  override def buildModel(): Module[T] = {
    val input = Input()
    val mean = Mean(-1, squeeze = false).inputs(input)
    val sub = CSubTableExpand().inputs(input, mean)
    val square = Square().inputs(sub)
    val mean2 = Mean(-1, squeeze = false).inputs(square)
    val add = AddConstant(1e-6).inputs(mean2)
    val sqrt = Power(-0.5, 1, 0).inputs(add)
    val mul = CMulTableExpand().inputs(sub, sqrt)
    val linear = new VectorProduct[T](hiddenSize).inputs(mul)
    Graph(input, linear)
  }
  override def updateOutput(input: Activity): Activity = {
    output = model.updateOutput(input)
    output
  }
}

/**
 * Implement x * weight vector + bias vector
 * @param hiddenSize
 * @param ev$1
 * @param ev
 * @tparam T The numeric type in this module parameters
 */
private[nn] class VectorProduct[T: ClassTag](val hiddenSize: Int)
   (implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  var weight = Tensor[T](hiddenSize).fill(ev.one)
  var bias = Tensor[T](hiddenSize).fill(ev.zero)
  var gradWeight = Tensor[T](hiddenSize)
  var gradBias = Tensor[T](hiddenSize)

  private val buffer = Tensor[T]()
  private var inputSize: Array[Int] = _
  private var gradOutputSize: Array[Int] = _
  private var outputSize: Array[Int] = _

  private def combine(src: Array[Int], target: Array[Int]): Array[Int] = {
    if (src.length <= 2) return src
    val targetArr = if (target == null) new Array[Int](src.length - 1) else target
    require(src.length == targetArr.length + 1,
      "combine method requires src.length == target.length + 1" +
        s" Current src.length = ${src.length}" +
        s" Current target.length = ${targetArr.length}")
    targetArr(0) = src(0) * src(1)
    var j = 1
    while (j < targetArr.length) {
      targetArr(j) = src(j + 1)
      j += 1
    }
    targetArr
  }

  private def split(src: Array[Int], target: Array[Int], srcInput: Array[Int]): Array[Int] = {
    if (src.length == srcInput.length) return src
    val dim1 = srcInput(0)
    val dim2 = srcInput(1)
    val targetArr = if (target == null) new Array[Int](srcInput.length) else target
    require(src.length == targetArr.length - 1,
      "split method requires src.length == target.length - 1" +
        s" Current src.length = ${src.length}" +
        s" Current target.length = ${targetArr.length}")
    require(dim1 * dim2 == src(0),
      "split method requires dim1 * dim2 == src(0), " +
        s"Current dim1 = ${dim1}, dim2 = ${dim2}, src(0) = ${src(0)}")

    targetArr(0) = dim1
    targetArr(1) = dim2
    var j = 1
    while (j < src.length) {
      targetArr(j + 1) = src(j)
      j += 1
    }
    targetArr
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val _inputSize = input.size
    inputSize = combine(_inputSize, inputSize)

    input.resize(inputSize)
    output.resizeAs(input).copy(input)
    val size = output.size(1)
    var i = 1
    while(i <= size) {
      output.select(1, i).cmul(weight).add(bias)
      i += 1
    }

    outputSize = split(output.size, outputSize, _inputSize)
    input.resize(_inputSize)
    output.resize(outputSize)

    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val _inputSize = input.size
    val _gradOutputSize = gradOutput.size
    gradOutputSize = combine(_gradOutputSize, gradOutputSize)
    input.resize(inputSize)
    gradOutput.resize(gradOutputSize)
    gradInput.resizeAs(input).zero()

    val size = gradInput.size(1)
    var i = 1
    while(i <= size) {
      gradInput.select(1, i).addcmul(gradOutput.select(1, i), weight)
      i += 1
    }

    gradInput.resize(_inputSize)
    input.resize(_inputSize)
    gradOutput.resize(_gradOutputSize)

    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    val _inputSize = input.size
    val _gradOutputSize = gradOutput.size
    input.resize(inputSize)
    gradOutput.resize(gradOutputSize)

    buffer.resizeAs(input).zero()
    buffer.addcmul(input, gradOutput)
    gradWeight = buffer.sum(1).squeeze()
    gradBias = gradOutput.sum(1).squeeze()

    input.resize(_inputSize)
    gradOutput.resize(_gradOutputSize)
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(this.weight, this.bias), Array(this.gradWeight, this.gradBias))
  }
}
