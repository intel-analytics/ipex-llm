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

import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.utils.RandomGenerator._

import scala.reflect.ClassTag

/**
  * the cosine similarity of the input to k mean centers
  * @param inputSize
  * @param outputSize
  * @param ev$1
  * @param ev
  * @tparam T
  */
class Cosine[T: ClassTag](inputSize : Int, outputSize : Int)(
  implicit ev: TensorNumeric[T]) extends TensorModule[T]{

  this.gradWeight = Tensor[T](outputSize, inputSize)
  val weight = Tensor[T](outputSize, inputSize)

  @transient
  var _weightNorm: Tensor[T] = null
  @transient
  var _inputNorm: Tensor[T] = null
  @transient
  var __norm: T = ev.fromType(0)
  @transient
  var _sum: Tensor[T] = null
  @transient
  var _weight: Tensor[T] = null
  @transient
  var _gradOutput: Tensor[T] = null

  reset()

  override def reset(): Unit = {
    val stdv = 1 / math.sqrt(weight.size(1))
    weight.apply1(_ => ev.fromType[Double](RNG.uniform(-stdv, stdv)))
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim() == 1 || input.dim() == 2, "input must be vector or matrix")

    if (null == _weightNorm) _weightNorm = Tensor[T]()
    if (null == _inputNorm) _inputNorm = Tensor[T]()
    if (null == _sum) _sum = Tensor[T]()
    if (null == _weight) _weight = Tensor[T]()
    if (null == _gradOutput) _gradOutput = Tensor[T]()

    weight.norm(_weightNorm, 2, 2)
    _weightNorm.add(ev.fromType(1e-12))

    if (input.dim() == 1) {
      output.resize(outputSize).zero()
      output.addmv(ev.fromType(1), weight, input)

      __norm = ev.plus(input.norm(), ev.fromType(1e-12))
      output.cdiv(_weightNorm.view(outputSize)).div(__norm)
    } else if (input.dim() == 2) {
      val batchSize = input.size(1)
      val nElement = output.nElement()
      output.resize(batchSize, outputSize)
      if (output.nElement() != nElement) output.zero()
      output.addmm(ev.fromType(0), output, ev.fromType(1), input, weight.t())

      input.norm(_inputNorm, 2, 2)
      output.cdiv(_weightNorm.view(1, outputSize).expandAs(output))
      output.cdiv(Tensor[T](_inputNorm.storage(), _inputNorm.storageOffset(),
        _inputNorm.size(), _inputNorm.stride()).expandAs(output))
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]) : Tensor[T] = {
    require(input.dim() == 1 || input.dim() == 2, "input must be vector or matrix")
    val nElement = gradInput.nElement()
    gradInput.resizeAs(input)
    if (gradInput.nElement() != nElement) gradInput.zero()

    if (input.dim() == 1) {
      _weight.resizeAs(weight).copy(weight)
      _weight.cdiv(Tensor[T](_weightNorm.storage(), _weightNorm.storageOffset(),
        _weightNorm.size(), _weightNorm.stride()).expandAs(weight))
      _weight.div(__norm)
      _weight.addr(ev.fromType(1), _weight, ev.divide(ev.fromType(-1),
        ev.times(__norm, __norm)), output, input)
      gradInput.addmv(ev.fromType(0), ev.fromType(1), _weight.t(), gradOutput)
    } else if (input.dim() == 2) {
      val inputNorm = _inputNorm.expandAs(input)
      val weightNorm = _weightNorm.view(1, outputSize).expandAs(gradOutput)

      gradInput.copy(input).cdiv(inputNorm)
      _gradOutput.resizeAs(gradOutput).copy(gradOutput)
      _gradOutput.cmul(output)

      _sum.sum(_gradOutput, 2)
      gradInput.cmul(_sum.expandAs(input))

      _gradOutput.resizeAs(gradOutput).copy(gradOutput)
      _gradOutput.cdiv(weightNorm)

      gradInput.addmm(ev.fromType(-1), gradInput, ev.fromType(1), _gradOutput, weight)

      gradInput.cdiv(inputNorm)

    }

    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T],
   scale: Double = 1.0): Unit = {
    require(input.dim() == 1 || input.dim() == 2, "input must be vector or matrix")

    if (input.dim() == 1) {
      _gradOutput.resizeAs(gradOutput).copy(gradOutput)

      var weightNorm = Tensor[T]()
      weightNorm = _weightNorm.view(outputSize)
      _gradOutput.cdiv(weightNorm)
      gradWeight.addr(ev.divide(ev.fromType(scale), __norm), _gradOutput, input)

      _gradOutput.cdiv(weightNorm)
      _gradOutput.cmul(output)

      _weight.resizeAs(weight).copy(weight)
      _weight.cmul(_gradOutput.view(outputSize, 1).expandAs(weight))
      gradWeight.add(ev.fromType(-1), _weight)
    } else if (input.dim() == 2) {
      _weight.resizeAs(weight).copy(weight)
      _gradOutput.resizeAs(gradOutput).copy(gradOutput)

      _gradOutput.cmul(output)
      _sum.sum(_gradOutput, 1)

      val grad = _sum(1)
      grad.cdiv(_weightNorm.select(2, 1))
      _weight.cmul(grad.view(outputSize, 1).expandAs(_weight))

      val input_ = _gradOutput
      input_.resizeAs(input).copy(input)
      input_.cdiv(_inputNorm.expandAs(input))
      _weight.addmm(ev.fromType(-1), _weight, ev.fromType(1), gradOutput.t(), input_)
      _weight.cdiv(_weightNorm.expandAs(_weight))
      gradWeight.add(_weight)
    }
  }

  override def zeroGradParameters(): Unit = {
    gradWeight.zero()
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(this.weight), Array(this.gradWeight))
  }

  override def toString(): String = {
    s"nn.Cosine($inputSize, $outputSize)"
  }
}
