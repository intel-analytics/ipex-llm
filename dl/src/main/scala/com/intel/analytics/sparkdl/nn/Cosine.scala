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

class Cosine[@specialized(Float, Double) T: ClassTag](inputSize : Int, outputSize : Int)(
  implicit ev: TensorNumeric[T]) extends TensorModule[T]{

  this.gradWeight = Tensor[T](outputSize, inputSize)
  var weight = Tensor[T](outputSize, inputSize)

  @transient
  var _weightNorm: Tensor[T] = null
  @transient
  var _inputNorm: Tensor[T] = null
  @transient
  var __norm: T = ev.fromType(0)
  @transient
  var _sum: Tensor[T] = null

  reset()

  override def reset(): Unit = {
    val stdv = 1 / math.sqrt(weight.size(1))
    weight.apply1(_ => ev.fromType[Double](RNG.uniform(-stdv, stdv)))
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    if (null == _weightNorm) _weightNorm = Tensor[T]()
    if (null == _inputNorm) _inputNorm = Tensor[T]()

    _weightNorm = weight.norm(ev.fromType(2), 2).add(ev.fromType(1e-12))

    if (input.dim() == 1) {
      output.resize(outputSize).zero()
      output.addmv(ev.fromType(1), weight, input)

      __norm = ev.plus(ev.sqrt(input.cmul(input).sum()), ev.fromType(1e-12))
      output.cdiv(_weightNorm.view(outputSize)).div(__norm)
    } else if (input.dim() == 2) {
      val batchSize = input.size(1)
      val nElement = output.nElement()
      output.resize(batchSize, outputSize)
      if (output.nElement() != nElement) output.zero()
      val tmp = weight.t()
      output.addmm(ev.fromType(0), output, ev.fromType(1), input, tmp)

      _inputNorm = input.norm(ev.fromType(2), 2)
      output.cdiv(_weightNorm.view(1, outputSize).expandAs(output))
      output.cdiv(Tensor[T](_inputNorm.storage(), _inputNorm.storageOffset(), _inputNorm.size(), _inputNorm.stride()).expandAs(output))
    } else {
      sys.error("input must be vector or matrix")
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]) : Tensor[T] = {
    val nElement = gradInput.nElement()
    gradInput.resizeAs(input)
    if (gradInput.nElement() != nElement) gradInput.zero()

    if (input.dim() == 1) {
      val _weight = Tensor[T].resizeAs(weight).copy(weight)
      _weight.cdiv(Tensor[T](_weightNorm.storage(), _weightNorm.storageOffset(), _weightNorm.size(), _weightNorm.stride()).expandAs(weight))
      _weight.div(__norm)
      _weight.addr(ev.fromType(1), _weight, ev.divide(ev.fromType(-1), ev.times(__norm, __norm)), output, input)
      gradInput.addmv(ev.fromType(0), ev.fromType(1), _weight.t(), gradOutput)
    } else if (input.dim() == 2) {
      val inputNorm = _inputNorm.expandAs(input)
      val weightNorm = _weightNorm.view(1, outputSize).expandAs(gradOutput)

      gradInput.copy(input).cdiv(inputNorm)
      val _gradOutput = Tensor[T]()
      _gradOutput.resizeAs(gradOutput).copy(gradOutput)
      _gradOutput.cmul(output)

      if (null == _sum) _sum = Tensor[T]()
      _sum.sum(_gradOutput, 2)
      gradInput.cmul(_sum.expandAs(input))

      _gradOutput.resizeAs(gradOutput).copy(gradOutput)
      _gradOutput.cdiv(weightNorm)

      gradInput.addmm(ev.fromType(-1), gradInput, ev.fromType(1), _gradOutput, weight)

      gradInput.cdiv(inputNorm)

    }

    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T], scale: Double = 1.0): Unit = {
    if (input.dim() == 1) {
      val _gradOutput = Tensor[T]()
      _gradOutput.resizeAs(gradOutput).copy(gradOutput)

      var weightNorm = Tensor[T]()
      weightNorm = _weightNorm.view(outputSize)
      _gradOutput.cdiv(weightNorm)
      gradWeight.addr(ev.divide(ev.fromType(scale), __norm), _gradOutput, input)

      _gradOutput.cdiv(weightNorm)
      _gradOutput.cmul(output)

      val _weight = Tensor[T].resizeAs(weight).copy(weight)
      _weight.cmul(_gradOutput.view(outputSize, 1).expandAs(weight))
    } else if (input.dim() == 2) {
      val _weight = Tensor[T].resizeAs(weight).copy(weight)

      val _gradOutput = Tensor[T].resizeAs(gradOutput).copy(gradOutput)
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
    } else {
      sys.error("input must be vector or matrix")
    }
  }


}
