/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
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

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Engine, T}

import scala.concurrent.Future
import scala.reflect.ClassTag

class Recurrent[T : ClassTag] (
  hiddenSize: Int = 3,
  outputSize: Int = 4,
  bpttTruncate: Int = 2)
(implicit ev: TensorNumeric[T]) extends Container[Tensor[T], Tensor[T], T] {

  val hidden = Tensor[T]()
  @transient
  protected var results: Array[Future[Unit]] = null

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    input.squeeze()
    require(input.dim == 2 || input.dim == 3, "only support 2D or 3D (batch) input")
    require(modules.length == 3,
      "rnn container must include a cell, a non-linear layer and a linear layer")

    val module = modules(0)
    val transform = modules(1)
    val linear = modules(2)

    if (input.dim == 2) {
      val height = input.size(1)
      output.resize(Array(height, outputSize))
      hidden.resize(Array(height + 1, hiddenSize))

      var i = 1
      while (i <= height) {
        val curInput = T(input(i), hidden(i))
        val currentOutput = module.updateOutput(curInput)
        transform.updateOutput(currentOutput)
        output.update(i, linear.updateOutput(
          transform.output).asInstanceOf[Tensor[T]])
        hidden(i + 1) = Tensor[T]()
          .resizeAs(transform.output.asInstanceOf[Tensor[T]])
          .copy(transform.output.asInstanceOf[Tensor[T]])
        i += 1
      }
    } else {
      val batchSize = input.size(1)
      val height = input.size(2)
      output.resize(Array(batchSize, height, outputSize))
      hidden.resize(Array(batchSize, height + 1, hiddenSize))

      if (results == null || results.length != batchSize) {
        results = new Array[Future[Unit]](batchSize)
      }

      var i = 0
      while (i < batchSize) {
        val _i = i + 1
        results(i) = Engine.model.invoke(() => {
          val inputT = input.select(1, _i)
          val outputT = output.select(1, _i)
          val hiddenT = hidden.select(1, _i)
          var j = 1
          while (j <= height) {
            val curInput = T(inputT(j), hiddenT(j))
            val currentOutput = module.updateOutput(curInput)
            transform.updateOutput(currentOutput)
            outputT.update(j, linear.updateOutput(
              transform.output).asInstanceOf[Tensor[T]])
            hiddenT(j + 1) = Tensor[T]()
              .resizeAs(transform.output.asInstanceOf[Tensor[T]])
              .copy(transform.output.asInstanceOf[Tensor[T]])
            j += 1
          }
        })
        i += 1
      }
      Engine.model.sync(results)
    }
    output
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T],
                                 scale: Double = 1.0): Unit = {
    input.squeeze()
    require(input.dim == 2 || input.dim == 3, "only support 2D or 3D (batch) input")
    require(modules.length == 3,
      "rnn container must include a cell, a non-linear layer and a linear layer")

    val module = modules(0)
    val transform = modules(1)
    val linear = modules(2)

    if (input.dim == 2) {
      val height = input.size(1)
      var i = height
      while (i >= 1) {
        transform.output.asInstanceOf[Tensor[T]]
          .resizeAs(hidden(i + 1))
          .copy(hidden(i + 1))
        val deltaGradOutput = linear.updateGradInput(
          transform.output, gradOutput(i))
        linear.accGradParameters(
          transform.output, gradOutput(i)
        )
        var deltaHidden =
          transform.updateGradInput(hidden(i), deltaGradOutput)
        var bpttStep = i
        while (bpttStep >= Math.max(1, i - bpttTruncate)) {
          val curInput = T(input(bpttStep), hidden(bpttStep))
          module.accGradParameters(curInput, deltaHidden)
          transform.output.asInstanceOf[Tensor[T]]
            .copy(hidden(bpttStep))
          deltaHidden = transform.updateGradInput(Tensor(),
            module.updateGradInput(curInput, deltaHidden).toTable(2))
          bpttStep -= 1
        }
        i -= 1
      }
    } else {
      val batchSize = input.size(1)
      val height = input.size(2)

      if (results == null || results.length != batchSize) {
        results = new Array[Future[Unit]](batchSize)
      }

      var i = 0
      while (i < batchSize) {
        val _i = i + 1
        results(i) = Engine.model.invoke(() => {
          val inputT = input.select(1, _i)
          val gradOutputT = gradOutput.select(1, _i)
          val hiddenT = hidden.select(1, _i)
          var j = height
          while (j >= 1) {
            transform.output.asInstanceOf[Tensor[T]]
              .resizeAs(hiddenT(j + 1))
              .copy(hiddenT(j + 1))
            val deltaGradOutput = linear.updateGradInput(
              transform.output, gradOutputT(j))
            linear.accGradParameters(
              transform.output, gradOutputT(j))
            var deltaHidden =
              transform.updateGradInput(hiddenT(j), deltaGradOutput)
            var bpttStep = j
            while (bpttStep >= Math.max(1, j - bpttTruncate)) {
              val curInput = T(inputT(bpttStep), hiddenT(bpttStep))
              module.accGradParameters(curInput, deltaHidden)
              transform.output.asInstanceOf[Tensor[T]]
                .copy(hiddenT(bpttStep))
              deltaHidden = transform.updateGradInput(Tensor(),
                module.updateGradInput(curInput, deltaHidden).toTable(2))
              bpttStep -= 1
            }
            j -= 1
          }
        })
        i += 1
      }
      Engine.model.sync(results)
    }
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    input.squeeze()
    require(input.dim == 2 || input.dim == 3, "only support 2D or 3D (batch) input")
    require(modules.length == 3,
      "rnn container must include a cell, a non-linear layer and a linear layer")

    val module = modules(0)
    val transform = modules(1)
    val linear = modules(2)

    gradInput.resize(input.size).zero

    if (input.dim == 2) {
      val height = input.size(1)
      var i = height
      while (i >= 1) {
        transform.output.asInstanceOf[Tensor[T]]
          .copy(hidden(i + 1))
        val deltaGradOutput = linear.updateGradInput(
          transform.output, gradOutput(i))
        var deltaHidden = transform.updateGradInput(hidden(i), deltaGradOutput)
        var bpttStep = i
        while (bpttStep >= Math.max(1, i - bpttTruncate)) {
          val curInput = T(input(bpttStep), hidden(bpttStep))
          val gradInputBundle = module.updateGradInput(curInput, deltaHidden).toTable
          gradInput(bpttStep).add(gradInputBundle(1).asInstanceOf[Tensor[T]])
          transform.output.asInstanceOf[Tensor[T]]
            .copy(hidden(bpttStep))
          deltaHidden = transform.updateGradInput(Tensor(), gradInputBundle(2))
          bpttStep -= 1
        }
        i -= 1
      }
    } else {
      val batchSize = input.size(1)
      val height = input.size(2)

      if (results == null || results.length != batchSize) {
        results = new Array[Future[Unit]](batchSize)
      }

      var i = 0
      while (i < batchSize) {
        val _i = i + 1
        results(i) = Engine.model.invoke(() => {
          val inputT = input.select(1, _i)
          val gradOutputT = gradOutput.select(1, _i)
          val gradInputT = gradInput.select(1, _i)
          val hiddenT = hidden.select(1, _i)
          var j = height
          while (j >= 1) {
            transform.output.asInstanceOf[Tensor[T]]
              .copy(hiddenT(j + 1))
            val deltaGradOutput = linear.updateGradInput(
              transform.output, gradOutputT(j))
            var deltaHidden = transform.updateGradInput(hiddenT(j), deltaGradOutput)
            var bpttStep = j
            while (bpttStep >= Math.max(1, j - bpttTruncate)) {
              val curInput = T(inputT(bpttStep), hiddenT(bpttStep))
              val gradInputBundle = module.updateGradInput(curInput, deltaHidden).toTable
              gradInputT(bpttStep).add(gradInputBundle(1).asInstanceOf[Tensor[T]])
              transform.output.asInstanceOf[Tensor[T]]
                .copy(hiddenT(bpttStep))
              deltaHidden = transform.updateGradInput(Tensor(), gradInputBundle(2))
              bpttStep -= 1
            }
            j -= 1
          }
        })
        i += 1
      }
      Engine.model.sync(results)
    }

    gradInput
  }

  override def toString(): String = {
    var str = "nn.Recurrent"
    str
  }
}

object Recurrent {
  def apply[@specialized(Float, Double) T: ClassTag](
    hiddenSize: Int = 3,
    outputSize: Int = 4,
    bpttTruncate: Int = 2)(implicit ev: TensorNumeric[T]) : Recurrent[T] = {
    new Recurrent[T](hiddenSize, outputSize, bpttTruncate)
  }
}
