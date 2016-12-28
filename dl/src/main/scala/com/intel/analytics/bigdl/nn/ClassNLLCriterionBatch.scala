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

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.nn.abstractnn.TensorCriterion
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.Tensor

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}
import scala.reflect.ClassTag
import com.intel.analytics.bigdl.utils.Engine

class ClassNLLCriterionBatch[T: ClassTag](weights: Tensor[T] = null, sizeAverage: Boolean = true)
  (implicit ev: TensorNumeric[T]) extends ClassNLLCriterion[T] {
  // private var total_weight = ev.fromType[Int](0)
  if (weights != null) require(weights.dim() == 1, "weights input should be 1-D Tensor")

  @transient
  private var results: Array[Future[(T, T)]] = null
  @transient
  private var resultsBackward: Array[Future[_]] = null

  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    require(input.dim() == 1 || input.dim() == 2 || input.dim() == 3,
      "input tensor should be 1D 2D or 3D")
    val nClasses = input.size(input.dim())
    if (input.dim() != 3) {
      super.updateOutput(input, target)
    } else {
      val batchSize = input.size(1)
      val nframe = input.size(2)
      total_weight = ev.fromType[Int](0)
      output = ev.fromType[Int](0)

      if (results == null || results.length != batchSize) {
        results = new Array[Future[(T, T)]](batchSize)
      }

      var i = 1
      while (i <= batchSize) {
        val _i = i
        results(_i - 1) = Engine.model.invoke(() => {
          val targetT = target.select(1, _i)
          val inputT = input.select(1, _i)
          var j = 1
          var _total_weight = ev.fromType[Int](0)
          var _output: T = ev.fromType[Int](0)
          while (j < nframe) {
            val curTarget = ev.toType[Int](targetT.valueAt(j))
            assert(curTarget >= 1 && curTarget <= nClasses,
              s"curTarget ${curTarget} is out of range 1 to ${nClasses}")
            val curWeight = if (weights != null) weights.valueAt(curTarget) else ev.fromType[Int](1)
            val (o, w) = (ev.times(inputT.valueAt(j, curTarget), curWeight), curWeight)
            _output = ev.minus(_output, o)
            _total_weight = ev.plus(_total_weight, w)
            j += 1
          }
          _output = ev.divide(_output, _total_weight)
          (_output, ev.fromType[Int](1))
        })
        i += 1
      }

      i = 0
      while (i < batchSize) {
        val (o, w) = Await.result(results(i), Duration.Inf)
        output = ev.plus(output, o)
        total_weight = ev.plus(total_weight, w)
        i += 1
      }
      if (sizeAverage && total_weight != 0) {
        output = ev.divide(output, total_weight)
      }
      output
    }
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    require(input.dim() == 1 || input.dim() == 2 || input.dim() == 3,
      "input tensor should be 1D 2D or 3D")
    assert(ev.toType[Double](total_weight) > 0.0, "total weight must larger than 0")
    gradInput.resizeAs(input)
    gradInput.zero()

    if (input.dim() != 3) {
      super.updateGradInput(input, target)
    } else {

      val batchSize = input.size(1)
      val nframe = input.size(2)

      if (resultsBackward == null || resultsBackward.length != batchSize) {
        resultsBackward = new Array[Future[_]](batchSize)
      }

      var i = 1
      while (i <= batchSize) {
        val _i = i

        resultsBackward(_i - 1) = Engine.model.invoke(() => {
          val targetT = target.select(1, _i)
          val gradInputT = gradInput.select(1, _i)

          var j = 1
          while (j <= nframe) {
            val curTarget = ev.toType[Int](targetT.valueAt(j))
            gradInputT.setValue(j, curTarget, if (weights != null) ev.times(ev.fromType[Int](-1),
              weights.valueAt(curTarget))
            else ev.fromType[Int](-1))
            if (sizeAverage) gradInputT.setValue(j, curTarget, ev.divide(gradInputT.valueAt(j,
              curTarget), total_weight))
            j += 1
          }
        })
        i += 1
      }

      i = 0
      while (i < batchSize) {
        Await.result(resultsBackward(i), Duration.Inf)
        i += 1
      }

      gradInput
    }
  }
}
