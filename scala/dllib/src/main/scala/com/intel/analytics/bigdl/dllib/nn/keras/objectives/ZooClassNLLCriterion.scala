/*
 * Copyright 2018 Analytics Zoo Authors.
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

import com.intel.analytics.bigdl.nn.abstractnn.{SizeAverageStatus, TensorCriterion}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Engine

import scala.concurrent.{Await, Future}
import scala.concurrent.duration.Duration
import scala.reflect.ClassTag

class ZooClassNLLCriterion[@specialized(Float, Double) T: ClassTag]
(weights: Tensor[T] = null, sizeAverage: Boolean = true, logProbAsInput: Boolean = true,
 paddingValue: Int = -1)(implicit ev: TensorNumeric[T])
  extends TensorCriterion[T] {
  private var total_weight = ev.fromType[Int](0)
  if (weights != null) require(weights.dim() == 1,
    "weights input should be 1-D Tensor" +
      s"weights dim(${weights.dim()})")

  @transient
  private var results: Array[Future[(T, T)]] = null
  @transient
  private var resultsBackward: Array[Future[_]] = null

  private val epsilon: T = ev.fromType(1e-8)

  private val oneMinusEpsilon: T = ev.minus(ev.one, epsilon)

  sizeAverageStatus = if (sizeAverage) SizeAverageStatus.True else SizeAverageStatus.False

  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    require(input.dim() == 1 || input.dim() == 2,
      "ClassNLLCriterion: " +
        ErrorInfo.constrainInputAsVectorOrBatch +
        s"input dim(${input.dim()})")
    val nClasses = input.size(input.dim())
    if (input.dim() == 1) {
      require(input.dim() == target.dim(),
        "ClassNLLCriterion: " + ErrorInfo.constrainInputDimSameAsTarget +
          s" Input dimension is: ${ input.dim() } , target dimension is: ${ target.dim() }")
      val curTarget = ev.toType[Int](target.valueAt(1))
      assert(curTarget >= 1 && curTarget <= nClasses || curTarget == paddingValue,
        s"curTarget ${curTarget} is out of range, should be 1 to ${nClasses}")
      total_weight = if (weights != null) weights(Array(curTarget)) else ev.fromType[Int](1)
      output = if (curTarget == paddingValue) ev.zero
      else {
        if (!logProbAsInput) {
          val clipped = ev.clip(input.valueAt(curTarget), epsilon, oneMinusEpsilon)
          ev.times(ev.negative(ev.log(clipped)), total_weight)
        } else {
          ev.times(ev.negative(input.valueAt(curTarget)), total_weight)
        }
      }
    } else if (input.dim() == 2) {
      val batchSize = input.size(1)
      val targetSize = target.size()
      target.squeeze()
      require(target.dim() == 1,
        "ClassNLLCriterion: illegal target! Target should be 1D tensor after squeeze," +
          s"but target's size is: ${ target.size() }, please check your data.")

      total_weight = ev.fromType[Int](0)
      output = ev.fromType[Int](0)

      if (results == null || results.length != batchSize) {
        results = new Array[Future[(T, T)]](batchSize)
      }

      var i = 1
      while (i <= batchSize) {
        val _i = i
        results(_i - 1) = Engine.model.invoke( () => {
          val curTarget = ev.toType[Int](target.valueAt(_i))
          assert(curTarget >= 1 && curTarget <= nClasses || curTarget == paddingValue,
            s"curTarget ${curTarget} is out of range 1 to ${nClasses}")
          if (curTarget == paddingValue) (ev.zero, ev.zero)
          else {
            val curWeight = if (weights != null) weights.valueAt(curTarget) else ev.fromType[Int](1)
            if (!logProbAsInput) {
              val clipped = ev.clip(input.valueAt(_i, curTarget), epsilon, oneMinusEpsilon)
              (ev.times(ev.log(clipped), curWeight), curWeight)
            } else {
              (ev.times(input.valueAt(_i, curTarget), curWeight), curWeight)
            }

          }
        })
        i += 1
      }

      i = 0
      while (i < batchSize) {
        val (o, w) = Await.result(results(i), Duration.Inf)
        output = ev.minus(output, o)
        total_weight = ev.plus(total_weight, w)
        i += 1
      }
      target.resize(targetSize)
    }
    if (sizeAverage && total_weight != 0) {
      output = ev.divide(output, total_weight)
    }
    output
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    require(input.dim() == 1 || input.dim() == 2,
      "ClassNLLCriterion: " +
        ErrorInfo.constrainInputAsVectorOrBatch +
        s"input dim ${input.dim()}")
    gradInput.resizeAs(input)
    gradInput.zero()

    if (input.dim() == 1) {
      require(input.dim() == target.dim(),
        "InternalClassNLLCriterion: " + ErrorInfo.constrainInputDimSameAsTarget +
          s" Input dimension is: ${ input.dim() } , target dimension is: ${ target.dim() }")
      val curTarget = ev.toType[Int](target.valueAt(1))
      if (curTarget == paddingValue) return gradInput
      gradInput.setValue(curTarget, if (weights != null) ev.times(ev.fromType[Int](-1),
        weights.valueAt(curTarget))
      else ev.fromType[Int](-1))
      if (sizeAverage) gradInput.setValue(curTarget, ev.divide(gradInput.valueAt(curTarget),
        total_weight))
      if (!logProbAsInput) {
        val clipped = ev.clip(input.valueAt(curTarget), epsilon, oneMinusEpsilon)
        gradInput.setValue(curTarget,
          ev.times(gradInput.valueAt(curTarget), ev.inv(clipped)))
      }
    }
    else if (input.dim() == 2) {
      val batchSize = input.size(1)
      val targetSize = target.size()
      target.squeeze()
      if (resultsBackward == null || resultsBackward.length != batchSize) {
        resultsBackward = new Array[Future[_]](batchSize)
      }

      var i = 1
      while (i <= batchSize) {
        val _i = i
        resultsBackward(_i - 1) = Engine.model.invoke(() => {
          val curTarget = ev.toType[Int](target.valueAt(_i))
          if (curTarget != paddingValue) {
            gradInput.setValue(_i, curTarget, if (weights != null) ev.times(ev.fromType[Int](-1),
              weights.valueAt(curTarget))
            else ev.fromType[Int](-1))
            if (sizeAverage) gradInput.setValue(_i, curTarget, ev.divide(gradInput.valueAt(_i,
              curTarget), total_weight))
            if (!logProbAsInput) {
              val clipped = ev.clip(input.valueAt(_i, curTarget), epsilon, oneMinusEpsilon)
              gradInput.setValue(_i, curTarget,
                ev.times(gradInput.valueAt(_i, curTarget), ev.inv(clipped)))
            }
          }
        })
        i += 1
      }

      i = 0
      while (i < batchSize) {
        Await.result(resultsBackward(i), Duration.Inf)
        i += 1
      }
      target.resize(targetSize)
    }
    gradInput
  }
}

object ZooClassNLLCriterion {
  def apply[@specialized(Float, Double) T: ClassTag](
    weights: Tensor[T] = null,
    sizeAverage: Boolean = true,
    logProbAsInput: Boolean = true,
    paddingValue: Int = -1
  )(implicit ev: TensorNumeric[T]) : ZooClassNLLCriterion[T] = {
    new ZooClassNLLCriterion[T](weights, sizeAverage, logProbAsInput, paddingValue)
  }
}
