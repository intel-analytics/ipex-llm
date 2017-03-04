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

import com.intel.analytics.bigdl.nn.abstractnn.TensorCriterion
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.Tensor

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}
import scala.reflect.ClassTag
import com.intel.analytics.bigdl.utils.Engine

/**
 * This criterion accept a negative log like hood input tensor, which can be 1D(for a sample) or 2D
 * (for a batch). The target tensor is 1D. For a 1D input, the target tensor is a 1x1 tensor
 * contains the class id. The class id is range is contiguous, and the max class id should not be
 * larger than the 1D input length. The class id count from 1. For a 2D input, the target tensor
 * is a 1 x batch_size tensor.
 *
 * The calculation is simple. The loss is - log like hood of the class and the gradient is -1
 *
 * @param weights define weights for different class
 * @param sizeAverage average the loss/gradient for batch input
 * @param repeat the input can contain multiple set. For example, if repeat = 2, there're two sets
 *               input, and if there are 10 classes, for a 1D tensor input, its length should be 20
 * @param sumRepeatLoss If there're multiple input set, summ the loss togethor or only return the
 *                      first loss
 * @tparam T
 */
@SerialVersionUID(- 8696382776046599502L)
class ClassNLLCriterion[T: ClassTag](
  weights: Tensor[T] = null,
  sizeAverage: Boolean = true,
  repeat: Int = 1,
  sumRepeatLoss: Boolean = false
)(implicit ev: TensorNumeric[T]) extends TensorCriterion[T] {

  private var total_weight = ev.zero

  if (weights != null) require(weights.dim() == 1, "weights input should be 1-D Tensor")

  @transient
  private var results: Array[Future[(T, T)]] = null
  @transient
  private var resultsBackward: Array[Future[_]] = null

  private var nClasses = 0

  @inline
  private def getLoss(input : Tensor[T], i : Int, target : Tensor[T]) : (T, T) = {
    val curTarget = ev.toType[Int](target.valueAt(i))
    require(curTarget >= 1 && curTarget <= nClasses,
      s"curTarget ${curTarget} is out of range 1 to ${nClasses}")
    val curWeight = if (weights != null) weights(Array(curTarget)) else ev.one
    var loss = ev.times(ev.negative(input.valueAt(curTarget)), curWeight)
    if(sumRepeatLoss) {
      var r = 1
      while(r < repeat) {
        loss = ev.plus(loss, ev.times(ev.negative(input.valueAt(curTarget + r * nClasses)),
          curWeight))
        r += 1
      }
    }
    (loss, curWeight)
  }

  @inline
  private def getGradient(i : Int, target : Tensor[T]) : Unit = {
    val curTarget = ev.toType[Int](target.valueAt(i))
    val curWeight = if (weights != null) ev.negative(weights.valueAt(curTarget)) else ev.negativeOne
    var r = 0
    while(r < repeat) {
      gradInput.setValue(curTarget + r * nClasses, curWeight)
      if (sizeAverage) {
        gradInput.setValue(curTarget + r * nClasses,
          ev.divide(gradInput.valueAt(curTarget), total_weight))
      }
      r += 1
    }
  }

  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    require(input.dim() == 1 || input.dim() == 2,
      "ClassNLLCriterion: " + ErrorInfo.constrainInputAsVectorOrBatch)

    val length = input.size(input.dim())
    require(length % repeat == 0, s"$repeat cannot divide $length")
    nClasses = length / repeat

    if (input.dim() == 1) {
      require(input.dim() == target.dim(),
        "ClassNLLCriterion: " + ErrorInfo.constrainInputDimSameAsTarget)
      val res = getLoss(input, 1, target)
      output = res._1
      total_weight = res._2
    } else if (input.dim() == 2) {
      val batchSize = input.size(1)
      val targetSize = target.size()
      target.squeeze()
      total_weight = ev.zero
      output = ev.zero

      if (results == null || results.length != batchSize) {
        results = new Array[Future[(T, T)]](batchSize)
      }

      var i = 1
      while (i <= batchSize) {
        val _i = i
        results(_i - 1) = Engine.model.invoke( () => {
          getLoss(input, _i, target)
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
      "ClassNLLCriterion: " + ErrorInfo.constrainInputAsVectorOrBatch)
    require(ev.toType[Double](total_weight) > 0.0, "total weight must larger than 0")
    gradInput.resizeAs(input)
    gradInput.zero()

    if (input.dim() == 1) {
      require(input.dim() == target.dim(),
        "ClassNLLCriterion: " + ErrorInfo.constrainInputDimSameAsTarget)
      getGradient(1, target)
    } else if (input.dim() == 2) {
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
          getGradient(_i, target)
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

object ClassNLLCriterion {
  def apply[@specialized(Float, Double) T: ClassTag](
      weights: Tensor[T] = null,
      sizeAverage: Boolean = true)(implicit ev: TensorNumeric[T]) : ClassNLLCriterion[T] = {
    new ClassNLLCriterion[T](weights, sizeAverage)
  }
}
