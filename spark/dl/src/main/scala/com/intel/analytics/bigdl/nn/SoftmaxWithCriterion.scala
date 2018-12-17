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

import com.intel.analytics.bigdl.nn.NormMode.NormMode
import com.intel.analytics.bigdl.nn.abstractnn.TensorCriterion
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * Computes the multinomial logistic loss for a one-of-many classification task,
 * passing real-valued predictions through a softmax to get a probability distribution over classes.
 * It should be preferred over separate SoftmaxLayer + MultinomialLogisticLossLayer
 * as its gradient computation is more numerically stable.
 * @param ignoreLabel   (optional) Specify a label value that
 *                      should be ignored when computing the loss.
 * @param normalizeMode How to normalize the output loss.
 */
class SoftmaxWithCriterion[@specialized(Float, Double) T: ClassTag](ignoreLabel: Option[Int] = None,
  normalizeMode: NormMode = NormMode.VALID)
  (implicit ev: TensorNumeric[T]) extends TensorCriterion[T] {
  @transient var softmax: SoftMax[T] = _
  @transient var prob: Tensor[T] = _
  @transient var outerNum = 0 // batchsize
  @transient var innerNum = 1
  @transient var nClasses = 2

  /**
   * compute the loss
   * @param input input.size(1) is batch num
   *              input.size(2) is the softmaxAxis, number of classes as usual
   * @return
   */
  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    outerNum = input.size(1)
    innerNum = 1
    var i = 3
    while (i <= input.dim()) {
      innerNum = innerNum * input.size(i)
      i += 1
    }
    nClasses = input.size(2)
    if (softmax == null) {
      softmax = new SoftMax()
    }
    prob = softmax.forward(input)
    val probData = prob.storage().array()
    val labelData = target.storage().array()
    var loss = ev.fromType(0)
    val dim = prob.nElement() / outerNum
    var count = 0
    i = 0
    while (i < outerNum) {
      var j = 0
      while (j < innerNum) {
        val curTarget = ev.toType[Int](labelData(i * innerNum + j))
        if (ignoreLabel.isEmpty || ignoreLabel.get != curTarget) {
          assert(curTarget >= 1 && curTarget <= nClasses,
            s"curTarget $curTarget is out of range 1 to ${ nClasses } ")
          // avoid log(0)
          val prob = ev.max(probData(i * dim + (curTarget - 1) * innerNum + j),
            ev.fromType(Double.MinPositiveValue))
          loss = ev.minus(loss, ev.log(prob))
          count = count + 1
        }
        j += 1
      }
      i += 1
    }
    output = ev.divide(loss, getNormalizer(normalizeMode, count))
    output
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(prob).copy(prob)
    val labelData = target.storage().array()

    val dim = prob.nElement() / outerNum
    val gradData = gradInput.storage().array()
    var count = 0
    var i = 0
    while (i < outerNum) {
      var j = 0
      while (j < innerNum) {
        val curTarget = ev.toType[Int](labelData(i * innerNum + j))
        if (ignoreLabel.isEmpty || ignoreLabel.get != curTarget) {
          gradData(i * dim + (curTarget - 1) * innerNum + j) =
            ev.minus(gradData(i * dim + (curTarget - 1) * innerNum + j), ev.fromType(1))
          count = count + 1
        } else {
          var c = 0
          while (c < nClasses) {
            gradData(i * dim + c * innerNum + j) = ev.fromType(0)
            c += 1
          }
        }
        j += 1
      }
      i += 1
    }

    val lossWeight = ev.divide(ev.fromType(1), getNormalizer(normalizeMode, count))
    i = 0
    while (i < gradData.length) {
      gradData(i) = ev.times(gradData(i), lossWeight)
      i += 1
    }
    gradInput
  }

  /**
   * Read the normalization mode parameter and compute the normalizer based on the input size.
   * If normalizeMode is VALID, the count of valid outputs will be read from validCount,
   * unless it is -1 in which case all outputs are assumed to be valid.
   */
  def getNormalizer(normalizeMode: NormMode, validCount: Int): T = {
    def normalizer = {
      normalizeMode match {
        case NormMode.FULL => ev.fromType(outerNum * innerNum)
        case NormMode.VALID =>
          if (validCount == -1) {
            ev.fromType(outerNum * innerNum)
          }
          else {
            ev.fromType(validCount)
          }
        case NormMode.BATCH_SIZE => ev.fromType(outerNum)
        case NormMode.NONE => ev.fromType(1)
        case _ => throw new IllegalArgumentException("Unknown normalization mode")
      }
    }
    ev.max(ev.fromType(1), normalizer)
  }
}

object NormMode extends Enumeration {
  type NormMode = Value
  val FULL, VALID, BATCH_SIZE, NONE = Value
}

object SoftmaxWithCriterion {
  def apply[@specialized(Float, Double) T: ClassTag](ignoreLabel: Option[Int] = None,
    normalizeMode: NormMode = NormMode.VALID)
    (implicit ev: TensorNumeric[T]): SoftmaxWithCriterion[T] =
    new SoftmaxWithCriterion[T](ignoreLabel, normalizeMode)
}
