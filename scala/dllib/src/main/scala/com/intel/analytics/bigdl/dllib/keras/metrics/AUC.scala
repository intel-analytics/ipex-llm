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

package com.intel.analytics.bigdl.dllib.keras.metrics

import com.intel.analytics.bigdl.dllib.nn.abstractnn.Activity
import com.intel.analytics.bigdl.dllib.optim.{ValidationMethod, ValidationResult}
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric

import scala.collection.mutable.ArrayBuffer

/**
 * Represent an accuracy result. It's equal to the probability that a
 * classifier will rank a randomly chosen positive instance higher
 * than a randomly chosen negative one.
 * Refer: https://en.wikipedia.org/wiki/Receiver_operating_characteristic
 * @param tp True positive numbers
 * @param fp False positive numbers
 * @param po Positive numbers
 * @param ne Negative numbers
 */
class AucScore(private val tp: Tensor[Float], private val fp: Tensor[Float],
               private var po: Tensor[Float], private var ne: Tensor[Float])
  extends ValidationResult {
  require(fp.dim() == tp.dim(), "fp dimension should be the same with tp dimension")
  require(po.dim() == ne.dim(), "positive value dimension should be the" +
    "same with negative value dimension")

  override def result(): (Float, Int) = {
    (getScores.last, (po.valueAt(1) + ne.valueAt(1)).toInt)
  }

  // scalastyle:off methodName
  override def +(other: ValidationResult): ValidationResult = {
    val otherResult = other.asInstanceOf[AucScore]
    this.fp.add(otherResult.fp)
    this.tp.add(otherResult.tp)
    this.po.add(otherResult.po)
    this.ne.add(otherResult.ne)
    this
  }
  // scalastyle:on methodName

  override protected def format(): String = {
    val scores = getScores
    var str = s"(Average score: ${scores.last}, count: ${(po.valueAt(1) + ne.valueAt(1)).toInt})"
    if (scores.length > 1) {
      str += s"\nscore for each class is:\n"
      scores.take(scores.length - 1).foreach(s => str += s"${s} \n")
    }
    str
  }

  private def computeAUC(slicedTp: Tensor[Float], slicedFp: Tensor[Float],
    slicedPo: Float, slicedNe: Float): Float = {
    val epsilon = 1e-6.toFloat
    val tpr = slicedTp.clone().add(epsilon).div(slicedPo + epsilon)
    val fpr = slicedFp.clone().div(slicedNe + epsilon)

    (tpr.narrow(1, 1, tpr.nElement() - 1) + tpr.narrow(1, 2, tpr.nElement() - 1)).dot(
      (fpr.narrow(1, 1, fpr.nElement() - 1) - fpr.narrow(1, 2, fpr.nElement() - 1))) / 2
  }

  // return score, the first n element is auc for each class, the last element is the average auc
  private def getScores: Array[Float] = {
    val scores = new ArrayBuffer[Float]()
    if (fp.dim() == 1) scores.append(computeAUC(tp, fp, po.valueAt(1), ne.valueAt(1)))
    else {
        val classNum = fp.size(2)
        val weights = Tensor.ones[Float](classNum)
        for(i <- 1 to classNum) {
          scores.append(computeAUC(tp.select(2, i), fp.select(2, i), po.valueAt(i), ne.valueAt(i)))
        }
      val averageScore = Tensor(scores.toArray, Array(classNum)).dot(weights) / weights.sum()
      scores.append(averageScore)
    }
    scores.toArray
  }

  override def equals(obj: Any): Boolean = {
    if (obj == null) {
      return false
    }
    if (!obj.isInstanceOf[AucScore]) {
      return false
    }
    val other = obj.asInstanceOf[AucScore]
    if (this.eq(other)) {
      return true
    }
    this.tp.equals(other.tp) &&
    this.fp.equals(other.fp) &&
    this.po == other.po &&
    this.ne == other.ne
  }

  override def hashCode(): Int = {
    val seed = 37
    var hash = 1
    hash = hash * seed + this.po.sum().toInt
    hash = hash * seed + this.ne.sum().toInt
    hash = hash * seed + this.fp.sum().toInt
    hash = hash * seed + this.tp.sum().toInt
    hash
  }
}

/**
 * Area under ROC cure.
 * Metric for binary(0/1) classification, support single label and multiple labels.
 * @param thresholdNum The number of thresholds. The quality of approximation
 *                     may vary depending on thresholdNum.
 */
class AUC[T](thresholdNum: Int = 200)(implicit ev: TensorNumeric[T])
  extends ValidationMethod[T] {

  override def apply(output: Activity, target: Activity):
  ValidationResult = {
    val _output = if (output.asInstanceOf[Tensor[T]].dim() == 2) {
      output.asInstanceOf[Tensor[T]].clone().squeeze(2)
    } else {
      output.asInstanceOf[Tensor[T]].clone().squeeze()
    }

    val _target = if (target.asInstanceOf[Tensor[T]].dim() == 2) {
      target.asInstanceOf[Tensor[T]].clone().squeeze(2)
    } else {
      target.asInstanceOf[Tensor[T]].clone().squeeze()
    }
    require(_output.dim() <= 2 && _target.dim() <= 2,
      s"${_output.dim()} dim format is not supported")
    require(_output.dim() == _target.dim(),
      s"output dimension must be the same with target!!" +
        s"out dimension is: ${_output.dim()}, target dimension is: ${_target.dim()}")

    if (_output.dim() == 1) {
      val (tp, fp, po, ne) = getRocCurve(_output, _target)
      new AucScore(tp, fp, Tensor(Array[Float](po), Array(1)),
        Tensor(Array[Float](ne), Array(1)))
    } else {
      val classNum = _output.size(2)
      val _tp = Tensor[Float](thresholdNum, classNum)
      val _fp = Tensor[Float](thresholdNum, classNum)
      val _po = new Array[Float](classNum)
      val _ne = new Array[Float](classNum)

      for(i <- 1 to classNum) {
        val _output_i = _output.select(2, i)
        val _target_i = _target.select(2, i)
        val res = getRocCurve(_output_i, _target_i)
        _tp.select(2, i).copy(res._1)
        _fp.select(2, i).copy(res._2)
        _po(i - 1) = res._3
        _ne(i - 1) = res._4
      }
      new AucScore(_tp, _fp, Tensor(_po, Array(classNum)), Tensor(_ne, Array(classNum)))
    }
  }

  override def format(): String = {
    "AucScore"
  }

  // get tp(true positive), fp(flase positive), positive, negative
  private def getRocCurve(output: Tensor[T], target: Tensor[T]):
    (Tensor[Float], Tensor[Float], Float, Float) = {
    val thresholds = new Array[Float](thresholdNum)
    val kepsilon = 1e-7.toFloat
    thresholds(0) = 0 - kepsilon
    thresholds(thresholdNum - 1) = 1 + kepsilon
    for (i <- 1 until thresholdNum - 1) {
      thresholds(i) = (i + 1) * 1.0f / (thresholdNum - 1)
    }

    val tp = new Array[Float](thresholdNum)
    val fp = new Array[Float](thresholdNum)
    var j = 0

    while (j < thresholdNum) {
      output.map(target, (a, b) => {
        val fb = ev.toType[Float](b)
        if (fb != 1 && fb != 0) {
          throw new UnsupportedOperationException("Only support binary(0/1) target")
        }
        if (ev.isGreaterEq(a, ev.fromType[Float](thresholds(j)))) {
          if (fb == 1) tp(j) += 1
          else fp(j) += 1
        }
        a })
      j += 1
    }

    (Tensor(tp, Array(thresholdNum)), Tensor(fp, Array(thresholdNum)),
      ev.toType[Float](target.sum()),
      ev.toType[Float](target.mul(ev.fromType(-1)).add(ev.fromType(1)).sum))
  }
}
