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

package com.intel.analytics.bigdl.optim

import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * Precision Recall Area Under Curve will compute the precision-recall pairs and
 * get the area under the curve.
 *
 * Note: It will gather all output probabilities and targets to driver and will compute the
 * precision, recall and the auc every calling of `result()`
 *
 * @param ev tensor numeric environments
 * @tparam T class tag for tensor numeric
 */
class PrecisionRecallAUC[T: ClassTag]()(implicit ev: TensorNumeric[T]) extends ValidationMethod[T] {
  override def apply(output: Activity, target: Activity): ValidationResult = {
    require(output.isTensor && target.isTensor, s"only support tensor output and tensor target")
    require(!output.toTensor.isEmpty && !target.toTensor.isEmpty,
      s"the output and target should not be empty")
    val array = List(output, target).map(_.toTensor[Float].storage().array())
    val results = array.head.zip(array.last).toArray
    new PRAUCResult(results)
  }

  override protected def format(): String = s"PrecisionRecallAUC"
}

class PRAUCResult(val results: Array[(Float, Float)]) extends ValidationResult {
  override def result(): (Float, Int) = {
    val sorted = results.sortBy(_._1).reverse
    val totalPositive = sorted.count(_._2 == 1)

    var truePositive = 0.0f
    var falsePositive = 0.0f

    var areaUnderCurve = 0.0f
    var prevPrecision = 1.0f
    var prevRecall = 0.0f

    var i = 0
    while (truePositive != totalPositive) {
      val target = sorted(i)._2

      if (target == 1.0f) {
        truePositive += 1
      } else {
        falsePositive += 1
      }

      val precision = truePositive / (truePositive + falsePositive)
      val recall = truePositive / totalPositive

      areaUnderCurve += (recall - prevRecall) * (precision + prevPrecision)

      prevRecall = recall
      prevPrecision = precision

      i += 1
    }

    (areaUnderCurve / 2, results.length)
  }

  // scalastyle:off methodName
  override def +(other: ValidationResult): ValidationResult = {
    new PRAUCResult(results ++ other.asInstanceOf[PRAUCResult].results)
  }
  // scalastyle:on

  override protected def format(): String = {
    val getResult = result()
    s"Precision Recall AUC is ${getResult._1} on ${getResult._2}"
  }
}

