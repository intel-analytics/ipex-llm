/*
 * Copyright 2021 The BigDL Authors.
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

package com.intel.analytics.zoo.narwhal.fl.vertical.tree

import com.intel.analytics.bigdl.ppml.vfl.fgboost.TreeObjective
import com.intel.analytics.bigdl.ppml.vfl.fgboost.TreeUtils._

class LogisticObjective extends TreeObjective {
  def getGradient(predict: Array[Float],
                  label: Array[Float]): Array[Array[Float]] = {
    // transform to sigmoid
    val logPredict = predict.map(sigmoidFloat)
    val grad = logPredict.zip(label).map(x => x._1 - x._2)
    // hess = p * (1 - p)
    val hess = logPredict.map(x => x * (1 - x))
    Array(grad, hess)
  }

  def getLoss(predict: Array[Float],
              label: Array[Float]): Float = {
    var error: Float = 0f
    for (i <- predict.indices) {
      if (label(i) == 0.0 && predict(i) > 0) {
        error += 1
      } else if (label(i) == 1.0 && predict(i) <= 0) {
        error += 1
      }
    }
    error / predict.length
  }
}
