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

import com.intel.analytics.bigdl.tensor.Tensor

object EvaluateMethods {
  def calcAccuracy[T](output: Tensor[T], target: Tensor[T]): (Int, Int) = {
    var correct = 0
    var count = 0
    if (output.dim() == 2) {
      output.max(2)._2.squeeze().map(target, (a, b) => {
        if (a == b) {
          correct += 1
        }
        a
      })
      count += output.size(1)
    } else if (output.dim == 1) {
      require(target.size(1) == 1)
      output.max(1)._2.map(target, (a, b) => {
        if (a == b) {
          correct += 1
        }
        a
      })
      count += 1
    } else {
      throw new IllegalArgumentException
    }

    (correct, count)
  }

  def calcTop5Accuracy[T](output: Tensor[T], target: Tensor[T]): (Int, Int) = {
    var correct = 0
    var count = 0
    if (output.dim() == 2) {
      val indices = output.topk(5, 2, false)._2
      var i = 1
      while (i <= output.size(1)) {
        if (indices.valueAt(i, 1) == target.valueAt(i)
          || indices.valueAt(i, 2) == target.valueAt(i)
          || indices.valueAt(i, 3) == target.valueAt(i)
          || indices.valueAt(i, 4) == target.valueAt(i)
          || indices.valueAt(i, 5) == target.valueAt(i)) {
          correct += 1
        }
        i += 1
      }
      count += output.size(1)
    } else if (output.dim == 1) {
      require(target.size(1) == 1)
      val indices = output.topk(5, 1, false)._2
      if (indices.valueAt(1) == target.valueAt(1) || indices.valueAt(2) == target.valueAt(1)
        || indices.valueAt(3) == target.valueAt(1) || indices.valueAt(4) == target.valueAt(1)
        || indices.valueAt(5) == target.valueAt(1)) {
        correct += 1
      }
      count += 1
    } else {
      throw new IllegalArgumentException
    }

    (correct, count)
  }
}
