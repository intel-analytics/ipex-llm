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

package com.intel.analytics.bigdl.ppml.vfl

import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.ppml.example.DebugLogger
import com.intel.analytics.bigdl.ppml.fgboost.common.TreeUtils
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class TreeUtilsSpec extends FlatSpec with Matchers with BeforeAndAfter with DebugLogger {
  "Sort feature index by value" should "work" in {
    /**
     * create a Tensor
     * [ [4, 3, 1],
     *   [2, 5, 6] ]
     */
    val tensor1 = Tensor[Float](Array(4f, 3, 1), Array(3))
    val tensor2 = Tensor[Float](Array(2f, 5, 6), Array(3))
    val sortedIndex = TreeUtils.sortByFeature(Array(tensor1, tensor2))
    require(sortedIndex(0).sameElements(Array(1, 0)), "feature 0 sorted index wrong")
    require(sortedIndex(1).sameElements(Array(0, 1)), "feature 1 sorted index wrong")
    require(sortedIndex(2).sameElements(Array(0, 1)), "feature 2 sorted index wrong")
  }
}
