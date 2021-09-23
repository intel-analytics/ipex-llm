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

import com.intel.analytics.bigdl.dllib.optim.LossResult
import com.intel.analytics.bigdl.dllib.tensor.{Storage, Tensor}
import org.scalatest.{FlatSpec, Matchers}

class MSESpec extends FlatSpec with Matchers {
  "MSE" should "be correct in 1d tensor" in {
    val output = Tensor(Storage(Array[Double](
      5, 0, 5, 0
    )))
    val target = Tensor(Storage(Array[Double](
      3, 2, 1, 0
    )))
    val validation = new MSE[Double]()
    val result = validation(output, target)
    val test = new LossResult(6f, 1)
    result should be(test)
  }
}
