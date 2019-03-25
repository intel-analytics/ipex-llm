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

package com.intel.analytics.zoo.pipeline.api.keras.metrics

import com.intel.analytics.bigdl.optim.LossResult
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import org.scalatest.{FlatSpec, Matchers}

class MAESpec extends FlatSpec with Matchers {
  "MAE" should "be correct in 1d tensor" in {
    val output = Tensor(Storage(Array[Double](
      0, 0, 0, 0
    )))
    val target = Tensor(Storage(Array[Double](
      1, 2, 1, 0
    )))
    val validation = new MAE[Double]()
    val result = validation(output, target)
    val test = new LossResult(1f, 1)
    result should be(test)
  }
}
