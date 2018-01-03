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

package com.intel.analytics.bigdl.nn.mkldnn

import com.intel.analytics.bigdl.nn.ReLU
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

class MemoryReOrderSpec extends FlatSpec with Matchers {

  "reorder dnn should be same with bigdl relu" should "work correctly" in {
    val reludnn = ReLUDnn[Float](ip = false)
    val input = Tensor[Float](4, 20, 5, 5).rand(-1, 1)
    val output1 = reludnn.forward(input)
    val tmp = output1.clone()

    val reorder = MemoryReOrder[Float]()
    val output = reorder.forward(output1)


    DnnUtils.nearequals(output, tmp)
    println("done")
  }
}
