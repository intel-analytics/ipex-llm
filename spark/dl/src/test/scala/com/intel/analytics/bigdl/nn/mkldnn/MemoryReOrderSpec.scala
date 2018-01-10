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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.{Graph, _}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import org.scalatest.{FlatSpec, Matchers}

class MemoryReOrderSpec extends FlatSpec with Matchers {

  "reorder dnn should be same with bigdl relu" should "work correctly" in {
    val reludnn = ReLUDnn[Float](ip = false)
    val input = Tensor[Float](1, 3, 2, 2).rand(-1, 1)
    val gradOutput = Tensor[Float](1, 3, 2, 2).rand(-1, 1)
    val output1 = reludnn.forward(input)
    val grad1 = reludnn.backward(input, gradOutput)

    val reorder = MemoryReOrder[Float]()
    val output = reorder.forward(output1)
    val grad = reorder.backward(output1, grad1)

    println(output1)
    println(output)

    println(grad1)
    println(grad)

    println("done")
  }
}
