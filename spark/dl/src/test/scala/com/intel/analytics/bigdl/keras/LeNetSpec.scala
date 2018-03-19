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

package com.intel.analytics.bigdl.keras

import com.intel.analytics.bigdl.example.keras.LeNet
import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{FlatSpec, Matchers}

class LeNetSpec extends FlatSpec with Matchers {

  "LeNet" should "generate the correct outputShape" in {
    val cnn = LeNet(classNum = 10)
    cnn.getOutputShape().toSingle().toArray should be (Array(-1, 10))
  }

  "LeNet forward and backward" should "work properly" in {
    val cnn = LeNet(classNum = 10)
    val input = Tensor[Float](Array(2, 28, 28, 1)).rand()
    val output = cnn.forward(input)
    val gradInput = cnn.backward(input, output)
  }

  "LeNet forward with incompatible input tensor" should "raise an exception" in {
    intercept[RuntimeException] {
      val cnn = LeNet(classNum = 10)
      val input = Tensor[Float](Array(28, 28, 1)).rand()
      val output = cnn.forward(input)
    }
  }

}
