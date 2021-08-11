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

package com.intel.analytics.bigdl.models

import com.intel.analytics.bigdl.example.loadmodel.{AlexNet, AlexNet_OWT}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class AlexNetSpec extends FlatSpec with Matchers {
  "ALexNet_OWT graph" should "be same with original one" in {
    Random.setSeed(1)
    val batchSize = 4
    val input = Tensor[Float](batchSize, 3, 224, 224).apply1(e => Random.nextFloat())
    val gradOutput = Tensor[Float](batchSize, 1000).apply1(e => Random.nextFloat())

    RNG.setSeed(1000)
    val model = AlexNet_OWT(1000, false, true)
    RNG.setSeed(1000)
    val graphModel = AlexNet_OWT.graph(1000, false, true)

    var output1: Tensor[Float] = null
    var output2: Tensor[Float] = null
    for (i <- 1 to 2) {
      output1 = model.forward(input).toTensor[Float]
      output2 = graphModel.forward(input).toTensor[Float]
    }
    output1 should be(output2)

    var gradInput1: Tensor[Float] = null
    var gradInput2: Tensor[Float] = null
    for (i <- 1 to 2) {
      gradInput1 = model.backward(input, gradOutput).toTensor[Float]
      gradInput2 = graphModel.backward(input, gradOutput).toTensor[Float]
    }
    gradInput1 should be(gradInput2)
    model.getParametersTable().equals(graphModel.getParametersTable()) should be (true)
  }

  "ALexNet graph" should "be same with original one" in {
    Random.setSeed(1)
    val batchSize = 4
    val input = Tensor[Float](batchSize, 3, 256, 256).apply1(e => Random.nextFloat())
    val gradOutput = Tensor[Float](batchSize, 1000).apply1(e => Random.nextFloat())

    RNG.setSeed(1000)
    val model = AlexNet(1000, false)
    RNG.setSeed(1000)
    val graphModel = AlexNet.graph(1000, false)

    var output1: Tensor[Float] = null
    var output2: Tensor[Float] = null
    for (i <- 1 to 2) {
      output1 = model.forward(input).toTensor[Float]
      output2 = graphModel.forward(input).toTensor[Float]
    }
    output1 should be(output2)

    var gradInput1: Tensor[Float] = null
    var gradInput2: Tensor[Float] = null
    for (i <- 1 to 2) {
      gradInput1 = model.backward(input, gradOutput).toTensor[Float]
      gradInput2 = graphModel.backward(input, gradOutput).toTensor[Float]
    }
    gradInput1 should be(gradInput2)

    model.getParametersTable().equals(graphModel.getParametersTable()) should be (true)
  }
}
