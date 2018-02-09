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

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

import scala.util.Random

@com.intel.analytics.bigdl.tags.Parallel
class EuclideanSpec extends FlatSpec with Matchers {

  "A Euclidean with scaleW" should
    "generate correct output and grad with input one dimension" in {
    val seed = 100
    RNG.setSeed(seed)

    val input = Tensor[Double](7).apply1(e => Random.nextDouble())
    val gradOutput = Tensor[Double](7).apply1(e => Random.nextDouble())

    val layer1 = new Euclidean[Double](7, 7)
    val layer2 = new Euclidean[Double](7, 7)

    val (weights, grad) = layer1.getParameters()
    val (w, g) = layer2.getParameters()
    w.copy(weights)
    g.copy(grad)
    layer2.setScaleW(2)

    val output1 = layer1.forward(input)
    val output2 = layer2.forward(input)
    val gradInput1 = layer1.backward(input, gradOutput)
    val gradInput2 = layer2.backward(input, gradOutput)

    output1 should be (output2)
    gradInput1 should be (gradInput2)

    layer2.gradWeight should be (layer1.gradWeight.mul(2))
  }

  "A Euclidean " should "generate correct output and grad with input two dimensions" in {

    val seed = 100
    RNG.setSeed(seed)

    val input = Tensor[Double](8, 7).apply1(e => Random.nextDouble())
    val gradOutput = Tensor[Double](8, 7).apply1(e => Random.nextDouble())

    val layer1 = new Euclidean[Double](7, 7)
    val layer2 = layer1.cloneModule().asInstanceOf[Euclidean[Double]]
    layer2.setScaleW(2)

    val output1 = layer1.forward(input)
    val output2 = layer2.forward(input)
    val gradInput1 = layer1.backward(input, gradOutput)
    val gradInput2 = layer2.backward(input, gradOutput)

    output1 should be (output2)
    gradInput1 should be (gradInput2)

    layer2.gradWeight should be (layer1.gradWeight.mul(2))
  }
}

class EuclideanSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val euclidean = Euclidean[Float](7, 7).setName("euclidean")
    val input = Tensor[Float](8, 7).apply1(_ => Random.nextFloat())
    runSerializationTest(euclidean, input)
  }
}
