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
class MulSpec extends FlatSpec with Matchers {

  "A Mul Module with scaleW " should "generate correct output and grad" in {
    val inputN = 5
    val seed = 100
    RNG.setSeed(seed)
    val input = Tensor[Double](1, 5).apply1(e => Random.nextDouble())
    val gradOutput = Tensor[Double](5).apply1(e => Random.nextDouble())
    val layer1 = new Mul[Double]()
    val layer2 = new Mul[Double]()
    val (weights, grad) = layer1.getParameters()
    val (w, g) = layer2.getParameters()
    w.copy(weights)
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

class MulSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val mul = Mul[Float]().setName("mul")
    val input = Tensor[Float](10, 10).apply1(_ => Random.nextFloat())
    runSerializationTest(mul, input)
  }
}
