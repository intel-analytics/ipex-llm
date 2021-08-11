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

import com.intel.analytics.bigdl.nn.abstractnn.{TensorCriterion, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

import scala.util.Random

@com.intel.analytics.bigdl.tags.Parallel
class AddSpec extends FlatSpec with Matchers {

  "A Add with scaleB" should "work correctly" in {
    val inputN = 5
    val seed = 100
    RNG.setSeed(seed)
    val layer1 = new Add[Double](inputN)
    val layer2 = layer1.cloneModule().asInstanceOf[Add[Double]]
      .setScaleB(2.0)

    val input = Tensor[Double](1, 5)
    input(Array(1, 1)) = 1
    input(Array(1, 2)) = 2
    input(Array(1, 3)) = 3
    input(Array(1, 4)) = 4
    input(Array(1, 5)) = 5
    val gradOutput = Tensor[Double](5)
    gradOutput(Array(1)) = 2
    gradOutput(Array(2)) = 5
    gradOutput(Array(3)) = 10
    gradOutput(Array(4)) = 17
    gradOutput(Array(5)) = 26

    val output1 = layer1.forward(input)
    val gradInput1 = layer1.backward(input, gradOutput)
    val output2 = layer2.forward(input)
    val gradInput2 = layer2.backward(input, gradOutput)

    output1 should be (output2)
    gradInput1 should be (gradInput2)

    layer2.gradBias should be (layer1.gradBias.mul(2))
  }
}

class AddSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val add = Add[Float](5).setName("add")
    val input = Tensor[Float](5).apply1(_ => Random.nextFloat())
    runSerializationTest(add, input)
  }
}
