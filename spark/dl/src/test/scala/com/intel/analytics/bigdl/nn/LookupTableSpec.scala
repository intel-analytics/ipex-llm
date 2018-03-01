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
class LookupTableSpec extends FlatSpec with Matchers {

  "A LookupTable with padding input" should "generate correct output" in {
    val seed = 100
    RNG.setSeed(seed)
    val module = LookupTable[Double](9, 4, paddingValue = 1, maskZero = true)
    val input = Tensor[Double](5)
    input(Array(1)) = 5
    input(Array(2)) = 1
    input(Array(3)) = 6
    input(Array(4)) = 9
    input(Array(5)) = 4

    val output = module.forward(input)

    output.select(1, 2).sum() should be (0.0)
  }

  "A LookupTableSpec with scaleW" should "generate correct output and grad with input 1D" in {
    val seed = 100
    RNG.setSeed(seed)
    val module = LookupTable[Double](9, 4, 2, 0.1, 2.0, true)
    val input = Tensor[Double](5)
    input(Array(1)) = 5
    input(Array(2)) = 2
    input(Array(3)) = 6
    input(Array(4)) = 9
    input(Array(5)) = 4

    val gradOutput = Tensor[Double](5, 4).randn()
    val layer1 = new LookupTable[Double](9, 4, 2, 0.1, 2.0, true)
    val layer2 = new LookupTable[Double](9, 4, 2, 0.1, 2.0, true)
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

  "A LookupTableSpec with scaleW" should "generate correct output and grad with input 2D" in {

    val seed = 100
    RNG.setSeed(seed)
    val module = LookupTable[Double](10, 3, 3)
    val input = Tensor[Double](2, 4)
    input(Array(1, 1)) = 1
    input(Array(1, 2)) = 2
    input(Array(1, 3)) = 4
    input(Array(1, 4)) = 5
    input(Array(2, 1)) = 4
    input(Array(2, 2)) = 3
    input(Array(2, 3)) = 2
    input(Array(2, 4)) = 10

    val layer1 = new LookupTable[Double](10, 3, 3)
    val layer2 = new LookupTable[Double](10, 3, 3)
    val (weights, grad) = layer1.getParameters()
    val (w, g) = layer2.getParameters()
    w.copy(weights)
    layer2.setScaleW(2)

    val output1 = layer1.forward(input)
    val output2 = layer2.forward(input)
    output1 should be (output2)

    val gradOutput = Tensor[Double](output1)
    val gradInput1 = layer1.backward(input, gradOutput)
    val gradInput2 = layer2.backward(input, gradOutput)


    gradInput1 should be (gradInput2)

    layer2.gradWeight should be (layer1.gradWeight.mul(2))
  }
}

class LookupTableSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val lookupTable = LookupTable[Float](9, 4, 2, 0.1, 2.0, true).setName("lookupTable")
    val input = Tensor[Float](5)
    input(Array(1)) = 5
    input(Array(2)) = 2
    input(Array(3)) = 6
    input(Array(4)) = 9
    input(Array(5)) = 4
    runSerializationTest(lookupTable, input)
  }
}
