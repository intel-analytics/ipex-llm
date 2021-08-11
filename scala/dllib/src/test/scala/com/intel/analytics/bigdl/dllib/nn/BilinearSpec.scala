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
class BilinearSpec extends FlatSpec with Matchers {

  "A Bilinear with scaleB" should "work correctly" in {
    val seed = 100
    RNG.setSeed(seed)

    val input1 = Tensor[Double](5, 5).apply1(e => Random.nextDouble())
    val input2 = Tensor[Double](5, 3).apply1(e => Random.nextDouble())
    val gradOutput = Tensor[Double](5, 2).apply1(e => Random.nextDouble())

    var input = new Table()
    input(1.toDouble) = input1
    input(2.toDouble) = input2

    val layer1 = new Bilinear[Double](5, 3, 2)
    val layer2 = layer1.cloneModule().asInstanceOf[Bilinear[Double]].setScaleB(2.0)

    val output1 = layer1.forward(input)
    val gradInput1 = layer1.backward(input, gradOutput)
    val output2 = layer2.forward(input)
    val gradInput2 = layer2.backward(input, gradOutput)

    output1 should be (output2)
    gradInput1 should be (gradInput2)

    layer2.gradBias should be (layer1.gradBias.mul(2))
  }
}

class BilinearSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val input1 = Tensor[Float](5, 5).apply1(e => Random.nextFloat())
    val input2 = Tensor[Float](5, 3).apply1(e => Random.nextFloat())
    var input = new Table()
    input(1.toFloat) = input1
    input(2.toFloat) = input2
    val biLinear = Bilinear[Float](5, 3, 2)
    runSerializationTest(biLinear, input)
  }
}
