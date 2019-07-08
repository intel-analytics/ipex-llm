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

import com.intel.analytics.bigdl.nn
import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.Table
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

import scala.util.Random

@com.intel.analytics.bigdl.tags.Parallel
class CMulSpec extends FlatSpec with Matchers {

  "A CMul with scaleW" should "work correctly" in {
    val seed = 100
    RNG.setSeed(seed)

    val input = Tensor[Double](5, 4)
    var i = 0
    input.apply1(_ => {i += 1; i})
    val gradOutput = Tensor[Double](5, 4)
    i = 0
    gradOutput.apply1(_ => {i += 1; i*0.1})

    val layer1 = new CMul[Double](Array(5, 1))
    val layer2 = layer1.cloneModule().asInstanceOf[CMul[Double]].setScaleW(0.5)

    val output1 = layer1.forward(input)
    val gradInput1 = layer1.backward(input, gradOutput)
    val output2 = layer2.forward(input)
    val gradInput2 = layer2.backward(input, gradOutput)
    val cmul = CMul[Float](Array[Int](1, 4096, 1, 1))
    val cmul2 = cmul.cloneModule().asInstanceOf[CMul[Float]]
    val input1 = Tensor[Float](300, 4096).randn()
    i = 0
    input1.apply1(_ => {i += 1; i})
    val gradOutput_1 = Tensor[Float](300, 4096)
    i = 0
    gradOutput_1.apply1(_ => {i += 1; i})
    val output3 = cmul.forward(input1)
    val gradInput3 = cmul.backward(input1, gradOutput_1)
    val output4 = cmul2.forward(input1)
    val gradInput4 = cmul2.backward(input1, gradOutput_1)


    output1 should be (output2)
    gradInput1 should be (gradInput2)
    output3 should be (output4)
    gradInput3 should be (gradInput4)
    layer2.gradWeight should be (layer1.gradWeight.mul(0.5))
  }

  "CMUl" should "works well on batch input" in {
    val model = nn.CMul[Float](Array(1, 64, 1, 1))
    val model2 = model.cloneModule()

    val batchInput = Tensor[Float](64, 64, 112, 112).rand()
    val input = batchInput.select(1, 1).resize(Array(1, 64, 112, 112))

    model.evaluate()
    model2.evaluate()

    val out1 = model.forward(batchInput)
    val out2 = model2.forward(input).resize(Array(64, 112, 112))

    out2 should be(out1.select(1, 1))
  }
}

class CMulSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val input = Tensor[Float](5, 1).apply1(e => Random.nextFloat())
    val cmul = CMul[Float](Array(5, 1)).setName("cmul")
    runSerializationTest(cmul, input)
  }
}
