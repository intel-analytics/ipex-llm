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
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

@com.intel.analytics.bigdl.tags.Parallel
class MapTableSpec  extends FlatSpec with Matchers {
  "A MapTable" should "generate correct output" in {
    val input = T(
      Tensor[Float](10).randn(),
      Tensor[Float](10).randn())

    val gradOutput = T(
      Tensor[Float](3).randn(),
      Tensor[Float](3).randn())

    val linear1 = new Linear[Float](10, 3)
    val linear2 = linear1.cloneModule()
    val expectedOutput = T(
      linear1.updateOutput(input(1)),
      linear2.updateOutput(input(2)))

    val map = new MapTable[Float]()
    map.add(linear1)
    val mapOutput = map.forward(input)
    mapOutput should equal (expectedOutput)

    val expectedGradInput = T(
      linear1.updateGradInput(input(1), gradOutput(1)),
      linear2.updateGradInput(input(2), gradOutput(2)))
    val mapGradInput = map.backward(input, gradOutput)

    mapGradInput should equal (expectedGradInput)
  }

  "A MapTable constucted with module" should "generate correct output" in {
    val input = T(
      Tensor[Float](10).randn(),
      Tensor[Float](10).randn())

    val gradOutput = T(
      Tensor[Float](3).randn(),
      Tensor[Float](3).randn())

    val linear1 = new Linear[Float](10, 3)
    val linear2 = linear1.cloneModule()
    val expectedOutput = T(
      linear1.updateOutput(input(1)),
      linear2.updateOutput(input(2)))

    val map = new MapTable[Float](linear1)
    val mapOutput = map.forward(input)
    mapOutput should equal (expectedOutput)

    val expectedGradInput = T(
      linear1.updateGradInput(input(1), gradOutput(1)),
      linear2.updateGradInput(input(2), gradOutput(2)))
    val mapGradInput = map.backward(input, gradOutput)

    mapGradInput should equal (expectedGradInput)
  }

  "A MapTable clearstate" should "add not change modules" in {
    val linear1 = new Linear[Float](10, 3)
    val map = new MapTable[Float](linear1)

    map.clearState()
    map.modules.length should be (1)
  }
}

class MapTableSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val linear = Linear[Float](2, 2)
    val mapTable = new MapTable[Float]().setName("mapTable")
    mapTable.add(linear)
    val input1 = Tensor[Float](2).apply1(_ => Random.nextFloat())
    val input2 = Tensor[Float](2).apply1(_ => Random.nextFloat())
    val input = T()
    input(1.0.toFloat) = input1
    input(2.0.toFloat) = input2
    runSerializationTest(mapTable, input)
  }
}
