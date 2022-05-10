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

package com.intel.analytics.bigdl.dllib.nn

import org.scalatest.FlatSpec
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.{LayerException, TestUtils}
import com.intel.analytics.bigdl.dllib.utils.serializer.ModuleSerializationTest

import scala.util.Random

@com.intel.analytics.bigdl.tags.Parallel
class ReshapeSpec extends FlatSpec {
  "A Reshape Module " should "generate correct output and grad" in {
    val module = new Reshape[Double](Array(3, 2))
    for (batchSize <- 1 to 4) {
      val input = Tensor[Double](batchSize, 1, 6)
      input.rand()
      val inputOrg = input.clone()
      val output = module.forward(input)
      val gradOutput = Tensor[Double](batchSize, 3, 2)
      gradOutput.rand()
      val gradOutputOrg = gradOutput.clone()
      val gradInput = module.backward(input, gradOutput)
      TestUtils.conditionFailTest(output.nDimension() == 3)
      TestUtils.conditionFailTest(output.size(1) == batchSize)
      TestUtils.conditionFailTest(output.size(2) == 3)
      TestUtils.conditionFailTest(output.size(3) == 2)
      TestUtils.conditionFailTest(gradInput.isSameSizeAs(input))
      for (i <- 1 to batchSize) {
        for (j <- 0 to 5) {
          TestUtils.conditionFailTest(
            input(Array(i, 1, j + 1)) == output(Array(i, j / 2 + 1, j % 2 + 1)))
          TestUtils.conditionFailTest(
            gradInput(Array(i, 1, j + 1)) == gradOutput(Array(i, j / 2 + 1, j % 2 + 1)))
        }
      }
      TestUtils.conditionFailTest(input == inputOrg)
      TestUtils.conditionFailTest(gradOutput == gradOutputOrg)
    }

    intercept[com.intel.analytics.bigdl.dllib.utils.InvalidOperationException] {
      module.forward(Tensor[Double](2, 2))
    }

    intercept[com.intel.analytics.bigdl.dllib.utils.InvalidOperationException] {
      module.forward(Tensor[Double](3, 2, 2))
    }
  }

  "A Reshape Module default batch" should "generate correct output and grad" in {
    val module = new Reshape[Double](Array(3, 2))
    val input = Tensor[Double](2, 3)
    input.rand()
    val inputOrg = input.clone()
    val output = module.forward(input)
    val gradOutput = Tensor[Double](3, 2)
    gradOutput.rand()
    val gradOutputOrg = gradOutput.clone()
    val gradInput = module.backward(input, gradOutput)
    TestUtils.conditionFailTest(output.nDimension() == 2)
    TestUtils.conditionFailTest(output.size(1) == 3)
    TestUtils.conditionFailTest(output.size(2) == 2)
    for (j <- 0 to 5) {
      TestUtils.conditionFailTest(
        input(Array(j / 3 + 1, j % 3 + 1)) == output(Array(j / 2 + 1, j % 2 + 1)))
      TestUtils.conditionFailTest(
        gradInput(Array(j / 3 + 1, j % 3 + 1)) == gradOutput(Array(j / 2 + 1, j % 2 + 1)))
    }
    TestUtils.conditionFailTest(input == inputOrg)
    TestUtils.conditionFailTest(gradOutput == gradOutputOrg)
  }

  "A Reshape Module disable batch" should "generate correct output and grad" in {
    val module = new Reshape[Double](Array(3, 2), Some(false))
    val input = Tensor[Double](1, 2, 3)
    input.rand()
    val inputOrg = input.clone()
    val output = module.forward(input)
    val gradOutput = Tensor[Double](3, 2)
    gradOutput.rand()
    val gradOutputOrg = gradOutput.clone()
    val gradInput = module.backward(input, gradOutput)
    TestUtils.conditionFailTest(output.nDimension() == 2)
    TestUtils.conditionFailTest(output.size(1) == 3)
    TestUtils.conditionFailTest(output.size(2) == 2)
    for (j <- 0 to 5) {
      TestUtils.conditionFailTest(
        input(Array(1, j / 3 + 1, j % 3 + 1)) == output(Array(j / 2 + 1, j % 2 + 1)))
      TestUtils.conditionFailTest(
        gradInput(Array(1, j / 3 + 1, j % 3 + 1)) == gradOutput(Array(j / 2 + 1, j % 2 + 1)))
    }
    TestUtils.conditionFailTest(input == inputOrg)
    TestUtils.conditionFailTest(gradOutput == gradOutputOrg)

    intercept[com.intel.analytics.bigdl.dllib.utils.UnKnownException] {
      module.forward(Tensor[Double](2, 3, 2))
    }
  }

  "A Reshape Module enable batch" should "generate correct output and grad" in {
    val module = new Reshape[Double](Array(3, 2), Some(true))
    for (batchSize <- 1 to 4) {
      val input = Tensor[Double](batchSize, 1, 6)
      input.rand()
      val inputOrg = input.clone()
      val output = module.forward(input)
      val gradOutput = Tensor[Double](batchSize, 3, 2)
      gradOutput.rand()
      val gradOutputOrg = gradOutput.clone()
      val gradInput = module.backward(input, gradOutput)
      TestUtils.conditionFailTest(output.nDimension() == 3)
      TestUtils.conditionFailTest(output.size(1) == batchSize)
      TestUtils.conditionFailTest(output.size(2) == 3)
      TestUtils.conditionFailTest(output.size(3) == 2)
      TestUtils.conditionFailTest(gradInput.isSameSizeAs(input))
      for (i <- 1 to batchSize) {
        for (j <- 0 to 5) {
          TestUtils.conditionFailTest(
            input(Array(i, 1, j + 1)) == output(Array(i, j / 2 + 1, j % 2 + 1)))
          TestUtils.conditionFailTest(
            gradInput(Array(i, 1, j + 1)) == gradOutput(Array(i, j / 2 + 1, j % 2 + 1)))
        }
      }
      TestUtils.conditionFailTest(input == inputOrg)
      TestUtils.conditionFailTest(gradOutput == gradOutputOrg)
    }

    intercept[com.intel.analytics.bigdl.dllib.utils.InvalidOperationException] {
      module.forward(Tensor[Double](3, 2))
    }
  }
}

class ReshapeSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val reshape = Reshape[Float](Array(1, 4, 5)).setName("reshape")
    val input = Tensor[Float](2, 2, 5).apply1( _ => Random.nextFloat())
    runSerializationTest(reshape, input)
  }
}
