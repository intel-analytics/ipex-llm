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

import org.scalatest.FlatSpec
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.LayerException
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

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
      assert(output.nDimension() == 3)
      assert(output.size(1) == batchSize)
      assert(output.size(2) == 3)
      assert(output.size(3) == 2)
      assert(gradInput.isSameSizeAs(input))
      for (i <- 1 to batchSize) {
        for (j <- 0 to 5) {
          assert(input(Array(i, 1, j + 1)) == output(Array(i, j / 2 + 1, j % 2 + 1)))
          assert(gradInput(Array(i, 1, j + 1)) == gradOutput(Array(i, j / 2 + 1, j % 2 + 1)))
        }
      }
      assert(input == inputOrg)
      assert(gradOutput == gradOutputOrg)
    }

    intercept[LayerException] {
      module.forward(Tensor[Double](2, 2))
    }

    intercept[LayerException] {
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
    assert(output.nDimension() == 2)
    assert(output.size(1) == 3)
    assert(output.size(2) == 2)
    for (j <- 0 to 5) {
      assert(input(Array(j / 3 + 1, j % 3 + 1)) == output(Array(j / 2 + 1, j % 2 + 1)))
      assert(gradInput(Array(j / 3 + 1, j % 3 + 1)) == gradOutput(Array(j / 2 + 1, j % 2 + 1)))
    }
    assert(input == inputOrg)
    assert(gradOutput == gradOutputOrg)
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
    assert(output.nDimension() == 2)
    assert(output.size(1) == 3)
    assert(output.size(2) == 2)
    for (j <- 0 to 5) {
      assert(input(Array(1, j / 3 + 1, j % 3 + 1)) == output(Array(j / 2 + 1, j % 2 + 1)))
      assert(gradInput(Array(1, j / 3 + 1, j % 3 + 1)) == gradOutput(Array(j / 2 + 1, j % 2 + 1)))
    }
    assert(input == inputOrg)
    assert(gradOutput == gradOutputOrg)

    intercept[LayerException] {
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
      assert(output.nDimension() == 3)
      assert(output.size(1) == batchSize)
      assert(output.size(2) == 3)
      assert(output.size(3) == 2)
      assert(gradInput.isSameSizeAs(input))
      for (i <- 1 to batchSize) {
        for (j <- 0 to 5) {
          assert(input(Array(i, 1, j + 1)) == output(Array(i, j / 2 + 1, j % 2 + 1)))
          assert(gradInput(Array(i, 1, j + 1)) == gradOutput(Array(i, j / 2 + 1, j % 2 + 1)))
        }
      }
      assert(input == inputOrg)
      assert(gradOutput == gradOutputOrg)
    }

    intercept[LayerException] {
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
