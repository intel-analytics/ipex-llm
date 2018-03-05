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
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.FlatSpec

import scala.util.Random

  /**
  * Created by yao on 9/21/16.
  */
@com.intel.analytics.bigdl.tags.Parallel
class MulConstantSpec extends FlatSpec {
  "MulConstant" should "generate correct output and grad" in {
    val input = Tensor[Double](2, 2, 2).randn()
    val scalar = 25.0
    val expectedOutput = input.clone().apply1(_ * scalar)
    val gradOutput = Tensor[Double](2, 2, 2).rand()
    val expectedGrad = gradOutput.clone().apply1(_ * scalar)

    val module = new MulConstant[Double](scalar)
    val output = module.forward(input)
    assert(expectedOutput equals output)

    val gradInput = module.backward(input, gradOutput)
    assert(gradInput equals expectedGrad )
  }

  "MulConstant with inPlace = true" should "generate correct output and grad" in {
    var input = Tensor[Double](2, 2, 2).randn()
    val scalar = 25.0
    val expectedOutput = input.clone().apply1(_ * scalar)
    val gradOutput = Tensor[Double](2, 2, 2).rand()
    val expectedGrad = gradOutput.clone().apply1(_ * scalar)

    // Test forward
    val module = new MulConstant[Double](scalar, true)
    val output = module.forward(input)
    assert(expectedOutput equals output)

    // Test backward
    input = Tensor[Double](2, 2, 2).randn()
    val expectedInput = input.clone().apply1(_ / scalar)
    val gradInput = module.backward(input, gradOutput)
    assert(gradInput equals expectedGrad)
    assert(input equals expectedInput)
  }
}

class MulConstantSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val mulConst = MulConstant[Float](1.0).setName("mulConst")
    val input = Tensor[Float](10, 10).apply1(_ => Random.nextFloat())
    runSerializationTest(mulConst, input)
  }
}
