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
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

import scala.math.abs
import scala.util.Random

@com.intel.analytics.bigdl.tags.Parallel
class ThresholdSpec extends FlatSpec {
  "A Threshold Module " should "generate correct output and grad" in {
    val module = new Threshold[Double](1, 0.8)
    val input = Tensor[Double](2, 2, 2)
    input(Array(1, 1, 1)) = -0.89699813351035
    input(Array(1, 1, 2)) = 1.8529373928905
    input(Array(1, 2, 1)) = 1.8799053365365
    input(Array(1, 2, 2)) = 0.076761466450989
    input(Array(2, 1, 1)) = 1.8863626234233
    input(Array(2, 1, 2)) = 0.73405137099326
    input(Array(2, 2, 1)) = 1.3404842875898
    input(Array(2, 2, 2)) = -0.64910735283047
    val gradOutput = Tensor[Double](2, 2, 2)
    gradOutput(Array(1, 1, 1)) = 0.31924905977212
    gradOutput(Array(1, 1, 2)) = 0.22160539613105
    gradOutput(Array(1, 2, 1)) = 0.19705923949368
    gradOutput(Array(1, 2, 2)) = 0.386440459406
    gradOutput(Array(2, 1, 1)) = 0.12920403806493
    gradOutput(Array(2, 1, 2)) = 0.7669838971924
    gradOutput(Array(2, 2, 1)) = 0.10939974407665
    gradOutput(Array(2, 2, 2)) = 0.70845287665725
    val expectedOutput = Tensor[Double](2, 2, 2)
    expectedOutput(Array(1, 1, 1)) = 0.8
    expectedOutput(Array(1, 1, 2)) = 1.8529373928905
    expectedOutput(Array(1, 2, 1)) = 1.8799053365365
    expectedOutput(Array(1, 2, 2)) = 0.8
    expectedOutput(Array(2, 1, 1)) = 1.8863626234233
    expectedOutput(Array(2, 1, 2)) = 0.8
    expectedOutput(Array(2, 2, 1)) = 1.3404842875898
    expectedOutput(Array(2, 2, 2)) = 0.8
    val expectedGrad = Tensor[Double](2, 2, 2)
    expectedGrad(Array(1, 1, 1)) = 0
    expectedGrad(Array(1, 1, 2)) = 0.22160539613105
    expectedGrad(Array(1, 2, 1)) = 0.19705923949368
    expectedGrad(Array(1, 2, 2)) = 0
    expectedGrad(Array(2, 1, 1)) = 0.12920403806493
    expectedGrad(Array(2, 1, 2)) = 0
    expectedGrad(Array(2, 2, 1)) = 0.10939974407665
    expectedGrad(Array(2, 2, 2)) = 0
    val inputOrg = input.clone()
    val gradOutputOrg = gradOutput.clone()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    expectedOutput.map(output, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
    expectedGrad.map(gradInput, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
    assert(input == inputOrg)
    assert(gradOutput == gradOutputOrg)
  }

  "A Threshold Module with inPlace = true" should "generate correct output and grad" in {
    val module = new Threshold[Double](1, 0.8, true)
    val input = Tensor[Double](2, 2, 2)
    input(Array(1, 1, 1)) = -0.89699813351035
    input(Array(1, 1, 2)) = 1.8529373928905
    input(Array(1, 2, 1)) = 1.8799053365365
    input(Array(1, 2, 2)) = 0.076761466450989
    input(Array(2, 1, 1)) = 1.8863626234233
    input(Array(2, 1, 2)) = 0.73405137099326
    input(Array(2, 2, 1)) = 1.3404842875898
    input(Array(2, 2, 2)) = -0.64910735283047
    val gradOutput = Tensor[Double](2, 2, 2)
    gradOutput(Array(1, 1, 1)) = 0.31924905977212
    gradOutput(Array(1, 1, 2)) = 0.22160539613105
    gradOutput(Array(1, 2, 1)) = 0.19705923949368
    gradOutput(Array(1, 2, 2)) = 0.386440459406
    gradOutput(Array(2, 1, 1)) = 0.12920403806493
    gradOutput(Array(2, 1, 2)) = 0.7669838971924
    gradOutput(Array(2, 2, 1)) = 0.10939974407665
    gradOutput(Array(2, 2, 2)) = 0.70845287665725
    val expectedOutput = Tensor[Double](2, 2, 2)
    expectedOutput(Array(1, 1, 1)) = 0.8
    expectedOutput(Array(1, 1, 2)) = 1.8529373928905
    expectedOutput(Array(1, 2, 1)) = 1.8799053365365
    expectedOutput(Array(1, 2, 2)) = 0.8
    expectedOutput(Array(2, 1, 1)) = 1.8863626234233
    expectedOutput(Array(2, 1, 2)) = 0.8
    expectedOutput(Array(2, 2, 1)) = 1.3404842875898
    expectedOutput(Array(2, 2, 2)) = 0.8
    val expectedGrad = Tensor[Double](2, 2, 2)
    expectedGrad(Array(1, 1, 1)) = 0
    expectedGrad(Array(1, 1, 2)) = 0.22160539613105
    expectedGrad(Array(1, 2, 1)) = 0.19705923949368
    expectedGrad(Array(1, 2, 2)) = 0
    expectedGrad(Array(2, 1, 1)) = 0.12920403806493
    expectedGrad(Array(2, 1, 2)) = 0
    expectedGrad(Array(2, 2, 1)) = 0.10939974407665
    expectedGrad(Array(2, 2, 2)) = 0
    val inputOrg = input.clone()
    val gradOutputOrg = gradOutput.clone()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    expectedOutput.map(output, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
    expectedGrad.map(gradInput, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })
    // InPlace test
    assert(output == input)
    assert(gradInput == gradOutput)
  }

  "A Threshold Module with inPlace = true clearstate" should "not clear input" in {
    val module = new Threshold[Double](1, 0.8, true)
    val input = Tensor[Double](2, 2, 2)
    module.forward(input)
    module.clearState()
    assert(input.nElement() == 8)
  }
}

class ThresholdSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val threshold = Threshold[Float](0.5).setName("threshold")
    val input = Tensor[Float](5, 5).apply1(_ => Random.nextFloat())
    runSerializationTest(threshold, input)
  }
}
