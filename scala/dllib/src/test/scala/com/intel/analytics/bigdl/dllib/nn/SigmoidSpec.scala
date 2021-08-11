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
class SigmoidSpec extends FlatSpec {
  "A Sigmoid Module " should "generate correct output and grad" in {
    val module = new Sigmoid[Double]
    val input = Tensor[Double](2, 2, 2)
    input(Array(1, 1, 1)) = 0.063364277360961
    input(Array(1, 1, 2)) = 0.90631252736785
    input(Array(1, 2, 1)) = 0.22275671223179
    input(Array(1, 2, 2)) = 0.37516756891273
    input(Array(2, 1, 1)) = 0.99284988618456
    input(Array(2, 1, 2)) = 0.97488326719031
    input(Array(2, 2, 1)) = 0.94414822547697
    input(Array(2, 2, 2)) = 0.68123375508003
    val gradOutput = Tensor[Double](2, 2, 2)
    gradOutput(Array(1, 1, 1)) = 0.38652365817688
    gradOutput(Array(1, 1, 2)) = 0.034144022269174
    gradOutput(Array(1, 2, 1)) = 0.68105488433503
    gradOutput(Array(1, 2, 2)) = 0.41517980070785
    gradOutput(Array(2, 1, 1)) = 0.91740695876069
    gradOutput(Array(2, 1, 2)) = 0.35317355184816
    gradOutput(Array(2, 2, 1)) = 0.24361599306576
    gradOutput(Array(2, 2, 2)) = 0.65869987895712
    val expectedOutput = Tensor[Double](2, 2, 2)
    expectedOutput(Array(1, 1, 1)) = 0.51583577126786
    expectedOutput(Array(1, 1, 2)) = 0.71224499952187
    expectedOutput(Array(1, 2, 1)) = 0.55546003768115
    expectedOutput(Array(1, 2, 2)) = 0.59270705262321
    expectedOutput(Array(2, 1, 1)) = 0.72965046058394
    expectedOutput(Array(2, 1, 2)) = 0.72609176575892
    expectedOutput(Array(2, 2, 1)) = 0.71993681755829
    expectedOutput(Array(2, 2, 2)) = 0.66401400310487
    val expectedGrad = Tensor[Double](2, 2, 2)
    expectedGrad(Array(1, 1, 1)) = 0.096533985368059
    expectedGrad(Array(1, 1, 2)) = 0.0069978877068295
    expectedGrad(Array(1, 2, 1)) = 0.16816892172375
    expectedGrad(Array(1, 2, 2)) = 0.1002266468557
    expectedGrad(Array(2, 1, 1)) = 0.18096830763559
    expectedGrad(Array(2, 1, 2)) = 0.070240043677749
    expectedGrad(Array(2, 2, 1)) = 0.049119755820981
    expectedGrad(Array(2, 2, 2)) = 0.14695555224503
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
}

class SigmoidSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val sigmoid = Sigmoid[Float]().setName("sigmoid")
    val input = Tensor[Float](10).apply1(_ => Random.nextFloat())
    runSerializationTest(sigmoid, input)
  }
}
