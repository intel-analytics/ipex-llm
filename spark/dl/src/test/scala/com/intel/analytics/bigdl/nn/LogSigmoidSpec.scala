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
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

@com.intel.analytics.bigdl.tags.Parallel
class LogSigmoidSpec extends FlatSpec with Matchers {
  "A LogSigmoid Module " should "generate correct output" in {
    val module = new LogSigmoid[Double]()
    val input = Tensor[Double](2)
    input(Array(1)) = 0.1274271844660194
    input(Array(2)) = 0.6225728155339806
    val expectedOutput = Tensor[Double](2)
    expectedOutput(Array(1)) = -0.6314619274871387
    expectedOutput(Array(2)) = -0.4295475734209622
    val output = module.forward(input)
    output should equal(expectedOutput)
  }

  "A LogSigmoid Module " should "generate correct output and grad" in {
    val module = new LogSigmoid[Double]()
    val input = Tensor[Double](3, 3)
    input(Array(1, 1)) = 0.33655226649716
    input(Array(1, 2)) = 0.77367000770755
    input(Array(1, 3)) = 0.031494265655056
    input(Array(2, 1)) = 0.11129087698646
    input(Array(2, 2)) = 0.14688249188475
    input(Array(2, 3)) = 0.49454387230799
    input(Array(3, 1)) = 0.45682632108219
    input(Array(3, 2)) = 0.85653987620026
    input(Array(3, 3)) = 0.42569971177727
    val gradOutput = Tensor[Double](3, 3)
    gradOutput(Array(1, 1)) = 0.56766371615231
    gradOutput(Array(1, 2)) = 0.55222836649045
    gradOutput(Array(1, 3)) = 0.47152533312328
    gradOutput(Array(2, 1)) = 0.27471435652114
    gradOutput(Array(2, 2)) = 0.65794085455127
    gradOutput(Array(2, 3)) = 0.6130160340108
    gradOutput(Array(3, 1)) = 0.054757355013862
    gradOutput(Array(3, 2)) = 0.93723741802387
    gradOutput(Array(3, 3)) = 0.45930492319167
    val expectedGrad = Tensor[Double](3, 3)
    expectedGrad(Array(1, 1)) = 0.23651550644275185
    expectedGrad(Array(1, 2)) = 0.17433062335998667
    expectedGrad(Array(1, 3)) = 0.232050387377785
    expectedGrad(Array(2, 1)) = 0.12972175703022804
    expectedGrad(Array(2, 2)) = 0.3048537722992378
    expectedGrad(Array(2, 3)) = 0.2322250224916943
    expectedGrad(Array(3, 1)) = 0.021231560882982305
    expectedGrad(Array(3, 2)) = 0.27935558213351497
    expectedGrad(Array(3, 3)) = 0.18149602459589909

    module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    gradInput should be(expectedGrad)
  }
}

class LogSigmoidSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val logSigmoid = LogSigmoid[Float]().setName("logSigmoid")
    val input = Tensor[Float](10).apply1(_ => Random.nextFloat())
    runSerializationTest(logSigmoid, input)
  }
}
