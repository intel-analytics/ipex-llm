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
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.util.Random

@com.intel.analytics.bigdl.tags.Parallel
class LogSoftMaxSpec extends FlatSpec with Matchers with BeforeAndAfter {
  before {
    Engine.setNodeAndCore(1, 4)
  }


  "A LogSoftMax Module" should " be fast using MKL" in {
    val layer = LogSoftMax[Float]()
    layer.clearState()
    val batchSize = 20
    val input = Tensor[Float](batchSize, 10000)
    val gradOutput = Tensor[Float](batchSize, 10000)
    var startTime = System.nanoTime()
    var duration = (System.nanoTime() - startTime) / 1e9
    var sum = 0.0
    for (i <- 1 to 5) {
      layer.forward(input)
      layer.backward(input, gradOutput)
    }
    for (i <- 1 to 5) {
      startTime = System.nanoTime()
      layer.forward(input)
      layer.backward(input, gradOutput)
      duration = (System.nanoTime() - startTime) / 1e9
      println(s"speed: = ${duration} seconds")
      sum += duration
    }
    println(s"avg speed: = ${sum / 5}")
    layer.clearState()
  }

  "A LogSoftMax Module " should "generate correct output" in {
    val module = new LogSoftMax[Double]()
    val input = Tensor[Double](2)
    input(Array(1)) = 0.1274271844660194
    input(Array(2)) = 0.6225728155339806
    val expectedOutput = Tensor[Double](2)
    expectedOutput(Array(1)) = -0.9710581069556
    expectedOutput(Array(2)) = -0.47591247588764
    val output = module.forward(input)
    output should be(expectedOutput)
  }

  "A LogSoftMax Module " should "generate correct output and grad" in {
    val module = new LogSoftMax[Double]()
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
    val expectedOutput = Tensor[Double](3, 3)
    expectedOutput(Array(1, 1)) = -1.1894637490911
    expectedOutput(Array(1, 2)) = -0.75234600788072
    expectedOutput(Array(1, 3)) = -1.4945217499332
    expectedOutput(Array(2, 1)) = -1.2537001628522
    expectedOutput(Array(2, 2)) = -1.2181085479539
    expectedOutput(Array(2, 3)) = -0.87044716753068
    expectedOutput(Array(3, 1)) = -1.2414854064608
    expectedOutput(Array(3, 2)) = -0.84177185134272
    expectedOutput(Array(3, 3)) = -1.2726120157657
    val expectedGrad = Tensor[Double](3, 3)
    expectedGrad(Array(1, 1)) = 0.083261006513078
    expectedGrad(Array(1, 2)) = -0.19774248918721
    expectedGrad(Array(1, 3)) = 0.11448148267413
    expectedGrad(Array(2, 1)) = -0.166492308996
    expectedGrad(Array(2, 2)) = 0.20074813405794
    expectedGrad(Array(2, 3)) = -0.034255825061936
    expectedGrad(Array(3, 1)) = -0.36460248987794
    expectedGrad(Array(3, 2)) = 0.3118052217279
    expectedGrad(Array(3, 3)) = 0.052797268150042
    val inputOrg = input.clone()
    val gradOutputOrg = gradOutput.clone()
    val output = module.forward(input)
    val gradInput = module.backward(input, gradOutput)
    output should be(expectedOutput)
    gradInput should be(expectedGrad)
    input should be(inputOrg)
    gradOutput should be(gradOutputOrg)
  }

  "LogSoftMax float module" should "won't return Infinity when input is bigger than 89" in {
    val module = new LogSoftMax[Float]()
    Random.setSeed(100)
    val input = Tensor[Float](2, 5).apply1(e => Random.nextFloat() + 90)
    val output = module.forward(input).toTensor[Float]
    output.apply1(v => {v.isInfinity should be (false); v})
  }
}

class LogSoftMaxSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val logSoftMax = LogSoftMax[Float]().setName("logSoftMax")
    val input = Tensor[Float](10).apply1(_ => Random.nextFloat())
    runSerializationTest(logSoftMax, input)
  }
}
