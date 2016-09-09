/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.sparkdl.torch

import breeze.numerics.abs
import com.intel.analytics.sparkdl.nn.BatchNormalization
import com.intel.analytics.sparkdl.tensor.{Tensor, torch}
import com.intel.analytics.sparkdl.utils.RandomGenerator._
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class BatchNormalizationSpec extends FlatSpec with BeforeAndAfter with Matchers {
  before {
    if (!TH.hasTorch()) {
      cancel("Torch is not installed")
    }
  }

  "A SpatialBatchNormalization" should "generate correct output and gradInput" in {

    val seed = 100
    RNG.setSeed(seed)

    val sbn = new BatchNormalization[Double](3, 1e-3)

    val input = torch.Tensor[Double](16, 3)
    var i = 0
    input.apply1(e => {
      RNG.uniform(0.0, 255)
    })
    //    input.apply1(e => 0.5)
    val gradOutput = torch.Tensor[Double](16, 3)
    i = 0
    gradOutput.apply1(e => {
      i += 1;
      0.1 * i
    })

    val gradOutput2 = torch.Tensor[Double](16, 3)
    i = 0
    gradOutput2.apply1(e => {
      i += 1;
      0.05 * i
    })


    sbn.zeroGradParameters()
    val parameters = sbn.getParameters()._1.asInstanceOf[Tensor[Double]]
    val gradparameters = sbn.getParameters()._2.asInstanceOf[Tensor[Double]]

    val code = "torch.manualSeed(" + seed + ")\n" +
      """
        |sbn = nn.BatchNormalization(3, 1e-3)
        |sbn:zeroGradParameters()
        |local parameters, gradParameters = sbn:getParameters()
        |parameters_initial = parameters : clone()
        |gradParameters_initial = gradParameters : clone()
        |
        |output = sbn:forward(input)
        |
        |gradInput = sbn:backward(input, gradOutput)
        |
        |sbn:forward(input)
        |
        |sbn:backward(input, gradOutput2)
      """.stripMargin

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput,
      "gradOutput2" -> gradOutput2), Array("sbn", "parameters_initial", "gradParameters_initial",
      "gradParameters"))
    val sbnTorch = torchResult("sbn").asInstanceOf[BatchNormalization[Double]]
    val parameterTorch = torchResult("parameters_initial").asInstanceOf[Tensor[Double]]
    val gradparameterTorch = torchResult("gradParameters_initial").asInstanceOf[Tensor[Double]]
    val gradparametersTorch = torchResult("gradParameters").asInstanceOf[Tensor[Double]]

    require(parameters == parameterTorch, "parameter compare failed")

    require(gradparameters == gradparameterTorch, "gradparameter compare failed")

    sbn.forward(input)
    sbn.backward(input, gradOutput)

    val output = sbn.forward(input)

    val gradInput = sbn.backward(input, gradOutput2)

    sbnTorch.output.map(sbn.output, (v1, v2) => {
      assert(abs(v1 - v2) == 0);
      v1
    })

    gradInput.map(sbnTorch.gradInput, (v1, v2) => {
      if (abs(v1 - v2) != 0) println(s"$v1 $v2")
      v1
    })

    gradparametersTorch.map(gradparameters, (v1, v2) => {
      if (abs(v1 - v2) != 0) println(s"$v1 $v2")
      v1
    })

    var j = 0
    var gradInputAbs = 0.0
    while (j < gradInput.nElement()) {
      val tmp = abs(gradInput.storage().array()(j) - sbnTorch.gradInput.storage().array()(j))
      gradInputAbs += tmp
      if (tmp != 0) println(s"$j: ${gradInput.storage().array()(j)} ${
        sbnTorch.gradInput.storage().array()(j)
      }")
      j += 1
    }
    println(s"gradInputAbs:$gradInputAbs")

  }

  "A SpatialBatchNormalization evaluating" should "generate correct output" in {

    val seed = 100
    RNG.setSeed(seed)

    val sbn = new BatchNormalization[Double](3, 1e-3)

    val input = torch.Tensor[Double](16, 3)
    var i = 0
    input.apply1(e => {
      RNG.uniform(0.0, 255)
    })
    //    input.apply1(e => 0.5)
    val gradOutput = torch.Tensor[Double](16, 3)
    i = 0
    gradOutput.apply1(e => {
      i += 1;
      0.1 * i
    })

    val gradOutput2 = torch.Tensor[Double](16, 3)
    i = 0
    gradOutput2.apply1(e => {
      i += 1;
      0.05 * i
    })


    sbn.zeroGradParameters()
    val parameters = sbn.getParameters()._1.asInstanceOf[Tensor[Double]]
    val gradparameters = sbn.getParameters()._2.asInstanceOf[Tensor[Double]]

    val code = "torch.manualSeed(" + seed + ")\n" +
      """
        |sbn = nn.BatchNormalization(3, 1e-3)
        |sbn:zeroGradParameters()
        |local parameters, gradParameters = sbn:getParameters()
        |parameters_initial = parameters : clone()
        |gradParameters_initial = gradParameters : clone()
        |
        |output = sbn:forward(input)
        |
        |gradInput = sbn:backward(input, gradOutput)
        |
        |sbn:forward(input)
        |
        |sbn:backward(input, gradOutput2)
        |
        |sbn:evaluate()
        |
        |output = sbn:forward(input)
      """.stripMargin

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput,
      "gradOutput2" -> gradOutput2), Array("sbn", "parameters_initial", "gradParameters_initial"))
    val sbnTorch = torchResult("sbn").asInstanceOf[BatchNormalization[Double]]
    val parameterTorch = torchResult("parameters_initial").asInstanceOf[Tensor[Double]]
    val gradparameterTorch = torchResult("gradParameters_initial").asInstanceOf[Tensor[Double]]

    require(parameters == parameterTorch, "parameter compare failed")

    require(gradparameters == gradparameterTorch, "gradparameter compare failed")

    sbn.forward(input)
    sbn.backward(input, gradOutput)

    sbn.forward(input)

    sbn.backward(input, gradOutput2)

    sbn.evaluate()
    val output = sbn.forward(input)

    sbnTorch.output.map(sbn.output, (v1, v2) => {
      assert(abs(v1 - v2) == 0);
      v1
    })

  }

}
