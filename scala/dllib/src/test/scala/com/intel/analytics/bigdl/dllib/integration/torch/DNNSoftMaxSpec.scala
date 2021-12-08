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

package com.intel.analytics.bigdl.dllib.integration.torch
//package com.intel.analytics.bigdl.dllib.nn.mkldnn

import com.intel.analytics.bigdl.dllib.integration.torch.{TH, TorchSpec}
import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.dllib.nn.mkldnn.{HeapData, MklDnnRuntime, SoftMax}
import com.intel.analytics.bigdl.dllib.nn.mkldnn.Phase.{InferencePhase, TrainingPhase}
import com.intel.analytics.bigdl.dllib.nn.mkldnn._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import org.apache.commons.lang3.SerializationUtils
import org.scalatest.{FlatSpec, Matchers}

@com.intel.analytics.bigdl.tags.Serial
class DNNSoftMaxSpec extends TorchSpec with Matchers {
  "SoftMax forward 4-D" should "work correctly" in {
    torchCheck()
    // we should test the cases which contain 1
    val tests = List(
      (2, 3, 4, 4),
      (1, 3, 4, 4),
      (1, 3, 1, 1),
      (1, 1, 1, 1),
      (1, 1, 3, 3),
      (2, 1, 3, 3),
      (2, 2, 1, 1))

    for ((batchSize, channel, height, width) <- tests) {
      val sm = SoftMax()
      sm.setRuntime(new MklDnnRuntime)
      sm.initFwdPrimitives(Array(HeapData(Array(batchSize, channel, height, width),
        Memory.Format.nchw)), TrainingPhase)
      sm.initBwdPrimitives(Array(HeapData(Array(batchSize, channel, height, width),
        Memory.Format.nchw)), TrainingPhase)

      val input = Tensor(batchSize, channel, height, width).rand()

      val output = sm.forward(input)

//      val nnSm = nn.SoftMax()
//      val nnOutput = nnSm.forward(input)
//
//      Tools.dense(output) should be (nnOutput)

      val gradOutput = Tensor[Float]().resizeAs(output.toTensor).rand(-10, 10)
      sm.backward(input, gradOutput)
//      nnSm.backward(input, gradOutput)

      val code = "module = nn.SoftMax()\n" +
        "output = module:forward(input)\n" +
        "gradInput = module:backward(input,gradOutput)"

      val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
        Array("output", "gradInput"))
      val luaOutput = torchResult("output").asInstanceOf[Tensor[Float]]
      val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Float]]

      Tools.dense(output) should be (luaOutput)
      Equivalent.nearequals(Tools.dense(sm.gradInput).toTensor, luaGradInput,
        epsilon = 1e-5) should be (true)
    }
  }

  "SoftMax forward 3-D" should "work correctly" in {
    torchCheck()
    // we should test the cases which contain 1
    val tests = List(
      (3, 4, 4),
      (3, 4, 4),
      (3, 1, 1),
      (1, 1, 1),
      (1, 3, 3),
      (1, 3, 3),
      (2, 1, 1))

    for ((i, j, k) <- tests) {
      val sm = SoftMax()
      sm.setRuntime(new MklDnnRuntime)
      sm.initFwdPrimitives(Array(HeapData(Array(i, j, k), Memory.Format.ncw)), TrainingPhase)
      sm.initBwdPrimitives(Array(HeapData(Array(i, j, k), Memory.Format.ncw)), TrainingPhase)

      val input = Tensor(i, j, k).rand()

      val output = sm.forward(input)

//      val nnSm = nn.SoftMax()
//      val nnOutput = nnSm.forward(input)

//      Tools.dense(output) should be (nnOutput)

      val gradOutput = Tensor[Float]().resizeAs(output.toTensor).rand(-10, 10)
      sm.backward(input, gradOutput)
//      nnSm.backward(input, gradOutput)

      val code = "module = nn.SoftMax()\n" +
        "output = module:forward(input)\n" +
        "gradInput = module:backward(input,gradOutput)"

      val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
        Array("output", "gradInput"))
      val luaOutput = torchResult("output").asInstanceOf[Tensor[Float]]
      val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Float]]

      Tools.dense(output) should be (luaOutput)
      Equivalent.nearequals(Tools.dense(sm.gradInput).toTensor, luaGradInput,
        epsilon = 1e-5) should be (true)
    }
  }

  "SoftMax backward" should "work correctly" in {
    torchCheck()
    val (batchSize, channel, height, width) = (2, 3, 4, 4)
    val sm = SoftMax()
    sm.setRuntime(new MklDnnRuntime)
    sm.initFwdPrimitives(Array(HeapData(Array(batchSize, channel, height, width),
      Memory.Format.nchw)), TrainingPhase)
    sm.initBwdPrimitives(Array(HeapData(Array(batchSize, channel, height, width),
      Memory.Format.nchw)), TrainingPhase)

//    val nnSm = nn.SoftMax()

    val input = Tensor(batchSize, channel, height, width).rand()
    val gradOutput = Tensor().resizeAs(input).rand(-10, 10)

    sm.forward(input)
//    nnSm.forward(input)

    sm.backward(input, gradOutput)
//    nnSm.backward(input, gradOutput)

    val code = "module = nn.SoftMax()\n" +
      "output = module:forward(input)\n" +
      "gradInput = module:backward(input,gradOutput)"

    val (luaTime, torchResult) = TH.run(code, Map("input" -> input, "gradOutput" -> gradOutput),
      Array("output", "gradInput"))
    val luaOutput = torchResult("output").asInstanceOf[Tensor[Float]]
    val luaGradInput = torchResult("gradInput").asInstanceOf[Tensor[Float]]

//    Equivalent.nearequals(Tools.dense(sm.output).toTensor, nnSm.output.toTensor,
//      epsilon = 1e-5) should be (true)
//    Equivalent.nearequals(Tools.dense(sm.gradInput).toTensor, nnSm.gradInput.toTensor,
//      epsilon = 1e-5) should be (true)
    Equivalent.nearequals(Tools.dense(sm.output).toTensor, luaOutput,
      epsilon = 1e-5) should be (true)
    Equivalent.nearequals(Tools.dense(sm.gradInput).toTensor, luaGradInput,
      epsilon = 1e-5) should be (true)
  }

  "SoftMax multi times forward" should "work correctly" in {
    torchCheck()
    val (batchSize, channel, height, width) = (2, 3, 4, 4)
    val sm = SoftMax()
    sm.setRuntime(new MklDnnRuntime)
    sm.initFwdPrimitives(Array(HeapData(Array(batchSize, channel, height, width),
      Memory.Format.nchw)), InferencePhase)
    sm.evaluate()

//    val nnSm = nn.SoftMax()

    (0 until 5).foreach { _ =>
      val input = Tensor(batchSize, channel, height, width).rand(-1, 1)
      sm.forward(input)
//      nnSm.forward(input)
      val code = "module = nn.SoftMax()\n" +
        "output = module:forward(input)"

      val (luaTime, torchResult) = TH.run(code, Map("input" -> input),
        Array("output"))
      val luaOutput = torchResult("output").asInstanceOf[Tensor[Float]]

//      Tools.dense(sm.output) should be (nnSm.output)
      Tools.dense(sm.output) should be (luaOutput)
    }
  }
}
