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

package com.intel.analytics.bigdl.integration.torch.models

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.example.loadmodel.AlexNet_OWT
import com.intel.analytics.bigdl.integration.torch.{TH, TorchSpec}
import com.intel.analytics.bigdl.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.SGD
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.T

import scala.math._
import scala.util.Random

class AlexNetSpec extends TorchSpec {
  "AlexNet float" should "generate correct output" in {
    torchCheck()

    Random.setSeed(1)
    val input = Tensor[Double](8, 3, 224, 224).apply1(e => Random.nextDouble())
    val labels = Tensor[Double](8).apply1(e => Random.nextInt(100))

    val seed = 100
    RNG.setSeed(seed)
    val model = AlexNet_OWT(1000, false, true)
    model.zeroGradParameters()


    val code = "torch.manualSeed(" + seed + ")\n" +
      """local nClasses = 1000
local feature = nn.Sequential()
feature:add(nn.SpatialConvolutionMM(3,64,11,11,4,4,2,2))       -- 224 -> 55
feature:add(nn.ReLU())
feature:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27
feature:add(nn.SpatialConvolutionMM(64,192,5,5,1,1,2,2))       --  27 -> 27
feature:add(nn.ReLU())
feature:add(nn.SpatialMaxPooling(3,3,2,2))                   --  27 ->  13
feature:add(nn.SpatialConvolutionMM(192,384,3,3,1,1,1,1))      --  13 ->  13
feature:add(nn.ReLU())
feature:add(nn.SpatialConvolutionMM(384,256,3,3,1,1,1,1))      --  13 ->  13
feature:add(nn.ReLU())
feature:add(nn.SpatialConvolutionMM(256,256,3,3,1,1,1,1))      --  13 ->  13
feature:add(nn.ReLU())
feature:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6

-- 1.3. Create Classifier (fully connected layers)
local classifier = nn.Sequential()
classifier:add(nn.View(256*6*6))
--classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(256*6*6, 4096))
classifier:add(nn.ReLU())
--classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(4096, 4096))
classifier:add(nn.ReLU())
classifier:add(nn.Linear(4096, nClasses))
classifier:add(nn.LogSoftMax())


-- 1.4. Combine 1.1 and 1.3 to produce final model
model = nn.Sequential():add(feature):add(classifier)

local parameters, gradParameters = model:getParameters()
model:zeroGradParameters()
parameters_initial = parameters : clone()
gradParameters_initial = gradParameters : clone()

local criterion =  nn.ClassNLLCriterion()

state = {
  learningRate = 1e-2,
  momentum = 0.9,
  dampening = 0.0,
  weightDecay = 5e-4
}

feval = function(x)
model:zeroGradParameters()
model_initial = model : clone()

local output1 = model:forward(input)
local err1 = criterion:forward(output1, labels)
local gradOutput1 = criterion:backward(output1, labels)
model:backward(input, gradOutput1)
return err1, gradParameters
end

for i = 1,1,1 do
  optim.sgd(feval, parameters, state)
end

output=model.output
err=criterion.output
gradOutput=criterion.gradInput
gradInput = model.gradInput
      """

    TH.runNM(code, Map("input" -> input, "labels" -> labels), Array("output", "gradOutput", "err",
      "parameters_initial", "gradParameters_initial", "gradInput", "model"), suffix)

    val parameterTorch = TH.map("parameters_initial", suffix).asInstanceOf[Tensor[Double]]
    val parameters = model.getParameters()._1.asInstanceOf[Tensor[Float]]

    for (i <- 0 until parameters.nElement()) {
      if (abs(parameters.storage().array()(i) - parameterTorch.storage().array()(i)) > 1e-8) {
        println(s"${parameters.storage().array()(i)} ${parameterTorch.storage().array()(i)}")
      }
    }

    val criterion = new ClassNLLCriterion[Float]()
    val state = T("learningRate" -> 1e-2, "momentum" -> 0.9, "weightDecay" -> 5e-4,
      "dampening" -> 0.0)
    val sgd = new SGD[Float]

    val floatInput = Tensor[Float](8, 3, 224, 224)
    val floatLabel = Tensor[Float](8)

    for (i <- 0 until floatInput.nElement()) {
      floatInput.storage().array()(i) = input.storage().array()(i).toFloat
    }
    for (i <- 0 until floatLabel.nElement()) {
      floatLabel.storage().array()(i) = labels.storage().array()(i).toFloat
    }

    model.zeroGradParameters()
    val output = TH.map("output", suffix).asInstanceOf[Tensor[Double]]
    val outputTest = model.forward(floatInput).toTensor
    var abss = 0.0
    for (i <- 0 until outputTest.nElement()) {
      val tmp = abs(outputTest.storage().array()(i) - output.storage().array()(i))
      abss += tmp
    }
    assert(abss < 1e-2)
    println(s"outputAbs:$abss")

    val errTest = criterion.forward(outputTest, floatLabel)
    val err = TH.map("err", suffix).asInstanceOf[Double]
    println(s"${abs(errTest - err)}")
    assert(abs(errTest - err) < 1e-6)

    val gradOutputTest = criterion.backward(outputTest, floatLabel).toTensor
    val gradOutput = TH.map("gradOutput", suffix).asInstanceOf[Tensor[Double]]
    abss = 0.0
    for (i <- 0 until gradOutputTest.nElement()) {
      val tmp = abs(gradOutputTest.storage().array()(i) - gradOutput.storage().array()(i))
      abss += tmp
    }
    assert(abss == 0.0)
    println(s"gradOutputTestAbs:$abss")

    val gradInput = model.backward(floatInput, gradOutputTest).toTensor[Float]
    val gradInputTorch = TH.map("gradInput", suffix).asInstanceOf[Tensor[Double]]

    abss = 0.0
    for (i <- 0 until gradInputTorch.nElement()) {
      val tmp = abs(gradInputTorch.storage().array()(i) - gradInput.storage().array()(i))
      abss += tmp
    }
    println(s"gradInputTestAbs:$abss")

    val (weights, grad) = model.getParameters()
    val modelTorch = TH.map("model", suffix).asInstanceOf[Module[Double]]
    val (weightsTorch, gradTorch) = modelTorch.getParameters()
    sgd.optimize(_ => (errTest, grad), weights, state, state)
    abss = 0.0
    for (i <- 0 until weights.nElement()) {
      val tmp = abs(weights.storage().array()(i) - weightsTorch.storage().array()(i))
      abss += tmp
    }
    assert(abss < 2e-2)
  }

  "AlexNet Float save to torch" should "generate correct output" in {
    torchCheck()

    Random.setSeed(1)
    val input = Tensor[Float](8, 3, 224, 224).apply1(e => Random.nextFloat())
    val labels = Tensor[Float](8).apply1(e => Random.nextInt(100))

    val seed = 100
    RNG.setSeed(seed)
    val model = AlexNet_OWT(1000, false, true)
    model.zeroGradParameters()


    val code = "torch.manualSeed(" + seed + ")\n" +
      """local nClasses = 1000
torch.setdefaulttensortype('torch.FloatTensor')

local parameters, gradParameters = model:getParameters()
model:zeroGradParameters()
parameters_initial = parameters : clone()
gradParameters_initial = gradParameters : clone()

local criterion =  nn.ClassNLLCriterion()

state = {
  learningRate = 1e-2,
  momentum = 0.9,
  dampening = 0.0,
  weightDecay = 5e-4
}
feval = function(x)
  model:zeroGradParameters()
  model_initial = model : clone()
  local output1 = model:forward(input)
  local err1 = criterion:forward(output1, labels)
  local gradOutput1 = criterion:backward(output1, labels)
  model:backward(input, gradOutput1)
  return err1, gradParameters
end

for i = 1,5,1 do
  optim.sgd(feval, parameters, state)
end

output=model.output
err=criterion.output
gradOutput=criterion.gradInput
gradInput = model.gradInput
      """

    TH.runNM(code, Map("model" -> model, "input" -> input, "labels" -> labels),
      Array("output", "gradOutput", "err",
        "parameters_initial", "gradParameters_initial", "gradInput", "model"), suffix)

    val parameterTorch = TH.map("parameters_initial", suffix).asInstanceOf[Tensor[Float]]
    val parameters = model.getParameters()._1.asInstanceOf[Tensor[Float]]

    parameterTorch should be (parameters)

    val criterion = new ClassNLLCriterion[Float]()
    val state = T("learningRate" -> 1e-2, "momentum" -> 0.9, "weightDecay" -> 5e-4,
      "dampening" -> 0.0)
    val sgd = new SGD[Float]
    val (weights, grad) = model.getParameters()

    for (i <- 1 to 4) {
      model.zeroGradParameters()
      val outputtest = model.forward(input).toTensor[Float]
      val loss = criterion.forward(outputtest, labels)
      val gradoutputtest = criterion.backward(outputtest, labels)
      model.backward(input, gradoutputtest)
      sgd.optimize(_ => (loss, grad), weights, state, state)
    }

    model.zeroGradParameters()
    val output = TH.map("output", suffix).asInstanceOf[Tensor[Float]]
    val outputTest = model.forward(input).toTensor
    var abss = 0.0
    for (i <- 0 until outputTest.nElement()) {
      val tmp = abs(outputTest.storage().array()(i) - output.storage().array()(i))
      abss += tmp
    }
    assert(abss < 1e-2)
    println(s"outputAbs:$abss")

    val errTest = criterion.forward(outputTest, labels)
    val err = TH.map("err", suffix).asInstanceOf[Double]
    println(s"err:${abs(errTest - err)}")
    assert(abs(errTest - err) < 1e-6)

    val gradOutputTest = criterion.backward(outputTest, labels).toTensor
    val gradOutput = TH.map("gradOutput", suffix).asInstanceOf[Tensor[Float]]
    abss = 0.0
    for (i <- 0 until gradOutputTest.nElement()) {
      val tmp = abs(gradOutputTest.storage().array()(i) - gradOutput.storage().array()(i))
      abss += tmp
    }
    assert(abss == 0.0)
    println(s"gradOutputTestAbs:$abss")

    val gradInput = model.backward(input, gradOutputTest).toTensor[Float]
    val gradInputTorch = TH.map("gradInput", suffix).asInstanceOf[Tensor[Float]]

    abss = 0.0
    for (i <- 0 until gradInputTorch.nElement()) {
      val tmp = abs(gradInputTorch.storage().array()(i) - gradInput.storage().array()(i))
      abss += tmp
    }
    println(s"gradInputTestAbs:$abss")

    val modelTorch = TH.map("model", suffix).asInstanceOf[Module[Float]]
    val (weightsTorch, gradTorch) = modelTorch.getParameters()
    sgd.optimize(_ => (errTest, grad), weights, state, state)
    abss = 0.0
    for (i <- 0 until weights.nElement()) {
      val tmp = abs(weights.storage().array()(i) - weightsTorch.storage().array()(i))
      abss += tmp
    }
    assert(abss < 2e-2)
  }



}
