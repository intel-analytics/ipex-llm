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

package com.intel.analytics.sparkdl.models

import com.intel.analytics.sparkdl.nn._
import com.intel.analytics.sparkdl.optim.SGD
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.tensor._
import com.intel.analytics.sparkdl.torch.TH
import com.intel.analytics.sparkdl.utils.RandomGenerator._
import com.intel.analytics.sparkdl.utils.{RandomGenerator, T}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.math._
import scala.reflect.ClassTag
import scala.util.Random

class AlexNetSpec extends FlatSpec with BeforeAndAfter with Matchers {
  "AlexNet float" should "generate correct output" in {
    if (!TH.hasTorch()) {
      cancel("Torch is not installed")
    }

    Random.setSeed(1)
    val input = torch.Tensor[Double](8, 3, 224, 224).apply1(e => Random.nextDouble())
    val labels = torch.Tensor[Double](8).apply1(e => Random.nextInt(100))

    val seed = 100
    RNG.setSeed(seed)
    val model = getModel[Float](1000)
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
classifier:add(nn.Threshold(0, 1e-6))
--classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(4096, 4096))
classifier:add(nn.Threshold(0, 1e-6))
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
      "parameters_initial", "gradParameters_initial", "gradInput", "model"))

    val parameterTorch = TH.map("parameters_initial").asInstanceOf[Tensor[Double]]
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

    val floatInput = torch.Tensor[Float](8, 3, 224, 224)
    val floatLabel = torch.Tensor[Float](8)

    for (i <- 0 until floatInput.nElement()) {
      floatInput.storage().array()(i) = input.storage().array()(i).toFloat
    }
    for (i <- 0 until floatLabel.nElement()) {
      floatLabel.storage().array()(i) = labels.storage().array()(i).toFloat
    }

    model.zeroGradParameters()
    val output = TH.map("output").asInstanceOf[Tensor[Double]]
    val outputTest = model.forward(floatInput)
    var abss = 0.0
    for (i <- 0 until outputTest.nElement()) {
      val tmp = abs(outputTest.storage().array()(i) - output.storage().array()(i))
      abss += tmp
    }
    assert(abss < 1e-2)
    println(s"outputAbs:$abss")

    val errTest = criterion.forward(outputTest, floatLabel)
    val err = TH.map("err").asInstanceOf[Double]
    println(s"${abs(errTest - err)}")
    assert(abs(errTest - err) < 1e-6)

    val gradOutputTest = criterion.backward(outputTest, floatLabel)
    val gradOutput = TH.map("gradOutput").asInstanceOf[Tensor[Double]]
    abss = 0.0
    for (i <- 0 until gradOutputTest.nElement()) {
      val tmp = abs(gradOutputTest.storage().array()(i) - gradOutput.storage().array()(i))
      abss += tmp
    }
    assert(abss == 0.0)
    println(s"gradOutputTestAbs:$abss")

    val gradInput = model.backward(floatInput, gradOutputTest)
    val gradInputTorch = TH.map("gradInput").asInstanceOf[Tensor[Double]]

    abss = 0.0
    for (i <- 0 until gradInputTorch.nElement()) {
      val tmp = abs(gradInputTorch.storage().array()(i) - gradInput.storage().array()(i))
      abss += tmp
    }
    println(s"gradInputTestAbs:$abss")

    val (weights, grad) = model.getParameters()
    val modelTorch = TH.map("model").asInstanceOf[Module[Double]]
    val (weightsTorch, gradTorch) = modelTorch.getParameters()
    sgd.optimize(_ => (errTest, grad), weights, state, state)
    abss = 0.0
    for (i <- 0 until weights.nElement()) {
      val tmp = abs(weights.storage().array()(i) - weightsTorch.storage().array()(i))
      abss += tmp
    }
    assert(abss < 2e-2)
    println(s"weightsAbs:$abss")
  }

  "AlexNet" should "generate correct output" in {
    if (!TH.hasTorch()) {
      cancel("Torch is not installed")
    }

    Random.setSeed(1)
    val input = torch.Tensor[Double](8, 3, 224, 224).apply1(e => Random.nextDouble())
    val labels = torch.Tensor[Double](8).apply1(e => Random.nextInt(100))

    val seed = 100
    RNG.setSeed(seed)

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
classifier:add(nn.Threshold(0, 1e-6))
--classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(4096, 4096))
classifier:add(nn.Threshold(0, 1e-6))
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
for i = 1,5,1 do
  optim.sgd(feval, parameters, state)
end
local output = model:forward(input)
local err = criterion:forward(output, labels)
local gradOutput = criterion:backward(output, labels)
--local stateDfdx = state.dfdx
gradInput = model:backward(input, gradOutput)
      """

    TH.runNM(code, Map("input" -> input, "labels" -> labels), Array("output", "gradOutput", "err",
      "parameters_initial", "gradParameters_initial", "gradInput", "model"))

    val model = getModel[Double](1000)
    model.zeroGradParameters()
    val parameters = model.getParameters()._1.asInstanceOf[Tensor[Double]]
    val parameterTorch = TH.map("parameters_initial").asInstanceOf[Tensor[Double]]
    require(parameters == parameterTorch, "parameter compare failed")

    val gradparameters = model.getParameters()._2.asInstanceOf[Tensor[Double]]
    val gradparameterTorch = TH.map("gradParameters_initial").asInstanceOf[Tensor[Double]]
    require(gradparameters == gradparameterTorch, "gradparameter compare failed")

    val (weights, grad) = model.getParameters()
    val criterion = new ClassNLLCriterion[Double]()

    val state = T("learningRate" -> 1e-2, "momentum" -> 0.9, "weightDecay" -> 5e-4,
      "dampening" -> 0.0)
    val sgd = new SGD[Double]

    for (i <- 1 to 5) {
      model.zeroGradParameters()
      val outputTest = model.forward(input)
      val loss = criterion.forward(outputTest, labels)
      val gradOutputTest = criterion.backward(outputTest, labels)
      model.backward(input, gradOutputTest)
      sgd.optimize(_ => (loss, grad), weights, state, state)
    }

    model.zeroGradParameters()
    val outputTest = model.forward(input)
    val output = TH.map("output").asInstanceOf[Tensor[Double]]
    var abss = 0.0
    outputTest.map(output, (v1, v2) => {
      if (abs(v1 - v2) != 0) abss += abs(v1 - v2)
      v1
    })
    assert(abss < 1e-13)
    println(s"outputAbs:$abss")

    val errTest = criterion.forward(outputTest, labels)
    val err = TH.map("err").asInstanceOf[Double]
    println(s"${abs(errTest - err)}")
    assert(abs(errTest - err) == 0)

    val gradOutputTest = criterion.backward(outputTest, labels)
    val gradOutput = TH.map("gradOutput").asInstanceOf[Tensor[Double]]
    abss = 0.0
    gradOutputTest.map(gradOutput, (v1, v2) => {
      if (abs(v1 - v2) != 0) abss += abs(v1 - v2)
      v1
    })
    assert(abss < 1e-13)
    println(s"gradOutputTestAbs:$abss")

    val gradInput = model.backward(input, gradOutputTest)
    val gradInputTorch = TH.map("gradInput").asInstanceOf[Tensor[Double]]
    abss = 0.0
    gradInput.map(gradInputTorch, (v1, v2) => {
      if (abs(v1 - v2) != 0) abss += abs(v1 - v2)
      v1
    })
    assert(abss < 1e-13)
    println(s"gradInputTestAbs:$abss")

  }

  // alexnet without dropout
  def getModel[T: ClassTag](classNum: Int)(implicit ev: TensorNumeric[T]): Module[T] = {
    val feature = new Sequential[T]
    feature.add(new SpatialConvolution[T](3, 64, 11, 11, 4, 4, 2, 2))
    feature.add(new ReLU(true))
    feature.add(new SpatialMaxPooling[T](3, 3, 2, 2))
    feature.add(new SpatialConvolution(64, 192, 5, 5, 1, 1, 2, 2))
    feature.add(new ReLU(true))
    feature.add(new SpatialMaxPooling[T](3, 3, 2, 2))
    feature.add(new SpatialConvolution[T](192, 384, 3, 3, 1, 1, 1, 1))
    feature.add(new ReLU(true))
    feature.add(new SpatialConvolution[T](384, 256, 3, 3, 1, 1, 1, 1))
    feature.add(new ReLU(true))
    feature.add(new SpatialConvolution[T](256, 256, 3, 3, 1, 1, 1, 1))
    feature.add(new ReLU(true))
    feature.add(new SpatialMaxPooling[T](3, 3, 2, 2))



    val classifier = new Sequential[T]
    classifier.add(new View[T](256 * 6 * 6))
    classifier.add(new Linear[T](256 * 6 * 6, 4096))
    classifier.add(new Threshold[T](0, 1e-6))
    classifier.add(new Linear[T](4096, 4096))
    classifier.add(new Threshold[T](0, 1e-6))
    classifier.add(new Linear[T](4096, classNum))
    classifier.add(new LogSoftMax[T])


    val model = new Sequential[T]
    model.add(feature).add(classifier)

    model
  }

}
