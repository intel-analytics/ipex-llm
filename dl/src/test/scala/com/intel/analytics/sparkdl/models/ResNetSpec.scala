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
import com.intel.analytics.sparkdl.utils.Table
import com.intel.analytics.sparkdl.nn._
import com.intel.analytics.sparkdl.optim.SGD
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.tensor._
import com.intel.analytics.sparkdl.torch.TH
import com.intel.analytics.sparkdl.utils.RandomGenerator._
import com.intel.analytics.sparkdl.utils.T
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.collection.immutable
import scala.math._
import scala.util.Random
import scala.collection.mutable.Map
import scala.reflect.ClassTag

class ResNetSpec extends FlatSpec with BeforeAndAfter with Matchers {
  "ResNet double" should "generate correct output" in {
    System.setProperty("java.io.tmpdir", "/disk2/test");
    if (!TH.hasTorch()) {
      cancel("Torch is not installed")
    }

    Random.setSeed(1)
    val classNum: Int = 1000
    val input = Tensor[Double](8, 3, 224, 224).apply1(e => Random.nextDouble())
    val labels = Tensor[Double](8).apply1(e => Random.nextInt(classNum))

    val seed = 100
    RNG.setSeed(seed)
    val opt: Table = new Table()
    opt("shortcutType") = "B"
    opt("depth") = 18
    opt("imagenet") = "imagenet"
    val model = ResNet[Float](classNum, opt)
    model.zeroGradParameters()


    val code = "torch.manualSeed(" + seed + ")\n" +
      """
        local Convolution = nn.SpatialConvolution
        local Avg = nn.SpatialAveragePooling
        local ReLU = nn.ReLU
        local Max = nn.SpatialMaxPooling
        local SBatchNorm = nn.SpatialBatchNormalization

        local nClasses = 1000
        local depth = 50
        local shortcutType = 'B'
        local iChannels
        local function shortcut(nInputPlane, nOutputPlane, stride)
          local useConv = shortcutType == 'C' or
                  (shortcutType == 'B' and nInputPlane ~= nOutputPlane)
          if useConv then
                 -- 1x1 convolution
            return nn.Sequential()
                    :add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
                    :add(SBatchNorm(nOutputPlane))
            elseif nInputPlane ~= nOutputPlane then
                 -- Strided, zero-padded identity shortcut
              return nn.Sequential()
                    :add(nn.SpatialAveragePooling(1, 1, stride, stride))
                    :add(nn.Concat(2)
                       :add(nn.Identity())
                       :add(nn.MulConstant(0)))
            else
              return nn.Identity()
           end
        end

        local function basicblock(n, stride)
          local nInputPlane = iChannels
          iChannels = n

          local s = nn.Sequential()
          s:add(Convolution(nInputPlane,n,3,3,stride,stride,1,1))
          s:add(SBatchNorm(n))
          s:add(ReLU(true))
          s:add(Convolution(n,n,3,3,1,1,1,1))
          s:add(SBatchNorm(n))

          return nn.Sequential()
                 --:add(shortcut(nInputPlane, n, stride))
                 --:add(s)
                 :add(nn.ConcatTable()
                    :add(s)
                 --   :add(s))
                    :add(shortcut(nInputPlane, n, stride)))
                 :add(nn.CAddTable(true))
                 :add(ReLU(true))
        end

        local function bottleneck(n, stride)
          local nInputPlane = iChannels
          iChannels = n * 4

          local s = nn.Sequential()
          s:add(Convolution(nInputPlane,n,1,1,1,1,0,0))
          s:add(SBatchNorm(n))
          s:add(ReLU(true))
          s:add(Convolution(n,n,3,3,stride,stride,1,1))
          s:add(SBatchNorm(n))
          s:add(ReLU(true))
          s:add(Convolution(n,n*4,1,1,1,1,0,0))
          s:add(SBatchNorm(n * 4))

          return nn.Sequential()
                 :add(nn.ConcatTable()
                   :add(s)
                   :add(shortcut(nInputPlane, n * 4, stride)))
                 :add(nn.CAddTable(true))
                 :add(ReLU(true))
        end


        local function layer(block, features, count, stride)
          local s = nn.Sequential()
          for i=1,count do
            s:add(block(features, i == 1 and stride or 1))
          end
          return s
        end

        local model = nn.Sequential()


        local cfg = {
                 --[10]  = {{1, 1, 1, 1}, 512, basicblock},
                 [18]  = {{2, 2, 2, 2}, 512, basicblock},
                 [34]  = {{3, 4, 6, 3}, 512, basicblock},
                 [50]  = {{3, 4, 6, 3}, 2048, bottleneck},
                 [101] = {{3, 4, 23, 3}, 2048, bottleneck},
                 [152] = {{3, 8, 36, 3}, 2048, bottleneck},
              }

              assert(cfg[depth], 'Invalid depth: ' .. tostring(depth))
              local def, nFeatures, block = table.unpack(cfg[depth])
              iChannels = 64
              --print(' | ResNet-' .. depth .. ' ImageNet')


        -- The ResNet ImageNet model
        model:add(Convolution(3,64,7,7,2,2,3,3))
        model:add(SBatchNorm(64))
        model:add(ReLU(true))
        model:add(Max(3,3,2,2,1,1))
        model:add(layer(block, 64, def[1]))
        model:add(layer(block, 128, def[2], 2))
        model:add(layer(block, 256, def[3], 2))
        model:add(layer(block, 512, def[4], 2))
        model:add(Avg(7, 7, 1, 1))
        model:add(nn.View(nFeatures):setNumInputDims(3))
        model:add(nn.Linear(nFeatures, nClasses))
        --model:add(nn.LogSoftMax())

        local parameters, gradParameters = model:getParameters()
                --model:zeroGradParameters()
                parameters_initial = parameters : clone()
                gradParameters_initial = gradParameters : clone()

                --local criterion =  nn.ClassNLLCriterion()
                local criterion = nn.CrossEntropyCriterion()
                state = {
                  learningRate = 1e-2,
                  momentum = 0.9,
                  dampening = 0.0,
                  weightDecay = 5e-4
                }

         feval = function(x)
              model:forward(input)
              criterion:forward(model.output, labels)
              --model:zeroGradParameters()
              criterion:backward(model.output, labels)
              model:backward(input, criterion.gradInput)
              return criterion.output, gradParameters
           end

             for i = 1, 5, 1 do
              w, err = optim.sgd(feval, parameters, state)
              --print(err)
             end

                output=model.output
                gradOutput=criterion.gradInput
                err = criterion.output
                gradInput = model.gradInput

      """

//    TH.runNM(code, immutable.Map("input" -> input, "labels" -> labels), Array("output", "gradOutput", "err",
//        "parameters_initial", "gradParameters_initial", "gradInput", "model"))
//
//    val parameterTorch = TH.map("parameters_initial").asInstanceOf[Tensor[Double]]
//    val parameters = model.getParameters()._1.asInstanceOf[Tensor[Float]]

    /*for (i <- 0 until parameters.nElement()) {
      if (abs(parameters.storage().array()(i) - parameterTorch.storage().array()(i)) > 1e-8) {
        println(s"${parameters.storage().array()(i)} ${parameterTorch.storage().array()(i)}")
      }
    }*/

    shareGradInput(model)
    shareFInput(model)

    val (weights, grad) = model.getParameters()
    val criterion = new CrossEntropyCriterion[Float]()

    val state = T("learningRate" -> 1e-2, "momentum" -> 0.9, "weightDecay" -> 5e-4,
      "dampening" -> 0.0)
    val sgd = new SGD[Float]

    val floatInput = Tensor[Float](8, 3, 224, 224)
    val floatLabels = Tensor[Float](8)
    for (i <- 0 until floatInput.nElement()) {
      floatInput.storage().array()(i) = input.storage().array()(i).toFloat
    }
    for (i <- 0 until floatLabels.nElement()) {
      floatLabels.storage().array()(i) = labels.storage().array()(i).toFloat
    }

    model.zeroGradParameters()
    for (i <- 1 to 4) {
      val outputtest = model.forward(floatInput)
      val loss = criterion.forward(outputtest, floatLabels)
      val gradoutputtest = criterion.backward(outputtest, floatLabels)
      model.backward(floatInput, gradoutputtest)
      sgd.optimize(_ => (loss, grad), weights, state, state)
      println("scala loss: i = " + i)
      println(loss)
    }

//    val output = TH.map("output").asInstanceOf[Tensor[Double]]
//    val outputTest = model.forward(floatInput)
//
//    var abss = 0.0
//    for (i <- 0 until outputTest.nElement()) {
//      val tmp = abs(outputTest.storage().array()(i) - output.storage().array()(i))
//      abss += tmp
//    }
//    //assert(abss < 1e-2)
//    println(s"this should be small: outputAbs:$abss")
//
//    val errTest = criterion.forward(outputTest, floatLabels)
//    println(s"Test scala loss: $errTest")
//    val err = TH.map("err").asInstanceOf[Double]
//    println(s"Test torch loss: $errTest")
//    println(s"${abs(errTest - err)}")
//    //assert(abs(errTest - err) < 1.5e-6)
//
//    val gradOutputTest = criterion.backward(outputTest, floatLabels)
//    val gradOutput = TH.map("gradOutput").asInstanceOf[Tensor[Double]]
//    abss = 0.0
//    for (i <- 0 until gradOutputTest.nElement()) {
//      val tmp = abs(gradOutputTest.storage().array()(i) - gradOutput.storage().array()(i))
//      abss += tmp
//    }
//    //assert(abss == 0.0)
//    //assert(abss < 2e-6)
//    println(s"this should be small: gradOutputTestAbs:$abss")
//
//    val gradInput = model.backward(floatInput, gradOutputTest)
//    val gradInputTorch = TH.map("gradInput").asInstanceOf[Tensor[Double]]
//
//    abss = 0.0
//    for (i <- 0 until gradInputTorch.nElement()) {
//      val tmp = abs(gradInputTorch.storage().array()(i) - gradInput.storage().array()(i))
//      abss += tmp
//    }
//    //assert(abss < 2e-6)
//    println(s"this should be small: gradInputTestAbs:$abss")
//
//    println(s"compare output between Lua and Scala:")
//    abss = 0.0
//    for (i <- 0 until outputTest.nElement()) {
//      val tmp = abs(outputTest.storage().array()(i) - output.storage().array()(i))
//      abss += tmp
//      val thOut = output.storage().array()(i)
//      val scOut = outputTest.storage().array()(i)
//      println(s"Lua = $thOut , Scala = $scOut")
//    }

//    val (weights, grad) = model.getParameters()
//    val modelTorch = TH.map("model").asInstanceOf[Module[Double]]
//    val (weightsTorch, gradTorch) = modelTorch.getParameters()
//    sgd.optimize(_ => (errTest, grad), weights, state, state)
//    abss = 0.0
//    for (i <- 0 until weights.nElement()) {
//      val tmp = abs(weights.storage().array()(i) - weightsTorch.storage().array()(i))
//      abss += tmp
//    }
//    assert(abss < 2e-2)
  }
 /*
  "AlexNet" should "generate correct output" in {
    if (!TH.hasTorch()) {
      cancel("Torch is not installed")
    }

    Random.setSeed(1)
    val input = Tensor[Double](8, 3, 224, 224).apply1(e => Random.nextDouble())
    val labels = Tensor[Double](8).apply1(e => Random.nextInt(nClasses))

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
local classifier = nn.Sequential()
classifier:add(nn.View(256*6*6))
classifier:add(nn.Linear(256*6*6, 4096))
classifier:add(nn.ReLU())
classifier:add(nn.Linear(4096, 4096))
classifier:add(nn.ReLU())
classifier:add(nn.Linear(4096, nClasses))
classifier:add(nn.LogSoftMax())
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
for i = 1,1, 1 do
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

    val model = AlexNet_OWT[Double](1000, false)
    model.zeroGradParameters()
    val parameters = model.getParameters()._1.asInstanceOf[Tensor[Double]]
    val parameterTorch = TH.map("parameters_initial").asInstanceOf[Tensor[Double]]
    parameters should be(parameterTorch)

    val gradParameters = model.getParameters()._2.asInstanceOf[Tensor[Double]]
    val gradParameterTorch = TH.map("gradParameters_initial").asInstanceOf[Tensor[Double]]
    gradParameters should be(gradParameterTorch)

    val (weights, grad) = model.getParameters()
    val criterion = new ClassNLLCriterion[Double]()

    val state = T("learningRate" -> 1e-2, "momentum" -> 0.9, "weightDecay" -> 5e-4,
      "dampening" -> 0.0)
    val sgd = new SGD[Double]

    for (i <- 1 to 1) {
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
    outputTest should be(output)

    val errTest = criterion.forward(outputTest, labels)
    val err = TH.map("err").asInstanceOf[Double]
    errTest should be(err)

    val gradOutputTest = criterion.backward(outputTest, labels)
    val gradOutput = TH.map("gradOutput").asInstanceOf[Tensor[Double]]
    gradOutputTest should be(gradOutput)

    val gradInput = model.backward(input, gradOutputTest)
    val gradInputTorch = TH.map("gradInput").asInstanceOf[Tensor[Double]]
    gradInput should be(gradInputTorch)
  }
  */


  def shareFInput[@specialized(Float, Double) T: ClassTag](model: Module[T])
                                                             (implicit ev: TensorNumeric[T]): Unit = {
    model.mapModules(m => {
      m.fInput = Tensor[T](Storage(Array(ev.fromType[Int](1))))
    })
  }


  def shareGradInput[@specialized(Float, Double) T: ClassTag](model: Module[T])
    (implicit ev: TensorNumeric[T]): Unit = {
    def sharingKey(m: Module[T]) = m.getClass.getName

    val cache = Map[Any, Storage[T]]()
    model.mapModules(m => {
      val moduleType = m.getClass.getName
      if (!moduleType.equals("com.intel.analytics.sparkdl.nn.ConcatAddTable")) {
        val key = sharingKey(m)
        if (!cache.contains(key)){
          cache.put(key, Storage(Array(ev.fromType[Int](1))))
        }

        m.gradInput = Tensor[T](cache.get(key).get, 1, Array(0))
      }
    })

    for ((m, i) <- model
      .findModules("com.intel.analytics.sparkdl.nn.ConcatAddTable")
      .zipWithIndex){
      if (!cache.contains(i % 2)) {
        cache.put(i % 2, Storage(Array(ev.fromType[Int](1))))
      }
      m.gradInput = Tensor[T](cache.get(i % 2).get, 1, Array(0))
    }
  }
}
