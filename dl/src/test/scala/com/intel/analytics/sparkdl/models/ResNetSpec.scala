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
import com.intel.analytics.sparkdl.models.ResNet.ShortcutType
import com.intel.analytics.sparkdl.nn._
import com.intel.analytics.sparkdl.optim.SGD
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.tensor._
import com.intel.analytics.sparkdl.torch.TH
import com.intel.analytics.sparkdl.utils.RandomGenerator._
import com.intel.analytics.sparkdl.utils.{T, Engine}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.collection.{immutable, mutable}
import scala.math._
import scala.reflect.ClassTag
import scala.util.Random

class ResNetSpec extends FlatSpec with BeforeAndAfter with Matchers {

  "ResNet double" should "generate correct output" in {
    //System.setProperty("java.io.tmpdir", "/disk2/test");
    Engine.setCoreNum(4)
    if (!TH.hasTorch()) {
      cancel("Torch is not installed")
    }

    Random.setSeed(1)
    val classNum: Int = 1000
    val input = Tensor[Double](4, 3, 224, 224).apply1(e => Random.nextDouble())
    val labels = Tensor[Double](4).apply1(e => Random.nextInt(classNum))

    val seed = 100
    RNG.setSeed(seed)
    val model = ResNet[Double](classNum, T("shortcutType" -> ShortcutType.B, "depth"->50))
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
              model:zeroGradParameters()
              criterion:backward(model.output, labels)
              model:backward(input, criterion.gradInput)
              return criterion.output, gradParameters
           end

             for i = 1, 10, 1 do
              w, err = optim.sgd(feval, parameters, state)
             end

                output=model.output
                gradOutput=criterion.gradInput
                err = criterion.output
                gradInput = model.gradInput

      """

    TH.runNM(code, immutable.Map("input" -> input, "labels" -> labels), Array("output", "gradOutput", "err",
      "parameters_initial", "gradParameters_initial", "gradInput", "model"))

    shareGradInput(model)

    val parameterTorch = TH.map("parameters_initial").asInstanceOf[Tensor[Double]]
    val parameters = model.getParameters()._1.asInstanceOf[Tensor[Double]]

    for (i <- 0 until parameters.nElement()) {
      if (abs(parameters.storage().array()(i) - parameterTorch.storage().array()(i)) > 1e-8) {
        println(s"${parameters.storage().array()(i)} ${parameterTorch.storage().array()(i)}")
      }
    }

    //val criterion = new ClassNLLCriterion[Double]()
    val (weights, grad) = model.getParameters()
    val criterion = new CrossEntropyCriterion[Double]()

    val state = T("learningRate" -> 1e-2, "momentum" -> 0.9, "weightDecay" -> 5e-4,
      "dampening" -> 0.0)
    val sgd = new SGD[Double]

    /* val floatInput = Tensor[Float](256, 3, 224, 224)
     val floatLabels = Tensor[Float](256)
     for (i <- 0 until floatInput.nElement()) {
       floatInput.storage().array()(i) = input.storage().array()(i).toFloat
     }
     for (i <- 0 until floatLabels.nElement()) {
       floatLabels.storage().array()(i) = labels.storage().array()(i).toFloat
     }*/

    def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
      model.forward(input)
      criterion.forward(model.output, labels)
      model.zeroGradParameters()
      criterion.backward(model.output, labels)
      model.backward(input, criterion.gradInput)
      (criterion.output, grad)
    }
    for (i <- 1 to 5) {
      sgd.optimize(feval, weights, state)
    }

    val output = TH.map("output").asInstanceOf[Tensor[Double]]
    val outputTest = model.output //model.forward(input)
    var abss = 0.0
    for (i <- 0 until outputTest.nElement()) {
      val tmp = abs(outputTest.storage().array()(i) - output.storage().array()(i))
      abss += tmp
    }
    println(s"outputAbs:$abss")
    assert(abss < 1e-2)


    val errTest = criterion.output //criterion.forward(outputTest, labels)
    val err = TH.map("err").asInstanceOf[Double]
    println(s"${abs(errTest - err)}")
    assert(abs(errTest - err) < 1.5e-6)

    val gradOutputTest = criterion.gradInput //criterion.backward(outputTest, labels)
    val gradOutput = TH.map("gradOutput").asInstanceOf[Tensor[Double]]
    abss = 0.0
    for (i <- 0 until gradOutputTest.nElement()) {
      val tmp = abs(gradOutputTest.storage().array()(i) - gradOutput.storage().array()(i))
      abss += tmp
    }
    assert(abss < 2e-6)
    println(s"gradOutputTestAbs:$abss")

    val gradInput = model.gradInput // model.backward(input, gradOutputTest)
    val gradInputTorch = TH.map("gradInput").asInstanceOf[Tensor[Double]]

    abss = 0.0
    for (i <- 0 until gradInputTorch.nElement()) {
      val tmp = abs(gradInputTorch.storage().array()(i) - gradInput.storage().array()(i))
      abss += tmp
    }
    println(s"gradInputTestAbs:$abss")

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

  def shareGradInput[@specialized(Float, Double) T: ClassTag](model: Module[T])
    (implicit ev: TensorNumeric[T]): Unit = {
    def sharingKey(m: Module[T]) = m.getClass.getName

    val cache = mutable.Map[Any, Storage[T]]()

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

    cache.put("gradWeightMM", Storage(Array(ev.fromType[Int](1))))
    cache.put("fInput", Storage(Array(ev.fromType[Int](1))))
    cache.put("fGradInput", Storage(Array(ev.fromType[Int](1))))
    for ((m, i) <- model
      .findModules("com.intel.analytics.sparkdl.nn.SpatialConvolution")
      .zipWithIndex){
      val tmpModel = m.asInstanceOf[SpatialConvolution[T]]
      tmpModel.gradWeightMM = Tensor[T](cache.get("gradWeightMM").get)
      tmpModel.fInput = Tensor[T](cache.get("fInput").get)
      tmpModel.fGradInput = Tensor[T](cache.get("fGradInput").get)
    }
  }

}
