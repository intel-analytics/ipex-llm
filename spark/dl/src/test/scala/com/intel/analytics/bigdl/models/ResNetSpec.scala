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

package com.intel.analytics.bigdl.models

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.models.resnet.{Convolution, ResNet}
import com.intel.analytics.bigdl.models.resnet.ResNet._
import com.intel.analytics.bigdl.nn.Graph.{apply => _, _}
import com.intel.analytics.bigdl.nn.abstractnn.Activity
import com.intel.analytics.bigdl.nn.{Graph, _}
import com.intel.analytics.bigdl.optim.SGD
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.torch.{TH, TorchSpec}
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import com.intel.analytics.bigdl.utils.{Engine, RandomGenerator, T, Table}
import org.apache.log4j.Logger
import com.intel.analytics.bigdl.numeric.NumericFloat

import scala.collection.{immutable, mutable}
import scala.math._
import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class ResNetSpec extends TorchSpec {

  "ResNet Float" should "generate correct output" in {
    // System.setProperty("java.io.tmpdir", "/disk2/test");
    Engine.setCoreNumber(4)
    torchCheck()

    for (i <- 1 to 1) {
      println(s"unitTest-${i}")
      unitTest(i, i + 100, 18, 4)
    }

  }


    def unitTest(inputSeed: Int, modelSeed: Int, depth: Int, batchSize: Int) {

      Random.setSeed(inputSeed)
      val classNum: Int = 1000
      val input = Tensor[Float](batchSize, 3, 224, 224).apply1( e => Random.nextFloat())
      val labels = Tensor[Float](batchSize).apply1(e => Random.nextInt(classNum))

    val seed = modelSeed
    RNG.setSeed(seed)
    val model = ResNet(classNum, T("shortcutType" -> ShortcutType.B,
      "depth" -> depth, "dataset" -> DatasetType.ImageNet))
    model.zeroGradParameters()


    val code =
      "torch.setdefaulttensortype('torch.FloatTensor')" +
      "torch.manualSeed(" + seed + ")\n" +
        "local depth = " + depth + "\n" +
      """
        local Convolution = nn.SpatialConvolution
        local Avg = nn.SpatialAveragePooling
        local ReLU = nn.ReLU
        local Max = nn.SpatialMaxPooling
        local SBatchNorm = nn.SpatialBatchNormalization
        local nClasses = 1000
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

             for i = 1, 1, 1 do
              w, err = optim.sgd(feval, parameters, state)
             end

                output=model.output
                gradOutput=criterion.gradInput
                err = criterion.output
                gradInput = model.gradInput

      """

    TH.runNM(code, immutable.Map("input" -> input, "labels" -> labels),
      Array("output", "gradOutput", "err", "parameters_initial",
        "gradParameters_initial", "gradInput", "model"))

    ResNet.shareGradInput(model)
    val parameterTorch = TH.map("parameters_initial").asInstanceOf[Tensor[Float]]
    val parameters = model.getParameters()._1

    for (i <- 0 until parameters.nElement()) {
      if (abs(parameters.storage().array()(i) - parameterTorch.storage().array()(i)) > 1e-8) {
        println(s"${parameters.storage().array()(i)} ${parameterTorch.storage().array()(i)}")
      }
    }

    val (weights, grad) = model.getParameters()
    val criterion = CrossEntropyCriterion[Float]()

    val state = T("learningRate" -> 1e-2, "momentum" -> 0.9, "weightDecay" -> 5e-4,
      "dampening" -> 0.0)
    val sgd = new SGD[Float]

    def feval(x: Tensor[Float]): (Float, Tensor[Float]) = {
      model.forward(input)
      criterion.forward(model.output.asInstanceOf[Tensor[Float]], labels)
      model.zeroGradParameters()
      val gradOutputTest = criterion.backward(model.output.asInstanceOf[Tensor[Float]], labels)
      model.backward(input, gradOutputTest)
      (criterion.output, grad)
    }
    for (i <- 1 to 1) {
      sgd.optimize(feval, weights, state)
    }

    val output = TH.map("output").asInstanceOf[Tensor[Float]]
    val outputTest = model.output.toTensor[Float]
    var abss = 0.0
    for (i <- 0 until outputTest.nElement()) {
      val tmp = abs(outputTest.storage().array()(i) - output.storage().array()(i))
      abss += tmp
    }
    println(s"outputAbs:$abss")
    assert(abss < 1e-2)


    val errTest = criterion.output
    val err = TH.map("err").asInstanceOf[Double]
    println(s"${abs(errTest - err)}")
    assert(abs(errTest - err) < 1.5e-6)

    val gradOutputTest = criterion.backward(outputTest, labels)
    val gradOutput = TH.map("gradOutput").asInstanceOf[Tensor[Float]]
    abss = 0.0
    for (i <- 0 until gradOutputTest.nElement()) {
      val tmp = abs(gradOutputTest.storage().array()(i) - gradOutput.storage().array()(i))
      abss += tmp
    }
    assert(abss < 2e-6)
    println(s"gradOutputTestAbs:$abss")

    val gradInput = model.gradInput.asInstanceOf[Tensor[Float]]
    val gradInputTorch = TH.map("gradInput").asInstanceOf[Tensor[Float]]

    abss = 0.0
    for (i <- 0 until gradInputTorch.nElement()) {
      val tmp = abs(gradInputTorch.storage().array()(i) - gradInput.storage().array()(i))
      abss += tmp
    }
    println(s"gradInputTestAbs:$abss")

  }

  "ResNet basicBlockFunc graph" should "be same with original one" in {
    val depth = 16
    ResNetTest.iChannels = 16

    RandomGenerator.RNG.setSeed(1000)
    val model = ResNetTest.basicBlock(16, 1)
    RandomGenerator.RNG.setSeed(1000)
    val input = Input()
    val output = ResNetTest.basicBlockFunc(16, 1, input)
    val graphModel = Graph(input, output)

    val inputData = Tensor(4, 16, 32, 32).rand()
    val gradients = Tensor(4, 16, 32, 32).rand()

    val output1 = model.forward(inputData)
    val output2 = graphModel.forward(inputData)

    output1 should be(output2)

    val gradInput1 = model.backward(inputData, gradients)
    val gradInput2 = graphModel.backward(inputData, gradients)

    gradInput1 should be(gradInput2)
  }

  "ResNet bottleneckFunc graph" should "be same with original one" in {
    val depth = 16
    ResNetTest.iChannels = 16

    RandomGenerator.RNG.setSeed(1000)
    val model = ResNetTest.bottleneck(16, 1)
    RandomGenerator.RNG.setSeed(1000)
    val input = Input()
    val output = ResNetTest.bottleneckFunc(16, 1, input)
    val graphModel = Graph(input, output)


    val inputData = Tensor(4, 16, 32, 32).rand()
    val gradients = Tensor(4, 64, 32, 32).rand()

    val output2 = graphModel.forward(inputData).toTensor[Float]
    val output1 = model.forward(inputData).toTensor[Float]

    output1.size() should be (output2.size())
    output1 should be(output2)

    val gradInput1 = model.backward(inputData, gradients)
    val gradInput2 = graphModel.backward(inputData, gradients)

    gradInput1 should be(gradInput2)
  }

  "ResNet-18 graph" should "be same with original one for ImageNet" in {
    val batchSize = 4
    val classNum = 1000
    val depth = 18
    val input = Tensor[Float](batchSize, 3, 224, 224).apply1( e => Random.nextFloat())
    val gradOutput = Tensor[Float](batchSize, 1000).apply1(e => Random.nextFloat())

    RNG.setSeed(1000)
    val model = ResNet(classNum, T("shortcutType" -> ShortcutType.B,
      "depth" -> depth, "dataset" -> DatasetType.ImageNet))
    RNG.setSeed(1000)
    val graphModel = ResNet.graph(classNum, T("shortcutType" -> ShortcutType.B,
      "depth" -> depth, "dataset" -> DatasetType.ImageNet))

    var output1: Tensor[Float] = null
    var output2: Tensor[Float] = null
    for (i <- 1 to 3) {
      output1 = model.forward(input).toTensor[Float]
      output2 = graphModel.forward(input).toTensor[Float]
    }
    output1 should be(output2)

    var gradInput1: Tensor[Float] = null
    var gradInput2: Tensor[Float] = null
    for (i <- 1 to 3) {
      gradInput1 = model.backward(input, gradOutput).toTensor[Float]
      gradInput2 = graphModel.backward(input, gradOutput).toTensor[Float]
    }
    gradInput1 should be(gradInput2)
  }


  "ResNet-50 graph" should "be same with original one for ImageNet" in {
    val batchSize = 4
    val classNum = 1000
    val depth = 50
    val input = Tensor[Float](batchSize, 3, 224, 224).apply1( e => Random.nextFloat())
    val gradOutput = Tensor[Float](batchSize, 1000).apply1(e => Random.nextFloat())

    RNG.setSeed(1000)
    val model = ResNet(classNum, T("shortcutType" -> ShortcutType.B,
      "depth" -> depth, "dataset" -> DatasetType.ImageNet))
    RNG.setSeed(1000)
    val graphModel = ResNet.graph(classNum, T("shortcutType" -> ShortcutType.B,
      "depth" -> depth, "dataset" -> DatasetType.ImageNet))

    var output1: Tensor[Float] = null
    var output2: Tensor[Float] = null
    for (i <- 1 to 3) {
      output1 = model.forward(input).toTensor[Float]
      output2 = graphModel.forward(input).toTensor[Float]
    }
    output1 should be(output2)

    var gradInput1: Tensor[Float] = null
    var gradInput2: Tensor[Float] = null
    for (i <- 1 to 3) {
      gradInput1 = model.backward(input, gradOutput).toTensor[Float]
      gradInput2 = graphModel.backward(input, gradOutput).toTensor[Float]
    }
    gradInput1 should be(gradInput2)
  }

  "ResNet graph" should "be same with original one for Cifar10" in {
    val batchSize = 4
    val classNum = 10
    val depth = 20
    val input = Tensor[Float](batchSize, 3, 32, 32).apply1( e => Random.nextFloat())
    val gradOutput = Tensor[Float](batchSize, classNum).apply1(e => Random.nextFloat())

    RNG.setSeed(1000)
    val model = ResNet(classNum, T("shortcutType" -> ShortcutType.B,
      "depth" -> depth, "dataset" -> DatasetType.CIFAR10))
    RNG.setSeed(1000)
    val graphModel = ResNet.graph(classNum, T("shortcutType" -> ShortcutType.B,
      "depth" -> depth, "dataset" -> DatasetType.CIFAR10))

    var output1: Tensor[Float] = null
    var output2: Tensor[Float] = null
    for (i <- 1 to 3) {
      output1 = model.forward(input).toTensor[Float]
      output2 = graphModel.forward(input).toTensor[Float]
    }
    output1 should be(output2)

    var gradInput1: Tensor[Float] = null
    var gradInput2: Tensor[Float] = null
    for (i <- 1 to 3) {
      gradInput1 = model.backward(input, gradOutput).toTensor[Float]
      gradInput2 = graphModel.backward(input, gradOutput).toTensor[Float]
    }
    gradInput1 should be(gradInput2)
  }

}

object ResNetTest {
  val logger = Logger.getLogger(getClass)
  val opt = T()
  var iChannels = 0
  val depth = opt.get("depth").getOrElse(18)
  val shortCutType = opt.get("shortcutType")
  val shortcutType = shortCutType.getOrElse(ShortcutType.B).asInstanceOf[ShortcutType]
  val dataSet = opt.get("dataset")
  val dataset = dataSet.getOrElse(DatasetType.CIFAR10).asInstanceOf[DatasetType]
  val optnet = opt.get("optnet").getOrElse(true)

  def shortcut(nInputPlane: Int, nOutputPlane: Int, stride: Int): Module[Float] = {
    val useConv = shortcutType == ShortcutType.C ||
      (shortcutType == ShortcutType.B && nInputPlane != nOutputPlane)

    if (useConv) {
      Sequential()
        .add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride, optnet = optnet))
        .add(SpatialBatchNormalization(nOutputPlane))
    } else if (nInputPlane != nOutputPlane) {
      Sequential()
        .add(SpatialAveragePooling(1, 1, stride, stride))
        .add(Concat(2)
          .add(Identity())
          .add(MulConstant(0f)))
    } else {
      Identity()
    }
  }


  def shortcutFunc(nInputPlane: Int, nOutputPlane: Int, stride: Int)(input: ModuleNode[Float])
  : ModuleNode[Float] = {
    val useConv = shortcutType == ShortcutType.C ||
      (shortcutType == ShortcutType.B && nInputPlane != nOutputPlane)

    if (useConv) {
      val conv1 = Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride,
        optnet = optnet).inputs(input)
      val bn1 = SpatialBatchNormalization(nOutputPlane).inputs(conv1)
      bn1
    } else if (nInputPlane != nOutputPlane) {
      val pool1 = SpatialAveragePooling(1, 1, stride, stride).inputs(input)
      val mul1 = MulConstant(0f).inputs(pool1)
      val concat = JoinTable(2, 0).inputs(pool1, mul1)
      concat
    } else {
      input
    }
  }

  def basicBlock(n: Int, stride: Int): Module[Float] = {
    val nInputPlane = iChannels
    iChannels = n

    val s = Sequential()
    s.add(Convolution(nInputPlane, n, 3, 3, stride, stride, 1, 1, optnet = optnet))
    s.add(SpatialBatchNormalization(n))
    s.add(ReLU(true))
    s.add(Convolution(n, n, 3, 3, 1, 1, 1, 1, optnet = optnet))
    s.add(SpatialBatchNormalization(n))

    Sequential()
      .add(ConcatTable()
        .add(s)
        .add(shortcut(nInputPlane, n, stride)))
      .add(CAddTable(true))
      .add(ReLU(true))
  }

  def basicBlockFunc(n: Int, stride: Int, input: ModuleNode[Float])
  : ModuleNode[Float] = {
    val nInputPlane = iChannels
    iChannels = n

    val conv1 = SpatialConvolution(nInputPlane, n, 3, 3, stride, stride, 1, 1).inputs(input)
    val bn1 = SpatialBatchNormalization(n).inputs(conv1)
    val relu1 = ReLU(true).inputs(bn1)
    val conv2 = SpatialConvolution(n, n, 3, 3, 1, 1, 1, 1).inputs(relu1)
    val bn2 = SpatialBatchNormalization(n).inputs(conv2)
    val shortcut = shortcutFunc(nInputPlane, n, stride)(input)
    val add = CAddTable(true).inputs(bn2, shortcut)
    val output = ReLU(true).inputs(add)
    output
  }

  def bottleneck(n: Int, stride: Int): Module[Float] = {
    val nInputPlane = 16 // iChannels
    iChannels = n * 4

    val s = Sequential()
    s.add(Convolution(nInputPlane, n, 1, 1, 1, 1, 0, 0, optnet = optnet))
      .add(SpatialBatchNormalization(n))
      .add(ReLU(true))
      .add(Convolution(n, n, 3, 3, stride, stride, 1, 1, optnet = optnet))
      .add(SpatialBatchNormalization(n))
      .add(ReLU(true))
      .add(Convolution(n, n*4, 1, 1, 1, 1, 0, 0, optnet = optnet))
      .add(SpatialBatchNormalization(n * 4))

    Sequential()
      .add(ConcatTable()
        .add(s)
        .add(shortcut(nInputPlane, n*4, stride)))
      .add(CAddTable(true))
      .add(ReLU(true))
  }

  def bottleneckFunc(n: Int, stride: Int, input: ModuleNode[Float]): ModuleNode[Float] = {
    val nInputPlane = 16 // iChannels
    iChannels = n * 4

    val conv1 = Convolution(nInputPlane, n, 1, 1, 1, 1, 0, 0, optnet = optnet).inputs(input)
    val bn1 = SpatialBatchNormalization(n).inputs(conv1)
    val relu = ReLU(true).inputs(bn1)
    val conv2 = Convolution(n, n, 3, 3, stride, stride, 1, 1, optnet = optnet).inputs(relu)
    val bn2 = SpatialBatchNormalization(n).inputs(conv2)
    val relu2 = ReLU(true).inputs(bn2)
    val conv3 = Convolution(n, n*4, 1, 1, 1, 1, 0, 0, optnet = optnet).inputs(relu2)
    val sbn = SpatialBatchNormalization(n * 4).inputs(conv3)

    val shortcut = shortcutFunc(nInputPlane, n * 4, stride)(input)
    val add = CAddTable(true).inputs(sbn, shortcut)
    val output = ReLU(true).inputs(add)
    output
  }
}
