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
import com.intel.analytics.bigdl.models.resnet.ResNet._
import com.intel.analytics.bigdl.models.resnet.{Convolution, ResNet}
import com.intel.analytics.bigdl.nn.Graph.{apply => _, _}
import com.intel.analytics.bigdl.nn.{Graph, _}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import com.intel.analytics.bigdl.utils.{RandomGenerator, T}
import org.apache.log4j.Logger
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class ResNetSpec extends FlatSpec with Matchers {

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
      "depth" -> depth, "dataSet" -> DatasetType.ImageNet))
    RNG.setSeed(1000)
    val graphModel = ResNet.graph(classNum, T("shortcutType" -> ShortcutType.B,
      "depth" -> depth, "dataset" -> DatasetType.ImageNet))
    var modelForwardTime = 0L
    var modelBackwardTime = 0L
    var graphForwardTime = 0L
    var graphBackwardTime = 0L

    var output1: Tensor[Float] = null
    var output2: Tensor[Float] = null
    var st = System.nanoTime()
    for (i <- 1 to 3) {
      output1 = model.forward(input).toTensor[Float]
    }
    modelForwardTime += System.nanoTime() - st
    st = System.nanoTime()
    for (i <- 1 to 3) {
      output2 = graphModel.forward(input).toTensor[Float]
    }
    graphForwardTime += System.nanoTime() - st
    output1 should be(output2)

    var gradInput1: Tensor[Float] = null
    var gradInput2: Tensor[Float] = null
    st = System.nanoTime()
    for (i <- 1 to 3) {
      gradInput1 = model.backward(input, gradOutput).toTensor[Float]
    }
    modelBackwardTime += System.nanoTime() - st
    st = System.nanoTime()
    for (i <- 1 to 3) {
      gradInput2 = graphModel.backward(input, gradOutput).toTensor[Float]
    }
    graphBackwardTime += System.nanoTime() - st
    gradInput1 should be(gradInput2)

    val (modelF, modelB) = model.getTimes().map(v => (v._2, v._3))
      .reduce((a, b) => (a._1 + b._1, a._2 + b._2))
    val (graphF, graphB) = graphModel.getTimes().map(v => (v._2, v._3))
      .reduce((a, b) => (a._1 + b._1, a._2 + b._2))
    modelForwardTime should be (modelF +- modelF / 100)
    modelBackwardTime should be (modelB +- modelB / 100)
    graphForwardTime should be (graphF +- graphF / 100)
    graphBackwardTime should be (graphB +- graphB / 100)
  }


  "ResNet-50 graph" should "be same with original one for ImageNet" in {
    val batchSize = 4
    val classNum = 1000
    val depth = 50
    val input = Tensor[Float](batchSize, 3, 224, 224).apply1( e => Random.nextFloat())
    val gradOutput = Tensor[Float](batchSize, 1000).apply1(e => Random.nextFloat())

    RNG.setSeed(1000)
    val model = ResNet(classNum, T("shortcutType" -> ShortcutType.B,
      "depth" -> depth, "dataSet" -> DatasetType.ImageNet))
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
      "depth" -> depth, "dataSet" -> DatasetType.CIFAR10))
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
  val dataSet = opt.get("dataSet")
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

    val conv1 = Convolution(nInputPlane, n, 3, 3, stride, stride, 1, 1).inputs(input)
    val bn1 = SpatialBatchNormalization(n).inputs(conv1)
    val relu1 = ReLU(true).inputs(bn1)
    val conv2 = Convolution(n, n, 3, 3, 1, 1, 1, 1).inputs(relu1)
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
