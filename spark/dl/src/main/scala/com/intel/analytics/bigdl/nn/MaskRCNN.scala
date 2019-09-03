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

package com.intel.analytics.bigdl.models.maskrcnn

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.models.resnet.{Convolution, ResNet, ResNetMask, Sbn}
import com.intel.analytics.bigdl.models.resnet.ResNet.{DatasetType, ShortcutType}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

class MaskRCNN(val inChannels: Int,
               val outChannels: Int)(implicit ev: TensorNumeric[Float])
  extends Container[Activity, Activity, Float] {
    val anchorSizes: Array[Float] = Array[Float](32, 64, 128, 256, 512)
    val aspectRatios: Array[Float] = Array[Float](0.5f, 1.0f, 2.0f)
    val anchorStride: Array[Float] = Array[Float](4, 8, 16, 32, 64)
    val preNmsTopNTest: Int = 1000
    val postNmsTopNTest: Int = 1000
    val preNmsTopNTrain: Int = 2000
    val postNmsTopNTrain: Int = 2000
    val rpnNmsThread: Float = 0.7f
    val minSize: Int = 0
    val fpnPostNmsTopN: Int = 2000

    val boxResolution: Int = 7
    val maskResolution: Int = 14
    val resolution: Int = 28
    val scales: Array[Float] = Array[Float](0.25f, 0.125f, 0.0625f, 0.03125f)
    val samplingRatio: Float = 2.0f
    val boxScoreThresh: Float = 0.05f
    val boxNmsThread: Float = 0.5f
    val maxPerImage: Int = 100
    val outputSize: Int = 1024
    val numClasses: Int = 81
    val layers: Array[Int] = Array[Int](256, 256, 256, 256)
    val dilation: Int = 1
    val useGn: Boolean = false

    private val ImageInfo : Tensor[Float] = Tensor[Float](2)
    private val backbone = buildBackbone(inChannels, outChannels)
    private val rpn = RegionRroposal(inChannels, anchorSizes, aspectRatios, anchorStride,
      preNmsTopNTest, postNmsTopNTest, preNmsTopNTrain, postNmsTopNTrain, rpnNmsThread,
      minSize, fpnPostNmsTopN)
    private val boxHead = BoxHead(inChannels, boxResolution, scales, samplingRatio,
      boxScoreThresh, boxNmsThread, maxPerImage, outputSize, numClasses)
    private val maskHead = MaskHead(inChannels, maskResolution, scales, samplingRatio,
      layers, dilation, numClasses)

    // add layer to modules
    modules.append(backbone.asInstanceOf[Module[Float]])
    modules.append(rpn.asInstanceOf[Module[Float]])
    modules.append(boxHead.asInstanceOf[Module[Float]])
    modules.append(maskHead.asInstanceOf[Module[Float]])

    def buildResNet50(): Module[Float] = {

    def shortcut(nInputPlane: Int, nOutputPlane: Int, stride: Int,
                 useConv: Boolean = false): Module[Float] = {
      if (useConv) {
        Sequential()
          .add(Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride))
          .add(Sbn(nOutputPlane))
      } else {
        Identity()
      }
    }

    def bottleneck(nInputPlane: Int, internalPlane: Int, nOutputPlane: Int,
                   stride: Int, useConv: Boolean = false): Module[Float] = {
      val s = Sequential()
        .add(Convolution(nInputPlane, internalPlane, 1, 1, stride, stride, 0, 0))
        .add(Sbn(internalPlane))
        .add(ReLU(true))
        .add(Convolution(internalPlane, internalPlane, 3, 3, 1, 1, 1, 1))
        .add(Sbn(internalPlane))
        .add(ReLU(true))
        .add(Convolution(internalPlane, nOutputPlane, 1, 1, 1, 1, 0, 0))
        .add(Sbn(nOutputPlane))

      val m = Sequential()
        .add(ConcatTable()
          .add(s)
          .add(shortcut(nInputPlane, nOutputPlane, stride, useConv)))
        .add(CAddTable(true))
        .add(ReLU(true))
      m
    }

    def layer(count: Int, nInputPlane: Int, nOutputPlane: Int,
              downOutputPlane: Int, stride: Int = 1): Module[Float] = {
      val s = Sequential()
        .add(bottleneck(nInputPlane, nOutputPlane, downOutputPlane, stride, true))
      for (i <- 2 to count) {
        s.add(bottleneck(downOutputPlane, nOutputPlane, downOutputPlane, 1, false))
      }
      s
    }

    val model = Sequential[Float]()
      .add(Convolution(3, 64, 7, 7, 2, 2, 3, 3, optnet = false, propagateBack = false))
      .add(Sbn(64))
      .add(ReLU(true))
      .add(SpatialMaxPooling(3, 3, 2, 2, 1, 1))

    val input = Input()
    val node0 = model.inputs(input)

    val startChannels = 64
    val node1 = layer(3, startChannels, 64, inChannels, 1).inputs(node0)
    val node2 = layer(4, inChannels, 128, inChannels * 2, 2).inputs(node1)
    val node3 = layer(6, inChannels * 2, 256, inChannels * 4, 2).inputs(node2)
    val node4 = layer(3, inChannels * 4, 512, inChannels * 8, 2).inputs(node3)

    Graph(input, Array(node1, node2, node3, node4))
  }

  private def buildBackbone(inChannels: Int, outChannels: Int): Module[Float] = {
    val resnet = buildResNet50()
    val inChannelList = Array(inChannels, inChannels*2, inChannels * 4, inChannels * 8)
    val fpn = FPN(inChannelList, outChannels, topBlocks = 1)
    val model = Sequential[Float]().add(resnet).add(fpn)
    model
  }

  override def updateOutput(input: Activity): Activity = {
    val inputWidth = input.toTensor[Float].size(3)
    val inputHeight = input.toTensor[Float].size(4)
    ImageInfo.setValue(1, inputWidth)
    ImageInfo.setValue(2, inputHeight)

    val features = this.backbone.forward(input)
    val proposals = this.rpn.forward(T(features, ImageInfo))
    val boxOutput = this.boxHead.forward(T(features, proposals)).toTable
    val postProcessorBox = boxOutput[Table](2)
    val proposalsBox = postProcessorBox[Tensor[Float]](2)
    val labelsBox = postProcessorBox[Tensor[Float]](1)
    val mask = this.maskHead.forward(T(features, proposalsBox, labelsBox))
    output = T(proposalsBox, labelsBox, mask)
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    throw new UnsupportedOperationException("MaskRCNN model only support inference now")
  }
}
