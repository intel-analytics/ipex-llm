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
import com.intel.analytics.bigdl.models.resnet.ResNet
import com.intel.analytics.bigdl.models.resnet.ResNet.{DatasetType, ShortcutType}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

class MaskRCNN(resNetOutChannels: Int,
               backboneOutChannels: Int)(implicit ev: TensorNumeric[Float])
  extends AbstractModule[Activity, Activity, Float] {

  val inChannels: Int = backboneOutChannels
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
  val boxScoreThresh: Float = 0.012f // 0.05f
  val boxNmsThread: Float = 0.5f
  val maxPerImage: Int = 100
  val outputSize: Int = 1024
  val numClasses: Int = 81
  val layers: Array[Int] = Array[Int](4, 4, 4, 4) // Array[Int](256, 256, 256, 256)
  val dilation: Int = 1
  val useGn: Boolean = false

  private val backbone = buildBackbone(resNetOutChannels, backboneOutChannels)
  private val rpn = RegionRroposal(inChannels,
    anchorSizes, aspectRatios, anchorStride, preNmsTopNTest, postNmsTopNTest,
    preNmsTopNTrain, postNmsTopNTrain, rpnNmsThread, minSize, fpnPostNmsTopN)

  private val boxHead = BoxHead(inChannels, boxResolution, scales, samplingRatio,
    boxScoreThresh, boxNmsThread, maxPerImage, outputSize, numClasses)
  private val maskHead = MaskHead(inChannels, maskResolution, scales, samplingRatio,
    layers, dilation, numClasses)

  // debug
  boxHead.getParameters()._1.fill(0.001f)
  maskHead.getParameters()._1.fill(0.001f)

  def buildBackbone(resNetOutChannels: Int, backboneOutChannels: Int): Module[Float] = {
    val body = ResNet(classNum = 1000, T("shortcutType" -> ShortcutType.B,
      "depth" -> 50, "optnet" -> false, "dataSet" -> DatasetType.ImageNet))

    val inChannels = Array(resNetOutChannels, resNetOutChannels*2,
      resNetOutChannels * 4, resNetOutChannels * 8)
    val fpn = FPN(inChannels, backboneOutChannels, topBlocks = 1)

    // val t = fpn_module.LastLevelMaxPool()
    val model = Sequential[Float]().add(body).add(fpn)
    model
  }

  def buildRoiHeads(features: Activity, proposals: Tensor[Float]) : Activity = {
    val boxOutput = this.boxHead.forward(T(features, proposals)).toTable
    val postProcessorBox = boxOutput[Table](2)
    val proposalsBox = postProcessorBox[Tensor[Float]](2)
    val labelsBox = postProcessorBox[Tensor[Float]](1)
    val maskOutput = this.maskHead.forward(T(features, proposalsBox, labelsBox))
    maskOutput
  }

  override def updateOutput(input: Activity): Activity = {
    val features = this.backbone.forward(input).toTensor[Float]
    val proposals = this.rpn.forward(T(input, features))
    output = buildRoiHeads(features, proposals)
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    gradInput
  }
}
