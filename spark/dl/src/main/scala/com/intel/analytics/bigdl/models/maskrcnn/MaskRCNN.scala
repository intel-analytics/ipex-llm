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
import com.intel.analytics.bigdl.nn.{FPN, Sequential}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.T

class MaskRCNN(resNetOutChannels: Int,
               backboneOutChannels: Int)(implicit ev: TensorNumeric[Float])
  extends AbstractModule[Activity, Activity, Float] {

  private val backbone = buildBackbone(resNetOutChannels, backboneOutChannels)


  def buildBackbone(resNetOutChannels: Int, backboneOutChannels: Int): Module[Float] = {
    val body = ResNet(classNum = 1000, T("shortcutType" -> ShortcutType.B,
      "depth" -> 50, "optnet" -> false, "dataSet" -> DatasetType.ImageNet))

    val inChannels = Array(resNetOutChannels, resNetOutChannels*2,
      resNetOutChannels * 4, resNetOutChannels * 8)
    val fpn = FPN(inChannels, backboneOutChannels)

    // val t = fpn_module.LastLevelMaxPool()
    val model = Sequential[Float]().add(body).add(fpn)
    model
  }

  override def updateOutput(input: Activity): Activity = {
    output
  }

  override def updateGradInput(input: Activity, gradOutput: Activity): Activity = {
    gradInput
  }

}
