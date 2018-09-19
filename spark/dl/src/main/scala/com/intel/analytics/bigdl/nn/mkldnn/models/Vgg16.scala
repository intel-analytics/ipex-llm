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

package com.intel.analytics.bigdl.nn.mkldnn.models

import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.nn.{ConstInitMethod, Xavier, Zeros}
import com.intel.analytics.bigdl.nn.mkldnn._

object Vgg_16 {
  def apply(batchSize: Int, classNum: Int, hasDropout: Boolean = true): Sequential = {
    val model = Sequential()
    model.add(Input(Array(batchSize, 3, 224, 224), Memory.Format.nchw))
    model.add(Conv(3, 64, 3, 3, 1, 1, 1, 1).setName("conv1_1"))
    model.add(ReLU().setName("relu1_1"))
    model.add(Conv(64, 64, 3, 3, 1, 1, 1, 1).setName("conv1_2"))
    model.add(ReLU().setName("relu1_2"))
    model.add(MaxPooling(2, 2, 2, 2).setName("pool1"))

    model.add(Conv(64, 128, 3, 3, 1, 1, 1, 1).setName("conv2_1"))
    model.add(ReLU().setName("relu2_1"))
    model.add(Conv(128, 128, 3, 3, 1, 1, 1, 1).setName("conv2_2"))
    model.add(ReLU().setName("relu2_2"))
    model.add(MaxPooling(2, 2, 2, 2).setName("pool2"))

    model.add(Conv(128, 256, 3, 3, 1, 1, 1, 1).setName("conv3_1"))
    model.add(ReLU().setName("relu3_1"))
    model.add(Conv(256, 256, 3, 3, 1, 1, 1, 1).setName("conv3_2"))
    model.add(ReLU().setName("relu3_2"))
    model.add(Conv(256, 256, 3, 3, 1, 1, 1, 1).setName("conv3_3"))
    model.add(ReLU().setName("relu3_3"))
    model.add(MaxPooling(2, 2, 2, 2).setName("pool3"))

    model.add(Conv(256, 512, 3, 3, 1, 1, 1, 1).setName("conv4_1"))
    model.add(ReLU().setName("relu4_1"))
    model.add(Conv(512, 512, 3, 3, 1, 1, 1, 1).setName("conv4_2"))
    model.add(ReLU().setName("relu4_2"))
    model.add(Conv(512, 512, 3, 3, 1, 1, 1, 1).setName("conv4_3"))
    model.add(ReLU().setName("relu4_3"))
    model.add(MaxPooling(2, 2, 2, 2).setName("pool4"))

    model.add(Conv(512, 512, 3, 3, 1, 1, 1, 1).setName("conv5_1"))
    model.add(ReLU().setName("relu5_1"))
    model.add(Conv(512, 512, 3, 3, 1, 1, 1, 1).setName("conv5_2"))
    model.add(ReLU().setName("relu5_2"))
    model.add(Conv(512, 512, 3, 3, 1, 1, 1, 1).setName("conv5_3"))
    model.add(ReLU().setName("relu5_3"))
    model.add(MaxPooling(2, 2, 2, 2).setName("pool5"))

    model.add(Linear(512 * 7 * 7, 4096).setInitMethod(Xavier, ConstInitMethod(0.1)).setName("fc6"))
    model.add(ReLU().setName("relu6"))
    if (hasDropout) model.add(Dropout(0.5).setName("drop6"))
    model.add(Linear(4096, 4096).setInitMethod(Xavier, ConstInitMethod(0.1)).setName("fc7"))
    model.add(ReLU().setName("relu7"))
    if (hasDropout) model.add(Dropout(0.5).setName("drop7"))
    model.add(Linear(4096, classNum).setInitMethod(Xavier, ConstInitMethod(0.1)).setName(("fc8")))
    model.add(ReorderMemory(HeapData(Array(batchSize, classNum), Memory.Format.nc)))

    model
  }

  private object Conv {
    def apply(
      nInputPlane: Int,
      nOutputPlane: Int,
      kernelW: Int,
      kernelH: Int,
      strideW: Int = 1,
      strideH: Int = 1,
      padW: Int = 0,
      padH: Int = 0,
      nGroup: Int = 1,
      propagateBack: Boolean = true): SpatialConvolution = {
      val conv = SpatialConvolution(nInputPlane, nOutputPlane, kernelW, kernelH,
        strideW, strideH, padW, padH, nGroup, propagateBack)
      conv.setInitMethod(Xavier.setVersion2(true), Zeros)
      conv
    }
  }
}
