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

package com.intel.analytics.bigdl.example.loadmodel

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat

object AlexNet_OWT {
  def apply(classNum: Int, hasDropout : Boolean = true, firstLayerPropagateBack :
  Boolean = false): Module[Float] = {
    val model = Sequential()
    model.add(SpatialConvolution(3, 64, 11, 11, 4, 4, 2, 2, 1, firstLayerPropagateBack)
      .setName("conv1"))
    model.add(ReLU(true).setName("relu1"))
    model.add(SpatialMaxPooling(3, 3, 2, 2).setName("pool1"))
    model.add(SpatialConvolution(64, 192, 5, 5, 1, 1, 2, 2).setName("conv2"))
    model.add(ReLU(true).setName("relu2"))
    model.add(SpatialMaxPooling(3, 3, 2, 2).setName("pool2"))
    model.add(SpatialConvolution(192, 384, 3, 3, 1, 1, 1, 1).setName("conv3"))
    model.add(ReLU(true).setName("relu3"))
    model.add(SpatialConvolution(384, 256, 3, 3, 1, 1, 1, 1).setName("conv4"))
    model.add(ReLU(true).setName("relu4"))
    model.add(SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1).setName("conv5"))
    model.add(ReLU(true).setName("relu5"))
    model.add(SpatialMaxPooling(3, 3, 2, 2).setName("poo5"))
    model.add(View(256 * 6 * 6))
    model.add(Linear(256 * 6 * 6, 4096).setName("fc6"))
    model.add(ReLU(true).setName("relu6"))
    if (hasDropout) model.add(Dropout(0.5).setName("drop6"))
    model.add(Linear(4096, 4096).setName("fc7"))
    model.add(ReLU(true).setName("relu7"))
    if (hasDropout) model.add(Dropout(0.5).setName("drop7"))
    model.add(Linear(4096, classNum).setName("fc8"))
    model.add(LogSoftMax())
    model
  }
}

object AlexNet {
  def apply(classNum: Int): Module[Float] = {
    val model = Sequential()
    model.add(SpatialConvolution(3, 96, 11, 11, 4, 4, 0, 0, 1, false).setName("conv1"))
    model.add(ReLU(true).setName("relu1"))
    model.add(SpatialCrossMapLRN(5, 0.0001, 0.75).setName("norm1"))
    model.add(SpatialMaxPooling(3, 3, 2, 2).setName("pool1"))
    model.add(SpatialConvolution(96, 256, 5, 5, 1, 1, 2, 2, 2).setName("conv2"))
    model.add(ReLU(true).setName("relu2"))
    model.add(SpatialCrossMapLRN(5, 0.0001, 0.75).setName("norm2"))
    model.add(SpatialMaxPooling(3, 3, 2, 2).setName("pool2"))
    model.add(SpatialConvolution(256, 384, 3, 3, 1, 1, 1, 1).setName("conv3"))
    model.add(ReLU(true).setName("relu3"))
    model.add(SpatialConvolution(384, 384, 3, 3, 1, 1, 1, 1, 2).setName("conv4"))
    model.add(ReLU(true).setName("relu4"))
    model.add(SpatialConvolution(384, 256, 3, 3, 1, 1, 1, 1, 2).setName("conv5"))
    model.add(ReLU(true).setName("relu5"))
    model.add(SpatialMaxPooling(3, 3, 2, 2).setName("pool5"))
    model.add(View(256 * 6 * 6))
    model.add(Linear(256 * 6 * 6, 4096).setName("fc6"))
    model.add(ReLU(true).setName("relu6"))
    model.add(Dropout(0.5).setName("drop6"))
    model.add(Linear(4096, 4096).setName("fc7"))
    model.add(ReLU(true).setName("relu7"))
    model.add(Dropout(0.5).setName("drop7"))
    model.add(Linear(4096, classNum).setName("fc8"))
    model.add(LogSoftMax().setName("loss"))
    model
  }
}
