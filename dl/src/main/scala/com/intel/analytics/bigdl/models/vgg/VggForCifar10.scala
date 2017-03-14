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
package com.intel.analytics.bigdl.models.vgg

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat

object VggForCifar10 {
  def apply(classNum: Int): Module[Float] = {
    val vggBnDo = Sequential[Float]()

    def convBNReLU(nInputPlane: Int, nOutPutPlane: Int)
    : Sequential[Float] = {
      vggBnDo.add(SpatialConvolution(nInputPlane, nOutPutPlane, 3, 3, 1, 1, 1, 1))
      vggBnDo.add(SpatialBatchNormalization(nOutPutPlane, 1e-3))
      vggBnDo.add(ReLU(true))
      vggBnDo
    }
    convBNReLU(3, 64).add(Dropout((0.3)))
    convBNReLU(64, 64)
    vggBnDo.add(SpatialMaxPooling(2, 2, 2, 2).ceil())

    convBNReLU(64, 128).add(Dropout(0.4))
    convBNReLU(128, 128)
    vggBnDo.add(SpatialMaxPooling(2, 2, 2, 2).ceil())

    convBNReLU(128, 256).add(Dropout(0.4))
    convBNReLU(256, 256).add(Dropout(0.4))
    convBNReLU(256, 256)
    vggBnDo.add(SpatialMaxPooling(2, 2, 2, 2).ceil())

    convBNReLU(256, 512).add(Dropout(0.4))
    convBNReLU(512, 512).add(Dropout(0.4))
    convBNReLU(512, 512)
    vggBnDo.add(SpatialMaxPooling(2, 2, 2, 2).ceil())

    convBNReLU(512, 512).add(Dropout(0.4))
    convBNReLU(512, 512).add(Dropout(0.4))
    convBNReLU(512, 512)
    vggBnDo.add(SpatialMaxPooling(2, 2, 2, 2).ceil())
    vggBnDo.add(View(512))

    val classifier = Sequential[Float]()
    classifier.add(Dropout(0.5))
    classifier.add(Linear(512, 512))
    classifier.add(BatchNormalization(512))
    classifier.add(ReLU(true))
    classifier.add(Dropout(0.5))
    classifier.add(Linear(512, classNum))
    classifier.add(LogSoftMax())
    vggBnDo.add(classifier)

    vggBnDo
  }
}

object Vgg_16 {
  def apply(classNum: Int): Module[Float] = {
    val model = Sequential()
    model.add(SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1))
    model.add(ReLU(true))
    model.add(SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1))
    model.add(ReLU(true))
    model.add(SpatialMaxPooling(2, 2, 2, 2))

    model.add(SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1))
    model.add(ReLU(true))
    model.add(SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1))
    model.add(ReLU(true))
    model.add(SpatialMaxPooling(2, 2, 2, 2))

    model.add(SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1))
    model.add(ReLU(true))
    model.add(SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
    model.add(ReLU(true))
    model.add(SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
    model.add(ReLU(true))
    model.add(SpatialMaxPooling(2, 2, 2, 2))

    model.add(SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1))
    model.add(ReLU(true))
    model.add(SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    model.add(ReLU(true))
    model.add(SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    model.add(ReLU(true))
    model.add(SpatialMaxPooling(2, 2, 2, 2))

    model.add(SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    model.add(ReLU(true))
    model.add(SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    model.add(ReLU(true))
    model.add(SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    model.add(ReLU(true))
    model.add(SpatialMaxPooling(2, 2, 2, 2))

    model.add(View(512 * 7 * 7))
    model.add(Linear(512 * 7 * 7, 4096))
    model.add(Threshold(0, 1e-6))
    model.add(Dropout(0.5))
    model.add(Linear(4096, 4096))
    model.add(Threshold(0, 1e-6))
    model.add(Dropout(0.5))
    model.add(Linear(4096, classNum))
    model.add(LogSoftMax())

    model
  }
}

object Vgg_19 {
  def apply(classNum: Int): Module[Float] = {
    val model = Sequential()
    model.add(SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1))
    model.add(ReLU(true))
    model.add(SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1))
    model.add(ReLU(true))
    model.add(SpatialMaxPooling(2, 2, 2, 2))

    model.add(SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1))
    model.add(ReLU(true))
    model.add(SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1))
    model.add(ReLU(true))
    model.add(SpatialMaxPooling(2, 2, 2, 2))

    model.add(SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1))
    model.add(ReLU(true))
    model.add(SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
    model.add(ReLU(true))
    model.add(SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
    model.add(ReLU(true))
    model.add(SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
    model.add(ReLU(true))
    model.add(SpatialMaxPooling(2, 2, 2, 2))

    model.add(SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1))
    model.add(ReLU(true))
    model.add(SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    model.add(ReLU(true))
    model.add(SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    model.add(ReLU(true))
    model.add(SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    model.add(ReLU(true))
    model.add(SpatialMaxPooling(2, 2, 2, 2))

    model.add(SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    model.add(ReLU(true))
    model.add(SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    model.add(ReLU(true))
    model.add(SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    model.add(ReLU(true))
    model.add(SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    model.add(ReLU(true))
    model.add(SpatialMaxPooling(2, 2, 2, 2))

    model.add(View(512 * 7 * 7))
    model.add(Linear(512 * 7 * 7, 4096))
    model.add(Threshold(0, 1e-6))
    model.add(Dropout(0.5))
    model.add(Linear(4096, 4096))
    model.add(Threshold(0, 1e-6))
    model.add(Dropout(0.5))
    model.add(Linear(4096, classNum))
    model.add(LogSoftMax())

    model
  }
}
