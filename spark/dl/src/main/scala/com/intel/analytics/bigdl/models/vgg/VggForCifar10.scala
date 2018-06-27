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
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat

object VggForCifar10 {
  def apply(classNum: Int, hasDropout: Boolean = true): Module[Float] = {
    val vggBnDo = Sequential[Float]()

    def convBNReLU(nInputPlane: Int, nOutPutPlane: Int)
    : Sequential[Float] = {
      vggBnDo.add(SpatialConvolution(nInputPlane, nOutPutPlane, 3, 3, 1, 1, 1, 1))
      vggBnDo.add(SpatialBatchNormalization(nOutPutPlane, 1e-3))
      vggBnDo.add(ReLU(true))
      vggBnDo
    }
    convBNReLU(3, 64)
    if (hasDropout) vggBnDo.add(Dropout((0.3)))
    convBNReLU(64, 64)
    vggBnDo.add(SpatialMaxPooling(2, 2, 2, 2).ceil())

    convBNReLU(64, 128)
    if (hasDropout) vggBnDo.add(Dropout(0.4))
    convBNReLU(128, 128)
    vggBnDo.add(SpatialMaxPooling(2, 2, 2, 2).ceil())

    convBNReLU(128, 256)
    if (hasDropout) vggBnDo.add(Dropout(0.4))
    convBNReLU(256, 256)
    if (hasDropout) vggBnDo.add(Dropout(0.4))
    convBNReLU(256, 256)
    vggBnDo.add(SpatialMaxPooling(2, 2, 2, 2).ceil())

    convBNReLU(256, 512)
    if (hasDropout) vggBnDo.add(Dropout(0.4))
    convBNReLU(512, 512)
    if (hasDropout) vggBnDo.add(Dropout(0.4))
    convBNReLU(512, 512)
    vggBnDo.add(SpatialMaxPooling(2, 2, 2, 2).ceil())

    convBNReLU(512, 512)
    if (hasDropout) vggBnDo.add(Dropout(0.4))
    convBNReLU(512, 512)
    if (hasDropout) vggBnDo.add(Dropout(0.4))
    convBNReLU(512, 512)
    vggBnDo.add(SpatialMaxPooling(2, 2, 2, 2).ceil())
    vggBnDo.add(View(512))

    val classifier = Sequential[Float]()
    if (hasDropout) classifier.add(Dropout(0.5))
    classifier.add(Linear(512, 512))
    classifier.add(BatchNormalization(512))
    classifier.add(ReLU(true))
    if (hasDropout) classifier.add(Dropout(0.5))
    classifier.add(Linear(512, classNum))
    classifier.add(LogSoftMax())
    vggBnDo.add(classifier)

    vggBnDo
  }

  def graph(classNum: Int, hasDropout: Boolean = true)
  : Module[Float] = {
    val input = Input()
    def convBNReLU(nInputPlane: Int, nOutPutPlane: Int)(input: ModuleNode[Float])
    : ModuleNode[Float] = {
      val conv = SpatialConvolution(nInputPlane, nOutPutPlane, 3, 3, 1, 1, 1, 1).inputs(input)
      val bn = SpatialBatchNormalization(nOutPutPlane, 1e-3).inputs(conv)
      ReLU(true).inputs(bn)
    }
    val relu1 = convBNReLU(3, 64)(input)
    val drop1 = if (hasDropout) Dropout(0.3).inputs(relu1) else relu1
    val relu2 = convBNReLU(64, 64)(drop1)
    val pool1 = SpatialMaxPooling(2, 2, 2, 2).ceil().inputs(relu2)

    val relu3 = convBNReLU(64, 128)(pool1)
    val drop2 = if (hasDropout) Dropout(0.4).inputs(relu3) else relu3
    val relu4 = convBNReLU(128, 128)(drop2)
    val pool2 = SpatialMaxPooling(2, 2, 2, 2).ceil().inputs(relu4)

    val relu5 = convBNReLU(128, 256)(pool2)
    val drop3 = if (hasDropout) Dropout(0.4).inputs(relu5) else relu5
    val relu6 = convBNReLU(256, 256)(drop3)
    val drop4 = if (hasDropout) Dropout(0.4).inputs(relu6) else relu6
    val relu7 = convBNReLU(256, 256)(drop4)
    val pool3 = SpatialMaxPooling(2, 2, 2, 2).ceil().inputs(relu7)

    val relu8 = convBNReLU(256, 512)(pool3)
    val drop5 = if (hasDropout) Dropout(0.4).inputs(relu8) else relu8
    val relu9 = convBNReLU(512, 512)(drop5)
    val drop6 = if (hasDropout) Dropout(0.4).inputs(relu9) else relu9
    val relu10 = convBNReLU(512, 512)(drop6)
    val pool4 = SpatialMaxPooling(2, 2, 2, 2).ceil().inputs(relu10)

    val relu11 = convBNReLU(512, 512)(pool4)
    val drop7 = if (hasDropout) Dropout(0.4).inputs(relu11) else relu11
    val relu12 = convBNReLU(512, 512)(drop7)
    val drop8 = if (hasDropout) Dropout(0.4).inputs(relu12) else relu12
    val relu13 = convBNReLU(512, 512)(drop8)
    val pool5 = SpatialMaxPooling(2, 2, 2, 2).ceil().inputs(relu13)
    val view = View(512).inputs(pool5)

    val drop9 = if (hasDropout) Dropout(0.5).inputs(view) else view
    val linear1 = Linear(512, 512).inputs(drop9)
    val bn = BatchNormalization(512).inputs(linear1)
    val relu = ReLU(true).inputs(bn)
    val drop10 = if (hasDropout) Dropout(0.5).inputs(relu) else relu
    val linear2 = Linear(512, classNum).inputs(drop10)
    val output = LogSoftMax().inputs(linear2)
    Graph(input, output)
  }
}

object Vgg_16 {
  def apply(classNum: Int, hasDropout: Boolean = true): Module[Float] = {
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
    if (hasDropout) model.add(Dropout(0.5))
    model.add(Linear(4096, 4096))
    model.add(Threshold(0, 1e-6))
    if (hasDropout) model.add(Dropout(0.5))
    model.add(Linear(4096, classNum))
    model.add(LogSoftMax())

    model
  }

  def graph(classNum: Int, hasDropout: Boolean = true)
  : Module[Float] = {
    val conv1 = SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1).inputs()
    val relu1 = ReLU(true).inputs(conv1)
    val conv2 = SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1).inputs(relu1)
    val relu2 = ReLU(true).inputs(conv2)
    val pool1 = SpatialMaxPooling(2, 2, 2, 2).inputs(relu2)

    val conv3 = SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1).inputs(pool1)
    val relu3 = ReLU(true).inputs(conv3)
    val conv4 = SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1).inputs(relu3)
    val relu4 = ReLU(true).inputs(conv4)
    val pool2 = SpatialMaxPooling(2, 2, 2, 2).inputs(relu4)

    val conv5 = SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1).inputs(pool2)
    val relu5 = ReLU(true).inputs(conv5)
    val conv6 = SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1).inputs(relu5)
    val relu6 = ReLU(true).inputs(conv6)
    val conv7 = SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1).inputs(relu6)
    val relu7 = ReLU(true).inputs(conv7)
    val pool3 = SpatialMaxPooling(2, 2, 2, 2).inputs(relu7)

    val conv8 = SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1).inputs(pool3)
    val relu8 = ReLU(true).inputs(conv8)
    val conv9 = SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1).inputs(relu8)
    val relu9 = ReLU(true).inputs(conv9)
    val conv10 = SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1).inputs(relu9)
    val relu10 = ReLU(true).inputs(conv10)
    val pool4 = SpatialMaxPooling(2, 2, 2, 2).inputs(relu10)

    val conv11 = SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1).inputs(pool4)
    val relu11 = ReLU(true).inputs(conv11)
    val conv12 = SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1).inputs(relu11)
    val relu12 = ReLU(true).inputs(conv12)
    val conv13 = SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1).inputs(relu12)
    val relu13 = ReLU(true).inputs(conv13)
    val pool5 = SpatialMaxPooling(2, 2, 2, 2).inputs(relu13)

    val view1 = View(512 * 7 * 7).inputs(pool5)
    val linear1 = Linear(512 * 7 * 7, 4096).inputs(view1)
    val th1 = Threshold(0, 1e-6).inputs(linear1)
    val drop1 = if (hasDropout) Dropout(0.5).inputs(th1) else th1
    val linear2 = Linear(4096, 4096).inputs(drop1)
    val th2 = Threshold(0, 1e-6).inputs(linear2)
    val drop2 = if (hasDropout) Dropout(0.5).inputs(th2) else th2
    val linear3 = Linear(4096, classNum).inputs(drop2)
    val output = LogSoftMax().inputs(linear3)

    Graph(conv1, output)
  }
}

object Vgg_19 {
  def apply(classNum: Int, hasDropout: Boolean = true): Module[Float] = {
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
    if (hasDropout) model.add(Dropout(0.5))
    model.add(Linear(4096, 4096))
    model.add(Threshold(0, 1e-6))
    if (hasDropout) model.add(Dropout(0.5))
    model.add(Linear(4096, classNum))
    model.add(LogSoftMax())

    model
  }

  def graph(classNum: Int, hasDropout: Boolean = true)
  : Module[Float] = {
    val conv1 = SpatialConvolution(3, 64, 3, 3, 1, 1, 1, 1).inputs()
    val relu1 = ReLU(true).inputs(conv1)
    val conv2 = SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1).inputs(relu1)
    val relu2 = ReLU(true).inputs(conv2)
    val pool1 = SpatialMaxPooling(2, 2, 2, 2).inputs(relu2)

    val conv3 = SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1).inputs(pool1)
    val relu3 = ReLU(true).inputs(conv3)
    val conv4 = SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1).inputs(relu3)
    val relu4 = ReLU(true).inputs(conv4)
    val pool2 = SpatialMaxPooling(2, 2, 2, 2).inputs(relu4)

    val conv5 = SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1).inputs(pool2)
    val relu5 = ReLU(true).inputs(conv5)
    val conv6 = SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1).inputs(relu5)
    val relu6 = ReLU(true).inputs(conv6)
    val conv7 = SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1).inputs(relu6)
    val relu7 = ReLU(true).inputs(conv7)
    val conv8 = SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1).inputs(relu7)
    val relu8 = ReLU(true).inputs(conv8)
    val pool3 = SpatialMaxPooling(2, 2, 2, 2).inputs(relu8)

    val conv9 = SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1).inputs(pool3)
    val relu9 = ReLU(true).inputs(conv9)
    val conv10 = SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1).inputs(relu9)
    val relu10 = ReLU(true).inputs(conv10)
    val conv11 = SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1).inputs(relu10)
    val relu11 = ReLU(true).inputs(conv11)
    val conv12 = SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1).inputs(relu11)
    val relu12 = ReLU(true).inputs(conv12)
    val pool4 = SpatialMaxPooling(2, 2, 2, 2).inputs(relu12)

    val conv13 = SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1).inputs(pool4)
    val relu13 = ReLU(true).inputs(conv13)
    val conv14 = SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1).inputs(relu13)
    val relu14 = ReLU(true).inputs(conv14)
    val conv15 = SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1).inputs(relu14)
    val relu15 = ReLU(true).inputs(conv15)
    val conv16 = SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1).inputs(relu15)
    val relu16 = ReLU(true).inputs(conv16)
    val pool5 = SpatialMaxPooling(2, 2, 2, 2).inputs(relu16)

    val view1 = View(512 * 7 * 7).inputs(pool5)
    val linear1 = Linear(512 * 7 * 7, 4096).inputs(view1)
    val th1 = Threshold(0, 1e-6).inputs(linear1)
    val drop1 = if (hasDropout) Dropout(0.5).inputs(th1) else th1
    val linear2 = Linear(4096, 4096).inputs(drop1)
    val th2 = Threshold(0, 1e-6).inputs(linear2)
    val drop2 = if (hasDropout) Dropout(0.5).inputs(th2) else th2
    val linear3 = Linear(4096, classNum).inputs(drop2)
    val output = LogSoftMax().inputs(linear3)

    Graph(conv1, output)
  }
}
