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

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.models.inception.Inception_Layer_v1
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.{T, Table}

object DynamicTestModels {

  object Autoencoder {
    val rowN = 28
    val colN = 28
    val featureSize = rowN * colN

    def graph(classNum: Int): Module[Float] = {
      val input = Reshape(Array(featureSize)).inputs()
      val linear1 = Linear(featureSize, classNum).inputs(input)
      val relu = ReLU().inputs(linear1)
      val linear2 = Linear(classNum, featureSize).inputs(relu)
      val output = Sigmoid().inputs(linear2)
      Graph.dynamic(input, output)
    }
  }

  object Inception_v1_NoAuxClassifier {
    def graph(classNum: Int, hasDropout: Boolean = true): Module[Float] = {
      val input = SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3, 1, false)
        .setInitMethod(weightInitMethod = Xavier, Zeros).setName("conv1/7x7_s2").inputs()
      val conv1_relu = ReLU(true).setName("conv1/relu_7x7").inputs(input)
      val pool1_s2 = SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool1/3x3_s2").inputs(conv1_relu)
      val pool1_norm1 = SpatialCrossMapLRN(5, 0.0001, 0.75).setName("pool1/norm1").inputs(pool1_s2)
      val conv2 = SpatialConvolution(64, 64, 1, 1, 1, 1).setInitMethod(weightInitMethod = Xavier,
        Zeros).setName("conv2/3x3_reduce").inputs(pool1_norm1)
      val conv2_relu = ReLU(true).setName("conv2/relu_3x3_reduce").inputs(conv2)
      val conv2_3x3 = SpatialConvolution(64, 192, 3, 3, 1, 1, 1, 1)
        .setInitMethod(weightInitMethod = Xavier, Zeros).setName("conv2/3x3").inputs(conv2_relu)
      val conv2_relu_3x3 = ReLU(true).setName("conv2/relu_3x3").inputs(conv2_3x3)
      val conv2_norm2 = SpatialCrossMapLRN(5, 0.0001, 0.75)
        .setName("conv2/norm2").inputs(conv2_relu_3x3)
      val pool2_s2 = SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool2/3x3_s2")
        .inputs(conv2_norm2)
      val inception_3a = Inception_Layer_v1(pool2_s2, 192,
        T(T(64), T(96, 128), T(16, 32), T(32)), "inception_3a/")
      val inception_3b = Inception_Layer_v1(inception_3a, 256,
        T(T(128), T(128, 192), T(32, 96), T(64)), "inception_3b/")
      val pool3 = SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool3/3x3_s2").inputs(inception_3b)
      val inception_4a = Inception_Layer_v1(pool3, 480,
        T(T(192), T(96, 208), T(16, 48), T(64)), "inception_4a/")
      val inception_4b = Inception_Layer_v1(inception_4a, 512,
        T(T(160), T(112, 224), T(24, 64), T(64)), "inception_4b/")
      val inception_4c = Inception_Layer_v1(inception_4b, 512,
        T(T(128), T(128, 256), T(24, 64), T(64)), "inception_4c/")
      val inception_4d = Inception_Layer_v1(inception_4c, 512,
        T(T(112), T(144, 288), T(32, 64), T(64)), "inception_4d/")
      val inception_4e = Inception_Layer_v1(inception_4d, 528,
        T(T(256), T(160, 320), T(32, 128), T(128)), "inception_4e/")
      val pool4 = SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool4/3x3_s2").inputs(inception_4e)
      val inception_5a = Inception_Layer_v1(pool4, 832,
        T(T(256), T(160, 320), T(32, 128), T(128)), "inception_5a/")
      val inception_5b = Inception_Layer_v1(inception_5a,
        832, T(T(384), T(192, 384), T(48, 128), T(128)), "inception_5b/")
      val pool5 = SpatialAveragePooling(7, 7, 1, 1).setName("pool5/7x7_s1").inputs(inception_5b)
      val drop = if (hasDropout) Dropout(0.4).setName("pool5/drop_7x7_s1").inputs(pool5) else pool5
      val view = View(1024).setNumInputDims(3).inputs(drop)
      val classifier = Linear(1024, classNum).setInitMethod(weightInitMethod = Xavier, Zeros)
        .setName("loss3/classifier").inputs(view)
      val loss = LogSoftMax().setName("loss3/loss3").inputs(classifier)

      Graph.dynamic(input, loss)
    }
  }

  object LeNet5 {
    def graph(classNum: Int): Module[Float] = {
      val input = Reshape(Array(1, 28, 28)).inputs()
      val conv1 = SpatialConvolution(1, 6, 5, 5).setName("conv1_5x5").inputs(input)
      val tanh1 = Tanh().inputs(conv1)
      val pool1 = SpatialMaxPooling(2, 2, 2, 2).inputs(tanh1)
      val tanh2 = Tanh().inputs(pool1)
      val conv2 = SpatialConvolution(6, 12, 5, 5).setName("conv2_5x5").inputs(tanh2)
      val pool2 = SpatialMaxPooling(2, 2, 2, 2).inputs(conv2)
      val reshape = Reshape(Array(12 * 4 * 4)).inputs(pool2)
      val fc1 = Linear(12 * 4 * 4, 100).setName("fc1").inputs(reshape)
      val tanh3 = Tanh().inputs(fc1)
      val fc2 = Linear(100, classNum).setName("fc2").inputs(tanh3)
      val output = LogSoftMax().inputs(fc2)

      Graph.dynamic(input, output)
    }
  }

  object VggForCifar10 {
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
      Graph.dynamic(input, output)
    }
  }

  object Vgg_16 {
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

      Graph.dynamic(conv1, output)
    }
  }

  object Vgg_19 {
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

      Graph.dynamic(conv1, output)
    }
  }
}
