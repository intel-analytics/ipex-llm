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

package com.intel.analytics.bigdl.models.inception

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.{Graph, _}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.{T, Table}
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.Module

object Inception_Layer_v1 {
  def apply(inputSize: Int, config: Table, namePrefix : String = "") : Module[Float] = {
    val concat = Concat(2)
    val conv1 = Sequential()
    conv1.add(SpatialConvolution(inputSize,
      config[Table](1)(1), 1, 1, 1, 1)
      .setInitMethod(weightInitMethod = Xavier, ConstInitMethod(0.1)).setName(namePrefix + "1x1"))
    conv1.add(ReLU(true).setName(namePrefix + "relu_1x1"))
    concat.add(conv1)
    val conv3 = Sequential()
    conv3.add(SpatialConvolution(inputSize,
      config[Table](2)(1), 1, 1, 1, 1)
      .setInitMethod(weightInitMethod = Xavier,
        ConstInitMethod(0.1)).setName(namePrefix + "3x3_reduce"))
    conv3.add(ReLU(true).setName(namePrefix + "relu_3x3_reduce"))
    conv3.add(SpatialConvolution(config[Table](2)(1),
      config[Table](2)(2), 3, 3, 1, 1, 1, 1)
      .setInitMethod(weightInitMethod = Xavier, ConstInitMethod(0.1)).setName(namePrefix + "3x3"))
    conv3.add(ReLU(true).setName(namePrefix + "relu_3x3"))
    concat.add(conv3)
    val conv5 = Sequential()
    conv5.add(SpatialConvolution(inputSize,
      config[Table](3)(1), 1, 1, 1, 1)
      .setInitMethod(weightInitMethod = Xavier,
        ConstInitMethod(0.1)).setName(namePrefix + "5x5_reduce"))
    conv5.add(ReLU(true).setName(namePrefix + "relu_5x5_reduce"))
    conv5.add(SpatialConvolution(config[Table](3)(1),
      config[Table](3)(2), 5, 5, 1, 1, 2, 2)
      .setInitMethod(weightInitMethod = Xavier, ConstInitMethod(0.1)).setName(namePrefix + "5x5"))
    conv5.add(ReLU(true).setName(namePrefix + "relu_5x5"))
    concat.add(conv5)
    val pool = Sequential()
    pool.add(SpatialMaxPooling(3, 3, 1, 1, 1, 1).ceil().setName(namePrefix + "pool"))
    pool.add(SpatialConvolution(inputSize,
      config[Table](4)(1), 1, 1, 1, 1)
      .setInitMethod(weightInitMethod = Xavier,
        ConstInitMethod(0.1)).setName(namePrefix + "pool_proj"))
    pool.add(ReLU(true).setName(namePrefix + "relu_pool_proj"))
    concat.add(pool).setName(namePrefix + "output")
    concat
  }

  def apply(input: ModuleNode[Float], inputSize: Int, config: Table, namePrefix : String)
  : ModuleNode[Float] = {
    val conv1x1 = SpatialConvolution(inputSize, config[Table](1)(1), 1, 1, 1, 1)
        .setInitMethod(weightInitMethod = Xavier,
          ConstInitMethod(0.1)).setName(namePrefix + "1x1").inputs(input)
    val relu1x1 = ReLU(true).setName(namePrefix + "relu_1x1").inputs(conv1x1)

    val conv3x3_1 = SpatialConvolution(inputSize, config[Table](2)(1), 1, 1, 1, 1).setInitMethod(
      weightInitMethod = Xavier,
      ConstInitMethod(0.1)).setName(namePrefix + "3x3_reduce").inputs(input)
    val relu3x3_1 = ReLU(true).setName(namePrefix + "relu_3x3_reduce").inputs(conv3x3_1)
    val conv3x3_2 = SpatialConvolution(
      config[Table](2)(1), config[Table](2)(2), 3, 3, 1, 1, 1, 1)
      .setInitMethod(weightInitMethod = Xavier,
        ConstInitMethod(0.1)).setName(namePrefix + "3x3").inputs(relu3x3_1)
    val relu3x3_2 = ReLU(true).setName(namePrefix + "relu_3x3").inputs(conv3x3_2)

    val conv5x5_1 = SpatialConvolution(inputSize, config[Table](3)(1), 1, 1, 1, 1).setInitMethod(
      weightInitMethod = Xavier,
      ConstInitMethod(0.1)).setName(namePrefix + "5x5_reduce").inputs(input)
    val relu5x5_1 = ReLU(true).setName(namePrefix + "relu_5x5_reduce").inputs(conv5x5_1)
    val conv5x5_2 = SpatialConvolution(
      config[Table](3)(1), config[Table](3)(2), 5, 5, 1, 1, 2, 2)
      .setInitMethod(weightInitMethod = Xavier,
        ConstInitMethod(0.1)).setName(namePrefix + "5x5").inputs(relu5x5_1)
    val relu5x5_2 = ReLU(true).setName(namePrefix + "relu_5x5").inputs(conv5x5_2)

    val pool = SpatialMaxPooling(3, 3, 1, 1, 1, 1).ceil()
      .setName(namePrefix + "pool").inputs(input)
    val convPool = SpatialConvolution(inputSize, config[Table](4)(1), 1, 1, 1, 1).setInitMethod(
      weightInitMethod = Xavier,
      ConstInitMethod(0.1)).setName(namePrefix + "pool_proj").inputs(pool)
    val reluPool = ReLU(true).setName(namePrefix + "relu_pool_proj").inputs(convPool)

    JoinTable(2, 0).inputs(relu1x1, relu3x3_2, relu5x5_2, reluPool)
  }
}

object Inception_v1_NoAuxClassifier {
  def apply(classNum: Int, hasDropout: Boolean = true): Module[Float] = {
    val model = Sequential()
    model.add(SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3, 1, false)
      .setInitMethod(weightInitMethod = Xavier, ConstInitMethod(0.1))
      .setName("conv1/7x7_s2"))
    model.add(ReLU(true).setName("conv1/relu_7x7"))
    model.add(SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool1/3x3_s2"))
    model.add(SpatialCrossMapLRN(5, 0.0001, 0.75).setName("pool1/norm1"))
    model.add(SpatialConvolution(64, 64, 1, 1, 1, 1).
      setInitMethod(weightInitMethod = Xavier, ConstInitMethod(0.1))
      .setName("conv2/3x3_reduce"))
    model.add(ReLU(true).setName("conv2/relu_3x3_reduce"))
    model.add(SpatialConvolution(64, 192, 3, 3, 1, 1, 1, 1)
      .setInitMethod(weightInitMethod = Xavier, ConstInitMethod(0.1)).setName("conv2/3x3"))
    model.add(ReLU(true).setName("conv2/relu_3x3"))
    model.add(SpatialCrossMapLRN(5, 0.0001, 0.75). setName("conv2/norm2"))
    model.add(SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool2/3x3_s2"))
    model.add(Inception_Layer_v1(192, T(T(64), T(96, 128), T(16, 32), T(32)), "inception_3a/"))
    model.add(Inception_Layer_v1(256, T(T(128), T(128, 192), T(32, 96), T(64)), "inception_3b/"))
    model.add(SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool3/3x3_s2"))
    model.add(Inception_Layer_v1(480, T(T(192), T(96, 208), T(16, 48), T(64)), "inception_4a/"))
    model.add(Inception_Layer_v1(512, T(T(160), T(112, 224), T(24, 64), T(64)), "inception_4b/"))
    model.add(Inception_Layer_v1(512, T(T(128), T(128, 256), T(24, 64), T(64)), "inception_4c/"))
    model.add(Inception_Layer_v1(512, T(T(112), T(144, 288), T(32, 64), T(64)), "inception_4d/"))
    model.add(Inception_Layer_v1(528, T(T(256), T(160, 320), T(32, 128), T(128)), "inception_4e/"))
    model.add(SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool4/3x3_s2"))
    model.add(Inception_Layer_v1(832, T(T(256), T(160, 320), T(32, 128), T(128)), "inception_5a/"))
    model.add(Inception_Layer_v1(832, T(T(384), T(192, 384), T(48, 128), T(128)), "inception_5b/"))
    model.add(SpatialAveragePooling(7, 7, 1, 1).setName("pool5/7x7_s1"))
    if (hasDropout) model.add(Dropout(0.4).setName("pool5/drop_7x7_s1"))
    model.add(View(1024).setNumInputDims(3))
    model.add(Linear(1024, classNum)
      .setInitMethod(weightInitMethod = Xavier, Zeros).setName("loss3/classifier"))
    model.add(LogSoftMax().setName("loss3/loss3"))
    model
  }

  def graph(classNum: Int, hasDropout: Boolean = true)
  : Module[Float] = {
   val input = SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3, 1, false)
      .setInitMethod(weightInitMethod = Xavier,
        ConstInitMethod(0.1)).setName("conv1/7x7_s2").inputs()
    val conv1_relu = ReLU(true).setName("conv1/relu_7x7").inputs(input)
    val pool1_s2 = SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool1/3x3_s2").inputs(conv1_relu)
    val pool1_norm1 = SpatialCrossMapLRN(5, 0.0001, 0.75).setName("pool1/norm1").inputs(pool1_s2)
    val conv2 = SpatialConvolution(64, 64, 1, 1, 1, 1).setInitMethod(weightInitMethod = Xavier,
      ConstInitMethod(0.1)).setName("conv2/3x3_reduce").inputs(pool1_norm1)
    val conv2_relu = ReLU(true).setName("conv2/relu_3x3_reduce").inputs(conv2)
    val conv2_3x3 = SpatialConvolution(64, 192, 3, 3, 1, 1, 1, 1)
      .setInitMethod(weightInitMethod = Xavier, ConstInitMethod(0.1)).
      setName("conv2/3x3").inputs(conv2_relu)
    val conv2_relu_3x3 = ReLU(true).setName("conv2/relu_3x3").inputs(conv2_3x3)
    val conv2_norm2 = SpatialCrossMapLRN(5, 0.0001, 0.75)
      .setName("conv2/norm2").inputs(conv2_relu_3x3)
    val pool2_s2 = SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool2/3x3_s2").inputs(conv2_norm2)
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

    Graph(input, loss)
  }
}

object Inception_v1 {
  def apply(classNum: Int, hasDropout: Boolean = true): Module[Float] = {
    val feature1 = Sequential()
    feature1.add(SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3, 1, false)
      .setInitMethod(weightInitMethod = Xavier, ConstInitMethod(0.1))
      .setName("conv1/7x7_s2"))
    feature1.add(ReLU(true).setName("conv1/relu_7x7"))
    feature1.add(SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool1/3x3_s2"))
    feature1.add(SpatialCrossMapLRN(5, 0.0001, 0.75).setName("pool1/norm1"))
    feature1.add(SpatialConvolution(64, 64, 1, 1, 1, 1)
      .setInitMethod(weightInitMethod = Xavier, ConstInitMethod(0.1))
      .setName("conv2/3x3_reduce"))
    feature1.add(ReLU(true).setName("conv2/relu_3x3_reduce"))
    feature1.add(SpatialConvolution(64, 192, 3, 3, 1, 1, 1, 1)
      .setInitMethod(weightInitMethod = Xavier, ConstInitMethod(0.1))
      .setName("conv2/3x3"))
    feature1.add(ReLU(true).setName("conv2/relu_3x3"))
    feature1.add(SpatialCrossMapLRN(5, 0.0001, 0.75). setName("conv2/norm2"))
    feature1.add(SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool2/3x3_s2"))
    feature1.add(Inception_Layer_v1(192, T(T(64), T(96, 128), T(16, 32), T(32)), "inception_3a/"))
    feature1.add(Inception_Layer_v1(256, T(T(128), T(128, 192), T(32, 96), T(64)), "inception_3b/"))
    feature1.add(SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool3/3x3_s2"))
    feature1.add(Inception_Layer_v1(480, T(T(192), T(96, 208), T(16, 48), T(64)), "inception_4a/"))

    val output1 = Sequential()
    output1.add(SpatialAveragePooling(5, 5, 3, 3).ceil().setName("loss1/ave_pool"))
    output1.add(SpatialConvolution(512, 128, 1, 1, 1, 1).setName("loss1/conv"))
    output1.add(ReLU(true).setName("loss1/relu_conv"))
    output1.add(View(128 * 4 * 4).setNumInputDims(3))
    output1.add(Linear(128 * 4 * 4, 1024).setName("loss1/fc"))
    output1.add(ReLU(true).setName("loss1/relu_fc"))
    if (hasDropout) output1.add(Dropout(0.7).setName("loss1/drop_fc"))
    output1.add(Linear(1024, classNum).setName("loss1/classifier"))
    output1.add(LogSoftMax().setName("loss1/loss"))

    val feature2 = Sequential()
    feature2.add(Inception_Layer_v1(512, T(T(160), T(112, 224), T(24, 64), T(64)), "inception_4b/"))
    feature2.add(Inception_Layer_v1(512, T(T(128), T(128, 256), T(24, 64), T(64)), "inception_4c/"))
    feature2.add(Inception_Layer_v1(512, T(T(112), T(144, 288), T(32, 64), T(64)), "inception_4d/"))

    val output2 = Sequential()
    output2.add(SpatialAveragePooling(5, 5, 3, 3).setName("loss2/ave_pool"))
    output2.add(SpatialConvolution(528, 128, 1, 1, 1, 1).setName("loss2/conv"))
    output2.add(ReLU(true).setName("loss2/relu_conv"))
    output2.add(View(128 * 4 * 4).setNumInputDims(3))
    output2.add(Linear(128 * 4 * 4, 1024).setName("loss2/fc"))
    output2.add(ReLU(true).setName("loss2/relu_fc"))
    if (hasDropout) output2.add(Dropout(0.7).setName("loss2/drop_fc"))
    output2.add(Linear(1024, classNum).setName("loss2/classifier"))
    output2.add(LogSoftMax().setName("loss2/loss"))

    val output3 = Sequential()
    output3.add(Inception_Layer_v1(528, T(T(256), T(160, 320), T(32, 128), T(128)),
      "inception_4e/"))
    output3.add(SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool4/3x3_s2"))
    output3.add(Inception_Layer_v1(832, T(T(256), T(160, 320), T(32, 128), T(128)),
      "inception_5a/"))
    output3.add(Inception_Layer_v1(832, T(T(384), T(192, 384), T(48, 128), T(128)),
      "inception_5b/"))
    output3.add(SpatialAveragePooling(7, 7, 1, 1).setName("pool5/7x7_s1"))
    if (hasDropout) output3.add(Dropout(0.4).setName("pool5/drop_7x7_s1"))
    output3.add(View(1024).setNumInputDims(3))
    output3.add(Linear(1024, classNum)
      .setInitMethod(weightInitMethod = Xavier, Zeros).setName("loss3/classifier"))
    output3.add(LogSoftMax().setName("loss3/loss3"))

    val split2 = Concat(2).setName("split2")
    split2.add(output3)
    split2.add(output2)

    val mainBranch = Sequential()
    mainBranch.add(feature2)
    mainBranch.add(split2)

    val split1 = Concat(2).setName("split1")
    split1.add(mainBranch)
    split1.add(output1)

    val model = Sequential()

    model.add(feature1)
    model.add(split1)

    model
  }

  def graph(classNum: Int, hasDropout: Boolean = true): Module[Float] = {
    val input = Input()
    val conv1 = SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3, 1, false)
      .setInitMethod(weightInitMethod = Xavier, ConstInitMethod(0.1))
      .setName("conv1/7x7_s2").inputs(input)
    val relu1 = ReLU(true).setName("conv1/relu_7x7").inputs(conv1)
    val pool1 = SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool1/3x3_s2").inputs(relu1)
    val lrn1 = SpatialCrossMapLRN(5, 0.0001, 0.75).setName("pool1/norm1").inputs(pool1)
    val conv2 = SpatialConvolution(64, 64, 1, 1, 1, 1)
      .setInitMethod(weightInitMethod = Xavier, ConstInitMethod(0.1))
      .setName("conv2/3x3_reduce").inputs(lrn1)
    val relu2 = ReLU(true).setName("conv2/relu_3x3_reduce").inputs(conv2)
    val conv3 = SpatialConvolution(64, 192, 3, 3, 1, 1, 1, 1)
      .setInitMethod(weightInitMethod = Xavier, ConstInitMethod(0.1))
      .setName("conv2/3x3").inputs(relu2)
    val relu3 = ReLU(true).setName("conv2/relu_3x3").inputs(conv3)
    val lrn2 = SpatialCrossMapLRN(5, 0.0001, 0.75). setName("conv2/norm2").inputs(relu3)
    val pool2 = SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool2/3x3_s2").inputs(lrn2)
    val layer1 = Inception_Layer_v1(pool2, 192, T(T(64), T(96, 128), T(16, 32), T(32)),
      "inception_3a/")
    val layer2 = Inception_Layer_v1(layer1, 256, T(T(128), T(128, 192), T(32, 96), T(64)),
      "inception_3b/")
    val pool3 = SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool3/3x3_s2").inputs(layer2)
    val feature1 = Inception_Layer_v1(pool3, 480, T(T(192), T(96, 208), T(16, 48), T(64)),
      "inception_4a/")

    val pool2_1 = SpatialAveragePooling(5, 5, 3, 3).ceil()
      .setName("loss1/ave_pool").inputs(feature1)
    val loss2_1 = SpatialConvolution(512, 128, 1, 1, 1, 1).setName("loss1/conv").inputs(pool2_1)
    val relu2_1 = ReLU(true).setName("loss1/relu_conv").inputs(loss2_1)
    val view2_1 = View(128 * 4 * 4).setNumInputDims(3).inputs(relu2_1)
    val linear2_1 = Linear(128 * 4 * 4, 1024).setName("loss1/fc").inputs(view2_1)
    val relu2_2 = ReLU(true).setName("loss1/relu_fc").inputs(linear2_1)
    val drop2_1 = if (hasDropout) Dropout(0.7).setName("loss1/drop_fc").inputs(relu2_2) else relu2_2
    val classifier2_1 = Linear(1024, classNum).setName("loss1/classifier").inputs(drop2_1)
    val output1 = LogSoftMax().setName("loss1/loss").inputs(classifier2_1)

    val layer3_1 = Inception_Layer_v1(feature1, 512, T(T(160), T(112, 224), T(24, 64), T(64)),
      "inception_4b/")
    val layer3_2 = Inception_Layer_v1(layer3_1, 512, T(T(128), T(128, 256), T(24, 64), T(64)),
      "inception_4c/")
    val feature2 = Inception_Layer_v1(layer3_2, 512, T(T(112), T(144, 288), T(32, 64), T(64)),
      "inception_4d/")

    val pool4_1 = SpatialAveragePooling(5, 5, 3, 3).setName("loss2/ave_pool").inputs(feature2)
    val conv4_1 = SpatialConvolution(528, 128, 1, 1, 1, 1).setName("loss2/conv").inputs(pool4_1)
    val relu4_1 = ReLU(true).setName("loss2/relu_conv").inputs(conv4_1)
    val view4_1 = View(128 * 4 * 4).setNumInputDims(3).inputs(relu4_1)
    val linear4_1 = Linear(128 * 4 * 4, 1024).setName("loss2/fc").inputs(view4_1)
    val relu4_2 = ReLU(true).setName("loss2/relu_fc").inputs(linear4_1)
    val drop4_1 = if (hasDropout) Dropout(0.7).setName("loss2/drop_fc").inputs(relu4_2) else relu4_2
    val linear4_2 = Linear(1024, classNum).setName("loss2/classifier").inputs(drop4_1)
    val output2 = LogSoftMax().setName("loss2/loss").inputs(linear4_2)

    val layer5_1 = Inception_Layer_v1(feature2, 528, T(T(256), T(160, 320), T(32, 128), T(128)),
      "inception_4e/")
    val pool5_1 = SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool4/3x3_s2").inputs(layer5_1)
    val layer5_2 = Inception_Layer_v1(pool5_1, 832, T(T(256), T(160, 320), T(32, 128), T(128)),
      "inception_5a/")
    val layer5_3 = Inception_Layer_v1(layer5_2, 832, T(T(384), T(192, 384), T(48, 128), T(128)),
      "inception_5b/")
    val pool5_4 = SpatialAveragePooling(7, 7, 1, 1).setName("pool5/7x7_s1").inputs(layer5_3)
    val drop5_1 = if (hasDropout) Dropout(0.4).setName("pool5/drop_7x7_s1")
      .inputs(pool5_4) else pool5_4
    val view5_1 = View(1024).setNumInputDims(3).inputs(drop5_1)
    val linear5_1 = Linear(1024, classNum)
      .setInitMethod(weightInitMethod = Xavier, Zeros).setName("loss3/classifier").inputs(view5_1)
    val output3 = LogSoftMax().setName("loss3/loss3").inputs(linear5_1)

    val split2 = JoinTable(2, 0).setName("split2").inputs(output3, output2, output1)
    Graph(input, split2)
  }
}

