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

/* Still in experimental stage! */

package com.intel.analytics.bigdl.models.inception

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.{T, Table}

object Inception_Layer_v2 {
  def apply(inputSize: Int, config: Table, namePrefix : String): Module[Float] = {
    val concat = Concat(2)
    if (config[Table](1)[Int](1) != 0) {
      val conv1 = Sequential()
      conv1.add(SpatialConvolution(inputSize, config[Table](1)(1), 1, 1, 1, 1)
        .setName(namePrefix + "1x1"))
      conv1.add(SpatialBatchNormalization(config[Table](1)(1), 1e-3)
        .setName(namePrefix + "1x1/bn"))
      conv1.add(ReLU(true).setName(namePrefix + "1x1/bn/sc/relu"))
      concat.add(conv1)
    }

    val conv3 = Sequential()
    conv3.add(SpatialConvolution(inputSize, config[Table](2)(1), 1, 1, 1, 1)
      .setName(namePrefix + "3x3_reduce"))
    conv3.add(SpatialBatchNormalization(config[Table](2)(1), 1e-3)
      .setName(namePrefix + "3x3_reduce/bn"))
    conv3.add(ReLU(true). setName(namePrefix + "3x3_reduce/bn/sc/relu"))
    if(config[Table](4)[String](1) == "max" && config[Table](4)[Int](2) == 0) {
      conv3.add(SpatialConvolution(config[Table](2)(1),
        config[Table](2)(2), 3, 3, 2, 2, 1, 1).setName(namePrefix + "3x3"))
    } else {
      conv3.add(SpatialConvolution(config[Table](2)(1),
        config[Table](2)(2), 3, 3, 1, 1, 1, 1).setName(namePrefix + "3x3"))
    }
    conv3.add(SpatialBatchNormalization(config[Table](2)(2), 1e-3)
      .setName(namePrefix + "3x3/bn"))
    conv3.add(ReLU(true).setName(namePrefix + "3x3/bn/sc/relu"))
    concat.add(conv3)

    val conv3xx = Sequential()
    conv3xx.add(SpatialConvolution(inputSize, config[Table](3)(1), 1, 1, 1, 1)
      .setName(namePrefix + "double3x3_reduce"))
    conv3xx.add(SpatialBatchNormalization(config[Table](3)(1), 1e-3)
      .setName(namePrefix + "double3x3_reduce/bn"))
    conv3xx.add(ReLU(true).setName(namePrefix + "double3x3_reduce/bn/sc/relu"))

    conv3xx.add(SpatialConvolution(config[Table](3)(1),
      config[Table](3)(2), 3, 3, 1, 1, 1, 1).setName(namePrefix + "double3x3a"))
    conv3xx.add(SpatialBatchNormalization(config[Table](3)(2), 1e-3)
      .setName(namePrefix + "double3x3a/bn"))
    conv3xx.add(ReLU(true).setName(namePrefix + "double3x3a/bn/sc/relu"))

    if(config[Table](4)[String](1) == "max" && config[Table](4)[Int](2) == 0) {
      conv3xx.add(SpatialConvolution(config[Table](3)(2),
        config[Table](3)(2), 3, 3, 2, 2, 1, 1).setName(namePrefix + "double3x3b"))
    } else {
      conv3xx.add(SpatialConvolution(config[Table](3)(2),
        config[Table](3)(2), 3, 3, 1, 1, 1, 1).setName(namePrefix + "double3x3b"))
    }
    conv3xx.add(SpatialBatchNormalization(config[Table](3)(2), 1e-3)
      .setName(namePrefix + "double3x3b/bn"))
    conv3xx.add(ReLU(true).setName(namePrefix + "double3x3b/bn/sc/relu"))
    concat.add(conv3xx)

    val pool = Sequential()
    config[Table](4)[String](1) match {
      case "max" =>
        if (config[Table](4)[Int](2) != 0) {
          pool.add(SpatialMaxPooling(3, 3, 1, 1, 1, 1).ceil().setName(namePrefix + "pool"))
        } else {
          pool.add(SpatialMaxPooling(3, 3, 2, 2).ceil().setName(namePrefix + "pool"))
        }
      case "avg" => pool.add(SpatialAveragePooling(3, 3, 1, 1, 1, 1).ceil()
        .setName(namePrefix + "pool"))
      case _ => throw new IllegalArgumentException
    }

    if (config[Table](4)[Int](2) != 0) {
      pool.add(SpatialConvolution(inputSize, config[Table](4)[Int](2), 1, 1, 1, 1)
        .setName(namePrefix + "pool_proj"))
      pool.add(SpatialBatchNormalization(config[Table](4)(2), 1e-3)
        .setName(namePrefix + "pool_proj/bn"))
      pool.add(ReLU(true).setName(namePrefix + "pool_proj/bn/sc/relu"))
    }
    concat.add(pool)
    concat.setName(namePrefix + "output")
  }

  def apply(input: ModuleNode[Float], inputSize: Int, config: Table, namePrefix : String)
  : ModuleNode[Float] = {
    val relu1 = if (config[Table](1)[Int](1) != 0) {
      val conv1 = SpatialConvolution(inputSize, config[Table](1)(1), 1, 1, 1, 1)
        .setName(namePrefix + "1x1").inputs(input)
      val bn1 = SpatialBatchNormalization(config[Table](1)(1), 1e-3)
        .setName(namePrefix + "1x1/bn").inputs(conv1)
      ReLU(true).setName(namePrefix + "1x1/bn/sc/relu").inputs(bn1)
    } else {
      null
    }

    val conv2 = SpatialConvolution(inputSize, config[Table](2)(1), 1, 1, 1, 1)
      .setName(namePrefix + "3x3_reduce").inputs(input)
    val bn2 = SpatialBatchNormalization(config[Table](2)(1), 1e-3)
      .setName(namePrefix + "3x3_reduce/bn").inputs(conv2)
    val relu2 = ReLU(true). setName(namePrefix + "3x3_reduce/bn/sc/relu").inputs(bn2)
    val conv3 = if (config[Table](4)[String](1) == "max" && config[Table](4)[Int](2) == 0) {
      SpatialConvolution(config[Table](2)(1),
        config[Table](2)(2), 3, 3, 2, 2, 1, 1).setName(namePrefix + "3x3").inputs(relu2)
    } else {
      SpatialConvolution(config[Table](2)(1),
        config[Table](2)(2), 3, 3, 1, 1, 1, 1).setName(namePrefix + "3x3").inputs(relu2)
    }
    val bn3 = SpatialBatchNormalization(config[Table](2)(2), 1e-3)
      .setName(namePrefix + "3x3/bn").inputs(conv3)
    val relu4 = ReLU(true).setName(namePrefix + "3x3/bn/sc/relu").inputs(bn3)

    val conv4 = SpatialConvolution(inputSize, config[Table](3)(1), 1, 1, 1, 1)
      .setName(namePrefix + "double3x3_reduce").inputs(input)
    val bn4 = SpatialBatchNormalization(config[Table](3)(1), 1e-3)
      .setName(namePrefix + "double3x3_reduce/bn").inputs(conv4)
    val relu5 = ReLU(true).setName(namePrefix + "double3x3_reduce/bn/sc/relu").inputs(bn4)

    val conv5 = SpatialConvolution(config[Table](3)(1),
      config[Table](3)(2), 3, 3, 1, 1, 1, 1).setName(namePrefix + "double3x3a").inputs(relu5)
    val bn5 = SpatialBatchNormalization(config[Table](3)(2), 1e-3)
      .setName(namePrefix + "double3x3a/bn").inputs(conv5)
    val relu6 = ReLU(true).setName(namePrefix + "double3x3a/bn/sc/relu").inputs(bn5)
    val conv6 = if (config[Table](4)[String](1) == "max" && config[Table](4)[Int](2) == 0) {
      SpatialConvolution(config[Table](3)(2),
        config[Table](3)(2), 3, 3, 2, 2, 1, 1).setName(namePrefix + "double3x3b").inputs(relu6)
    } else {
      SpatialConvolution(config[Table](3)(2),
        config[Table](3)(2), 3, 3, 1, 1, 1, 1).setName(namePrefix + "double3x3b").inputs(relu6)
    }
    val bn6 = SpatialBatchNormalization(config[Table](3)(2), 1e-3)
      .setName(namePrefix + "double3x3b/bn").inputs(conv6)
    val relu7 = ReLU(true).setName(namePrefix + "double3x3b/bn/sc/relu").inputs(bn6)

    val pool = config[Table](4)[String](1) match {
      case "max" =>
        if (config[Table](4)[Int](2) != 0) {
          SpatialMaxPooling(3, 3, 1, 1, 1, 1).ceil().setName(namePrefix + "pool").inputs(input)
        } else {
          SpatialMaxPooling(3, 3, 2, 2).ceil().setName(namePrefix + "pool").inputs(input)
        }
      case "avg" => SpatialAveragePooling(3, 3, 1, 1, 1, 1).ceil()
        .setName(namePrefix + "pool").inputs(input)
      case _ => throw new IllegalArgumentException
    }
    val reluPool = if (config[Table](4)[Int](2) != 0) {
      val conv = SpatialConvolution(inputSize, config[Table](4)[Int](2), 1, 1, 1, 1)
        .setName(namePrefix + "pool_proj").inputs(pool)
      val bn = SpatialBatchNormalization(config[Table](4)(2), 1e-3)
        .setName(namePrefix + "pool_proj/bn").inputs(conv)
      ReLU(true).setName(namePrefix + "pool_proj/bn/sc/relu").inputs(bn)
    } else {
      pool
    }
    if (relu1 == null) {
      JoinTable(2, 0).inputs(relu4, relu7, reluPool)
    } else {
      JoinTable(2, 0).inputs(relu1, relu4, relu7, reluPool)
    }
  }
}

object Inception_v2_NoAuxClassifier {
  def apply(classNum: Int): Module[Float] = {
    val model = Sequential()
    model.add(SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3, 1, false)
      .setName("conv1/7x7_s2"))
    model.add(SpatialBatchNormalization(64, 1e-3).setName("conv1/7x7_s2/bn"))
    model.add(ReLU(true).setName("conv1/7x7_s2/bn/sc/relu"))
    model.add(SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool1/3x3_s2"))
    model.add(SpatialConvolution(64, 64, 1, 1).setName("conv2/3x3_reduce"))
    model.add(SpatialBatchNormalization(64, 1e-3).setName("conv2/3x3_reduce/bn"))
    model.add(ReLU(true).setName("conv2/3x3_reduce/bn/sc/relu"))
    model.add(SpatialConvolution(64, 192, 3, 3, 1, 1, 1, 1).setName("conv2/3x3"))
    model.add(SpatialBatchNormalization(192, 1e-3).setName("conv2/3x3/bn"))
    model.add(ReLU(true).setName("conv2/3x3/bn/sc/relu"))
    model.add(SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool2/3x3_s2"))
    model.add(Inception_Layer_v2(192, T(T(64), T(64, 64), T(64, 96), T("avg", 32)),
      "inception_3a/"))
    model.add(Inception_Layer_v2(256, T(T(64), T(64, 96), T(64, 96), T("avg", 64)),
      "inception_3b/"))
    model.add(Inception_Layer_v2(320, T(T(0), T(128, 160), T(64, 96), T("max", 0)),
      "inception_3c/"))
    model.add(Inception_Layer_v2(576, T(T(224), T(64, 96), T(96, 128), T("avg", 128)),
      "inception_4a/"))
    model.add(Inception_Layer_v2(576, T(T(192), T(96, 128), T(96, 128), T("avg", 128)),
      "inception_4b/"))
    model.add(Inception_Layer_v2(576, T(T(160), T(128, 160), T(128, 160), T("avg", 96)),
      "inception_4c/"))
    model.add(Inception_Layer_v2(576, T(T(96), T(128, 192), T(160, 192), T("avg", 96)),
      "inception_4d/"))
    model.add(Inception_Layer_v2(576, T(T(0), T(128, 192), T(192, 256), T("max", 0)),
      "inception_4e/"))
    model.add(Inception_Layer_v2(1024, T(T(352), T(192, 320), T(160, 224), T("avg", 128)),
      "inception_5a/"))
    model.add(Inception_Layer_v2(1024, T(T(352), T(192, 320), T(192, 224), T("max", 128)),
      "inception_5b/"))
    model.add(SpatialAveragePooling(7, 7, 1, 1).ceil().setName("pool5/7x7_s1"))
    model.add(View(1024).setNumInputDims(3))
    model.add(Linear(1024, classNum).setName("loss3/classifier"))
    model.add(LogSoftMax().setName("loss3/loss"))

    model.reset()
    model
  }

  def graph(classNum: Int): Module[Float] = {
    val input = Input()
    val conv1 = SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3, 1, false)
      .setName("conv1/7x7_s2").inputs(input)
    val bn1 = SpatialBatchNormalization(64, 1e-3).setName("conv1/7x7_s2/bn").inputs(conv1)
    val relu1 = ReLU(true).setName("conv1/7x7_s2/bn/sc/relu").inputs(bn1)
    val pool1 = SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool1/3x3_s2").inputs(relu1)
    val conv2 = SpatialConvolution(64, 64, 1, 1).setName("conv2/3x3_reduce").inputs(pool1)
    val bn2 = SpatialBatchNormalization(64, 1e-3).setName("conv2/3x3_reduce/bn").inputs(conv2)
    val relu2 = ReLU(true).setName("conv2/3x3_reduce/bn/sc/relu").inputs(bn2)
    val conv3 = SpatialConvolution(64, 192, 3, 3, 1, 1, 1, 1).setName("conv2/3x3").inputs(relu2)
    val bn3 = SpatialBatchNormalization(192, 1e-3).setName("conv2/3x3/bn").inputs(conv3)
    val relu3 = ReLU(true).setName("conv2/3x3/bn/sc/relu").inputs(bn3)
    val pool2 = SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool2/3x3_s2").inputs(relu3)
    val layer1 = Inception_Layer_v2(pool2, 192, T(T(64), T(64, 64), T(64, 96), T("avg", 32)),
      "inception_3a/")
    val layer2 = Inception_Layer_v2(layer1, 256, T(T(64), T(64, 96), T(64, 96), T("avg", 64)),
      "inception_3b/")
    val layer3 = Inception_Layer_v2(layer2, 320, T(T(0), T(128, 160), T(64, 96), T("max", 0)),
      "inception_3c/")
    val layer4 = Inception_Layer_v2(layer3, 576, T(T(224), T(64, 96), T(96, 128), T("avg", 128)),
      "inception_4a/")
    val layer5 = Inception_Layer_v2(layer4, 576, T(T(192), T(96, 128), T(96, 128), T("avg", 128)),
      "inception_4b/")
    val layer6 = Inception_Layer_v2(layer5, 576, T(T(160), T(128, 160), T(128, 160), T("avg", 96)),
      "inception_4c/")
    val layer7 = Inception_Layer_v2(layer6, 576, T(T(96), T(128, 192), T(160, 192), T("avg", 96)),
      "inception_4d/")
    val layer8 = Inception_Layer_v2(layer7, 576, T(T(0), T(128, 192), T(192, 256), T("max", 0)),
      "inception_4e/")
    val layer9 = Inception_Layer_v2(layer8, 1024, T(T(352), T(192, 320), T(160, 224),
      T("avg", 128)), "inception_5a/")
    val layer10 = Inception_Layer_v2(layer9, 1024, T(T(352), T(192, 320), T(192, 224),
      T("max", 128)), "inception_5b/")

    val pool = SpatialAveragePooling(7, 7, 1, 1).ceil().setName("pool5/7x7_s1").inputs(layer10)
    val view1 = View(1024).setNumInputDims(3).inputs(pool)
    val linear = Linear(1024, classNum).setName("loss3/classifier").inputs(view1)
    val output = LogSoftMax().setName("loss3/loss").inputs(linear)

    val model = Graph(input, output)
    model.reset()
    model
  }
}

object Inception_v2 {
  def apply(classNum: Int): Module[Float] = {
    val features1 = Sequential()
    features1.add(SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3, 1, false)
      .setName("conv1/7x7_s2"))
    features1.add(SpatialBatchNormalization(64, 1e-3).setName("conv1/7x7_s2/bn"))
    features1.add(ReLU(true).setName("conv1/7x7_s2/bn/sc/relu"))
    features1.add(SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool1/3x3_s2"))
    features1.add(SpatialConvolution(64, 64, 1, 1).setName("conv2/3x3_reduce"))
    features1.add(SpatialBatchNormalization(64, 1e-3).setName("conv2/3x3_reduce/bn"))
    features1.add(ReLU(true).setName("conv2/3x3_reduce/bn/sc/relu"))
    features1.add(SpatialConvolution(64, 192, 3, 3, 1, 1, 1, 1).setName("conv2/3x3"))
    features1.add(SpatialBatchNormalization(192, 1e-3).setName("conv2/3x3/bn"))
    features1.add(ReLU(true).setName("conv2/3x3/bn/sc/relu"))
    features1.add(SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool2/3x3_s2"))
    features1.add(Inception_Layer_v2(192, T(T(64), T(64, 64), T(64, 96), T("avg", 32)),
      "inception_3a/"))
    features1.add(Inception_Layer_v2(256, T(T(64), T(64, 96), T(64, 96), T("avg", 64)),
      "inception_3b/"))
    features1.add(Inception_Layer_v2(320, T(T(0), T(128, 160), T(64, 96), T("max", 0)),
      "inception_3c/"))

    val output1 = Sequential()
    output1.add(SpatialAveragePooling(5, 5, 3, 3).ceil().setName("pool3/5x5_s3"))
    output1.add(SpatialConvolution(576, 128, 1, 1, 1, 1).setName("loss1/conv"))
    output1.add(SpatialBatchNormalization(128, 1e-3).setName("loss1/conv/bn"))
    output1.add(ReLU(true).setName("loss1/conv/bn/sc/relu"))
    output1.add(View(128 * 4 * 4).setNumInputDims(3))
    output1.add(Linear(128 * 4 * 4, 1024).setName("loss1/fc"))
    output1.add(ReLU(true).setName("loss1/fc/bn/sc/relu"))
    output1.add(Linear(1024, classNum).setName("loss1/classifier"))
    output1.add(LogSoftMax().setName("loss1/loss"))


    val features2 = Sequential()
    features2
      .add(Inception_Layer_v2(576, T(T(224), T(64, 96), T(96, 128), T("avg", 128)),
        "inception_4a/"))
      .add(Inception_Layer_v2(576, T(T(192), T(96, 128), T(96, 128), T("avg", 128)),
        "inception_4b/"))
      .add(Inception_Layer_v2(576, T(T(160), T(128, 160), T(128, 160), T("avg", 96)),
        "inception_4c/"))
      .add(Inception_Layer_v2(576, T(T(96), T(128, 192), T(160, 192), T("avg", 96)),
        "inception_4d/"))
      .add(Inception_Layer_v2(576, T(T(0), T(128, 192), T(192, 256), T("max", 0)),
        "inception_4e/"))

    val output2 = Sequential()
    output2.add(SpatialAveragePooling(5, 5, 3, 3).ceil().setName("pool4/5x5_s3"))
    output2.add(SpatialConvolution(1024, 128, 1, 1, 1, 1).setName("loss2/conv"))
    output2.add(SpatialBatchNormalization(128, 1e-3).setName("loss2/conv/bn"))
    output2.add(ReLU(true).setName("loss2/conv/bn/sc/relu"))
    output2.add(View(128 * 2 * 2).setNumInputDims(3))
    output2.add(Linear(128 * 2 * 2, 1024).setName("loss2/fc"))
    output2.add(ReLU(true).setName("loss2/fc/bn/sc/relu"))
    output2.add(Linear(1024, classNum).setName("loss2/classifier"))
    output2.add(LogSoftMax().setName("loss2/loss"))

    val output3 = Sequential()
    output3.add(Inception_Layer_v2(1024, T(T(352), T(192, 320), T(160, 224), T("avg", 128)),
      "inception_5a/"))
    output3.add(Inception_Layer_v2(1024, T(T(352), T(192, 320), T(192, 224), T("max", 128)),
      "inception_5b/"))
    output3.add(SpatialAveragePooling(7, 7, 1, 1).ceil().setName("pool5/7x7_s1"))
    output3.add(View(1024).setNumInputDims(3))
    output3.add(Linear(1024, classNum).setName("loss3/classifier"))
    output3.add(LogSoftMax().setName("loss3/loss"))

    val split2 = Concat(2)
    split2.add(output3)
    split2.add(output2)

    val mainBranch = Sequential()
    mainBranch.add(features2)
    mainBranch.add(split2)

    val split1 = Concat(2)
    split1.add(mainBranch)
    split1.add(output1)

    val model = Sequential()

    model.add(features1)
    model.add(split1)

    model.reset()
    model
  }

  def graph(classNum: Int, hasDropout: Boolean = true): Module[Float] = {
    val input = Input()
    val conv1 = SpatialConvolution(3, 64, 7, 7, 2, 2, 3, 3, 1, false)
      .setName("conv1/7x7_s2").inputs(input)
    val bn1 = SpatialBatchNormalization(64, 1e-3).setName("conv1/7x7_s2/bn").inputs(conv1)
    val relu1 = ReLU(true).setName("conv1/7x7_s2/bn/sc/relu").inputs(bn1)
    val pool1 = SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool1/3x3_s2").inputs(relu1)
    val conv2 = SpatialConvolution(64, 64, 1, 1).setName("conv2/3x3_reduce").inputs(pool1)
    val bn2 = SpatialBatchNormalization(64, 1e-3).setName("conv2/3x3_reduce/bn").inputs(conv2)
    val relu2 = ReLU(true).setName("conv2/3x3_reduce/bn/sc/relu").inputs(bn2)
    val conv3 = SpatialConvolution(64, 192, 3, 3, 1, 1, 1, 1).setName("conv2/3x3").inputs(relu2)
    val bn3 = SpatialBatchNormalization(192, 1e-3).setName("conv2/3x3/bn").inputs(conv3)
    val relu4 = ReLU(true).setName("conv2/3x3/bn/sc/relu").inputs(bn3)
    val pool2 = SpatialMaxPooling(3, 3, 2, 2).ceil().setName("pool2/3x3_s2").inputs(relu4)
    val layer1 = Inception_Layer_v2(pool2, 192, T(T(64), T(64, 64), T(64, 96), T("avg", 32)),
      "inception_3a/")
    val layer2 = Inception_Layer_v2(layer1, 256, T(T(64), T(64, 96), T(64, 96), T("avg", 64)),
      "inception_3b/")
    val features1 = Inception_Layer_v2(layer2, 320, T(T(0), T(128, 160), T(64, 96), T("max", 0)),
      "inception_3c/")

    val pool2_1 = SpatialAveragePooling(5, 5, 3, 3).ceil().setName("pool3/5x5_s3").inputs(features1)
    val conv2_1 = SpatialConvolution(576, 128, 1, 1, 1, 1).setName("loss1/conv").inputs(pool2_1)
    val bn2_1 = SpatialBatchNormalization(128, 1e-3).setName("loss1/conv/bn").inputs(conv2_1)
    val relu2_1 = ReLU(true).setName("loss1/conv/bn/sc/relu").inputs(bn2_1)
    val view2_1 = View(128 * 4 * 4).setNumInputDims(3).inputs(relu2_1)
    val linear2_1 = Linear(128 * 4 * 4, 1024).setName("loss1/fc").inputs(view2_1)
    val relu2_2 = ReLU(true).setName("loss1/fc/bn/sc/relu").inputs(linear2_1)
    val linear2_2 = Linear(1024, classNum).setName("loss1/classifier").inputs(relu2_2)
    val output1 = LogSoftMax().setName("loss1/loss").inputs(linear2_2)

    val layer3_1 = Inception_Layer_v2(features1, 576, T(T(224), T(64, 96), T(96, 128),
      T("avg", 128)), "inception_4a/")
     val layer3_2 = Inception_Layer_v2(layer3_1, 576, T(T(192), T(96, 128), T(96, 128),
       T("avg", 128)), "inception_4b/")
     val layer3_3 = Inception_Layer_v2(layer3_2, 576, T(T(160), T(128, 160), T(128, 160),
       T("avg", 96)), "inception_4c/")
     val layer3_4 = Inception_Layer_v2(layer3_3, 576, T(T(96), T(128, 192), T(160, 192),
       T("avg", 96)), "inception_4d/")
     val features2 = Inception_Layer_v2(layer3_4, 576, T(T(0), T(128, 192), T(192, 256),
       T("max", 0)), "inception_4e/")

    val pool3_1 = SpatialAveragePooling(5, 5, 3, 3).ceil().setName("pool4/5x5_s3").inputs(features2)
    val conv3_1 = SpatialConvolution(1024, 128, 1, 1, 1, 1).setName("loss2/conv").inputs(pool3_1)
    val bn3_1 = SpatialBatchNormalization(128, 1e-3).setName("loss2/conv/bn").inputs(conv3_1)
    val relu3_1 = ReLU(true).setName("loss2/conv/bn/sc/relu").inputs(bn3_1)
    val view3_1 = View(128 * 2 * 2).setNumInputDims(3).inputs(relu3_1)
    val linear3_1 = Linear(128 * 2 * 2, 1024).setName("loss2/fc").inputs(view3_1)
    val relu3_2 = ReLU(true).setName("loss2/fc/bn/sc/relu").inputs(linear3_1)
    val linear3_2 = Linear(1024, classNum).setName("loss2/classifier").inputs(relu3_2)
    val output2 = LogSoftMax().setName("loss2/loss").inputs(linear3_2)

    val rayer5_1 = Inception_Layer_v2(features2, 1024, T(T(352), T(192, 320), T(160, 224),
      T("avg", 128)), "inception_5a/")
    val layer5_2 = Inception_Layer_v2(rayer5_1, 1024, T(T(352), T(192, 320), T(192, 224),
      T("max", 128)), "inception_5b/")
    val pool5_1 = SpatialAveragePooling(7, 7, 1, 1).ceil().setName("pool5/7x7_s1").inputs(layer5_2)
    val view5_1 = View(1024).setNumInputDims(3).inputs(pool5_1)
    val linear5_1 = Linear(1024, classNum).setName("loss3/classifier").inputs(view5_1)
    val output3 = LogSoftMax().setName("loss3/loss").inputs(linear5_1)

    val split2 = JoinTable(2, 0).inputs(output3, output2, output1)
    val model = Graph(input, split2)
    model.reset()
    model
  }
}

