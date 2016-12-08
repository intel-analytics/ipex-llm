/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.models.imagenet

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

object GoogleNet_v1 {
  private def inception[D: ClassTag](inputSize: Int, config: Table, namePrefix : String)(
    implicit ev: TensorNumeric[D]): Module[D] = {
    val concat = Concat[D](2)
    val conv1 = Sequential[D]
    conv1.add(SpatialConvolution[D](inputSize,
      config[Table](1)(1), 1, 1, 1, 1).setInitMethod(Xavier).setName(namePrefix + "1x1"))
    conv1.add(ReLU[D](true).setName(namePrefix + "relu_1x1"))
    concat.add(conv1)
    val conv3 = Sequential[D]
    conv3.add(SpatialConvolution[D](inputSize,
      config[Table](2)(1), 1, 1, 1, 1).setInitMethod(Xavier).setName(namePrefix + "3x3_reduce"))
    conv3.add(ReLU[D](true).setName(namePrefix + "relu_3x3_reduce"))
    conv3.add(SpatialConvolution[D](config[Table](2)(1),
      config[Table](2)(2), 3, 3, 1, 1, 1, 1).setInitMethod(Xavier).setName(namePrefix + "3x3"))
    conv3.add(ReLU[D](true).setName(namePrefix + "relu_3x3"))
    concat.add(conv3)
    val conv5 = Sequential[D]
    conv5.add(SpatialConvolution[D](inputSize,
      config[Table](3)(1), 1, 1, 1, 1).setInitMethod(Xavier).setName(namePrefix + "5x5_reduce"))
    conv5.add(ReLU[D](true).setName(namePrefix + "relu_5x5_reduce"))
    conv5.add(SpatialConvolution[D](config[Table](3)(1),
      config[Table](3)(2), 5, 5, 1, 1, 2, 2).setInitMethod(Xavier).setName(namePrefix + "5x5"))
    conv5.add(ReLU[D](true).setName(namePrefix + "relu_5x5"))
    concat.add(conv5)
    val pool = Sequential[D]
    pool.add(SpatialMaxPooling[D](3, 3, 1, 1, 1, 1).ceil().setName(namePrefix + "pool"))
    pool.add(SpatialConvolution[D](inputSize,
      config[Table](4)(1), 1, 1, 1, 1).setInitMethod(Xavier).setName(namePrefix + "pool_proj"))
    pool.add(ReLU[D](true).setName(namePrefix + "relu_pool_proj"))
    concat.add(pool).setName(namePrefix + "output")
    concat
  }

  def apply[D: ClassTag](classNum: Int)
    (implicit ev: TensorNumeric[D]): Module[D] = {
    val feature1 = Sequential[D]
    feature1.add(SpatialConvolution[D](3, 64, 7, 7, 2, 2, 3, 3, 1, false).setInitMethod(Xavier)
      .setName("conv1/7x7_s2"))
    feature1.add(ReLU[D](true).setName("conv1/relu_7x7"))
    feature1.add(SpatialMaxPooling[D](3, 3, 2, 2).ceil().setName("pool1/3x3_s2"))
    feature1.add(SpatialCrossMapLRN[D](5, 0.0001, 0.75).setName("pool1/norm1"))
    feature1.add(SpatialConvolution[D](64, 64, 1, 1, 1, 1).setInitMethod(Xavier)
      .setName("conv2/3x3_reduce"))
    feature1.add(ReLU[D](true).setName("conv2/relu_3x3_reduce"))
    feature1.add(SpatialConvolution[D](64, 192, 3, 3, 1, 1, 1, 1).setInitMethod(Xavier)
      .setName("conv2/3x3"))
    feature1.add(ReLU[D](true).setName("conv2/relu_3x3"))
    feature1.add(SpatialCrossMapLRN[D](5, 0.0001, 0.75). setName("conv2/norm2"))
    feature1.add(SpatialMaxPooling[D](3, 3, 2, 2).ceil().setName("pool2/3x3_s2"))
    feature1.add(inception[D](192, T(T(64), T(96, 128), T(16, 32), T(32)), "inception_3a/"))
    feature1.add(inception[D](256, T(T(128), T(128, 192), T(32, 96), T(64)), "inception_3b/"))
    feature1.add(SpatialMaxPooling[D](3, 3, 2, 2).ceil().setName("pool3/3x3_s2"))
    feature1.add(inception[D](480, T(T(192), T(96, 208), T(16, 48), T(64)), "inception_4a/"))

    val output1 = Sequential[D]
    output1.add(SpatialAveragePooling[D](5, 5, 3, 3).ceil().setName("loss1/ave_pool"))
    output1.add(SpatialConvolution[D](512, 128, 1, 1, 1, 1).setName("loss1/conv"))
    output1.add(ReLU[D](true).setName("loss1/relu_conv"))
    output1.add(View[D](128 * 4 * 4).setNumInputDims(3))
    output1.add(Linear[D](128 * 4 * 4, 1024).setName("loss1/fc"))
    output1.add(ReLU[D](true).setName("loss1/relu_fc"))
    output1.add(Dropout[D](0.7).setName("loss1/drop_fc"))
    output1.add(Linear[D](1024, classNum).setName("loss1/classifier"))
    output1.add(LogSoftMax[D].setName("loss1/loss"))

    val feature2 = Sequential[D]
    feature2.add(inception[D](512, T(T(160), T(112, 224), T(24, 64), T(64)), "inception_4b/"))
    feature2.add(inception[D](512, T(T(128), T(128, 256), T(24, 64), T(64)), "inception_4c/"))
    feature2.add(inception[D](512, T(T(112), T(144, 288), T(32, 64), T(64)), "inception_4d/"))

    val output2 = Sequential[D]
    output2.add(SpatialAveragePooling[D](5, 5, 3, 3).setName("loss2/ave_pool"))
    output2.add(SpatialConvolution[D](528, 128, 1, 1, 1, 1).setName("loss2/conv"))
    output2.add(ReLU[D](true).setName("loss2/relu_conv"))
    output2.add(View[D](128 * 4 * 4).setNumInputDims(3))
    output2.add(Linear[D](128 * 4 * 4, 1024).setName("loss2/fc"))
    output2.add(ReLU[D](true).setName("loss2/relu_fc"))
    output2.add(Dropout[D](0.7).setName("loss2/drop_fc"))
    output2.add(Linear[D](1024, classNum).setName("loss2/classifier"))
    output2.add(LogSoftMax[D].setName("loss2/loss"))

    val output3 = Sequential[D]
    output3.add(inception[D](528, T(T(256), T(160, 320), T(32, 128), T(128)), "inception_4e/"))
    output3.add(SpatialMaxPooling[D](3, 3, 2, 2).ceil().setName("pool4/3x3_s2"))
    output3.add(inception[D](832, T(T(256), T(160, 320), T(32, 128), T(128)), "inception_5a/"))
    output3.add(inception[D](832, T(T(384), T(192, 384), T(48, 128), T(128)), "inception_5b/"))
    output3.add(SpatialAveragePooling[D](7, 7, 1, 1).setName("pool5/7x7_s1"))
    output3.add(Dropout[D](0.4).setName("pool5/drop_7x7_s1"))
    output3.add(View[D](1024).setNumInputDims(3))
    output3.add(Linear[D](1024, classNum).setInitMethod(Xavier).setName("loss3/classifier"))
    output3.add(LogSoftMax[D].setName("loss3/loss3"))

    val split2 = Concat[D](2).setName("split2")
    split2.add(output3)
    split2.add(output2)

    val mainBranch = Sequential[D]()
    mainBranch.add(feature2)
    mainBranch.add(split2)

    val split1 = Concat[D](2).setName("split1")
    split1.add(mainBranch)
    split1.add(output1)

    val model = Sequential[D]()

    model.add(feature1)
    model.add(split1)

    model.reset()
    model
  }
}

object GoogleNet_v2_NoAuxClassifier {
  def apply[D: ClassTag](classNum: Int)
    (implicit ev: TensorNumeric[D]): Module[D] = {
    val model = Sequential[D]
    model.add(SpatialConvolution[D](3, 64, 7, 7, 2, 2, 3, 3, 1, false)
      .setName("conv1/7x7_s2"))
    model.add(SpatialBatchNormalization(64, 1e-3).setName("conv1/7x7_s2/bn"))
    model.add(ReLU[D](true).setName("conv1/7x7_s2/bn/sc/relu"))
    model.add(SpatialMaxPooling[D](3, 3, 2, 2).ceil().setName("pool1/3x3_s2"))
    model.add(SpatialConvolution[D](64, 64, 1, 1).setName("conv2/3x3_reduce"))
    model.add(SpatialBatchNormalization(64, 1e-3).setName("conv2/3x3_reduce/bn"))
    model.add(ReLU[D](true).setName("conv2/3x3_reduce/bn/sc/relu"))
    model.add(SpatialConvolution[D](64, 192, 3, 3, 1, 1, 1, 1).setName("conv2/3x3"))
    model.add(SpatialBatchNormalization(192, 1e-3).setName("conv2/3x3/bn"))
    model.add(ReLU[D](true).setName("conv2/3x3/bn/sc/relu"))
    model.add(SpatialMaxPooling[D](3, 3, 2, 2).ceil().setName("pool2/3x3_s2"))
    model.add(inception(192, T(T(64), T(64, 64), T(64, 96), T("avg", 32)), "inception_3a/"))
    model.add(inception(256, T(T(64), T(64, 96), T(64, 96), T("avg", 64)), "inception_3b/"))
    model.add(inception(320, T(T(0), T(128, 160), T(64, 96), T("max", 0)), "inception_3c/"))
    model.add(inception(576, T(T(224), T(64, 96), T(96, 128), T("avg", 128)), "inception_4a/"))
    model.add(inception(576, T(T(192), T(96, 128), T(96, 128), T("avg", 128)), "inception_4b/"))
    model.add(inception(576, T(T(160), T(128, 160), T(128, 160), T("avg", 96)),
      "inception_4c/"))
    model.add(inception(576, T(T(96), T(128, 192), T(160, 192), T("avg", 96)), "inception_4d/"))
    model.add(inception(576, T(T(0), T(128, 192), T(192, 256), T("max", 0)), "inception_4e/"))
    model.add(inception(1024, T(T(352), T(192, 320), T(160, 224), T("avg", 128)),
      "inception_5a/"))
    model.add(inception(1024, T(T(352), T(192, 320), T(192, 224), T("max", 128)),
      "inception_5b/"))
    model.add(SpatialAveragePooling[D](7, 7, 1, 1).ceil().setName("pool5/7x7_s1"))
    model.add(View[D](1024).setNumInputDims(3))
    model.add(Linear[D](1024, classNum).setName("loss3/classifier"))
    model.add(LogSoftMax[D].setName("loss3/loss"))

    model.reset()
    model
  }

  def inception[D: ClassTag](inputSize: Int, config: Table, namePrefix : String)(
    implicit ev: TensorNumeric[D]): Module[D] = {
    val concat = Concat[D](2)
    if (config[Table](1)[Int](1) != 0) {
      val conv1 = Sequential[D]
      conv1.add(SpatialConvolution[D](inputSize, config[Table](1)(1), 1, 1, 1, 1)
        .setName(namePrefix + "1x1"))
      conv1.add(SpatialBatchNormalization(config[Table](1)(1), 1e-3)
        .setName(namePrefix + "1x1/bn"))
      conv1.add(ReLU[D](true).setName(namePrefix + "1x1/bn/sc/relu"))
      concat.add(conv1)
    }

    val conv3 = Sequential[D]
    conv3.add(SpatialConvolution[D](inputSize, config[Table](2)(1), 1, 1, 1, 1)
      .setName(namePrefix + "3x3_reduce"))
    conv3.add(SpatialBatchNormalization(config[Table](2)(1), 1e-3)
      .setName(namePrefix + "3x3_reduce/bn"))
    conv3.add(ReLU[D](true). setName(namePrefix + "3x3_reduce/bn/sc/relu"))
    if(config[Table](4)[String](1) == "max" && config[Table](4)[Int](2) == 0) {
      conv3.add(SpatialConvolution[D](config[Table](2)(1),
        config[Table](2)(2), 3, 3, 2, 2, 1, 1).setName(namePrefix + "3x3"))
    } else {
      conv3.add(SpatialConvolution[D](config[Table](2)(1),
        config[Table](2)(2), 3, 3, 1, 1, 1, 1).setName(namePrefix + "3x3"))
    }
    conv3.add(SpatialBatchNormalization(config[Table](2)(2), 1e-3)
      .setName(namePrefix + "3x3/bn"))
    conv3.add(ReLU[D](true).setName(namePrefix + "3x3/bn/sc/relu"))
    concat.add(conv3)

    val conv3xx = Sequential[D]
    conv3xx.add(SpatialConvolution[D](inputSize, config[Table](3)(1), 1, 1, 1, 1)
      .setName(namePrefix + "double3x3_reduce"))
    conv3xx.add(SpatialBatchNormalization(config[Table](3)(1), 1e-3)
      .setName(namePrefix + "double3x3_reduce/bn"))
    conv3xx.add(ReLU[D](true).setName(namePrefix + "double3x3_reduce/bn/sc/relu"))

    conv3xx.add(SpatialConvolution[D](config[Table](3)(1),
      config[Table](3)(2), 3, 3, 1, 1, 1, 1).setName(namePrefix + "double3x3a"))
    conv3xx.add(SpatialBatchNormalization(config[Table](3)(2), 1e-3)
      .setName(namePrefix + "double3x3a/bn"))
    conv3xx.add(ReLU[D](true).setName(namePrefix + "double3x3a/bn/sc/relu"))

    if(config[Table](4)[String](1) == "max" && config[Table](4)[Int](2) == 0) {
      conv3xx.add(SpatialConvolution[D](config[Table](3)(2),
        config[Table](3)(2), 3, 3, 2, 2, 1, 1).setName(namePrefix + "double3x3b"))
    } else {
      conv3xx.add(SpatialConvolution[D](config[Table](3)(2),
        config[Table](3)(2), 3, 3, 1, 1, 1, 1).setName(namePrefix + "double3x3b"))
    }
    conv3xx.add(SpatialBatchNormalization(config[Table](3)(2), 1e-3)
      .setName(namePrefix + "double3x3b/bn"))
    conv3xx.add(ReLU[D](true).setName(namePrefix + "double3x3b/bn/sc/relu"))
    concat.add(conv3xx)

    val pool = Sequential[D]
    config[Table](4)[String](1) match {
      case "max" =>
        if (config[Table](4)[Int](2) != 0) {
          pool.add(SpatialMaxPooling[D](3, 3, 1, 1, 1, 1).ceil().setName(namePrefix + "pool"))
        } else {
          pool.add(SpatialMaxPooling[D](3, 3, 2, 2).ceil().setName(namePrefix + "pool"))
        }
      case "avg" => pool.add(SpatialAveragePooling[D](3, 3, 1, 1, 1, 1).ceil()
        .setName(namePrefix + "pool"))
      case _ => throw new IllegalArgumentException
    }

    if (config[Table](4)[Int](2) != 0) {
      pool.add(SpatialConvolution[D](inputSize, config[Table](4)[Int](2), 1, 1, 1, 1)
        .setName(namePrefix + "pool_proj"))
      pool.add(SpatialBatchNormalization(config[Table](4)(2), 1e-3)
        .setName(namePrefix + "pool_proj/bn"))
      pool.add(ReLU[D](true).setName(namePrefix + "pool_proj/bn/sc/relu"))
    }
    concat.add(pool)
    concat.setName(namePrefix + "output")
  }
}

object GoogleNet_v2 {
  def apply[D: ClassTag](classNum: Int)
    (implicit ev: TensorNumeric[D]): Module[D] = {
    val features1 = Sequential[D]
    features1.add(SpatialConvolution[D](3, 64, 7, 7, 2, 2, 3, 3, 1, false)
      .setName("conv1/7x7_s2"))
    features1.add(SpatialBatchNormalization(64, 1e-3).setName("conv1/7x7_s2/bn"))
    features1.add(ReLU[D](true).setName("conv1/7x7_s2/bn/sc/relu"))
    features1.add(SpatialMaxPooling[D](3, 3, 2, 2).ceil().setName("pool1/3x3_s2"))
    features1.add(SpatialConvolution[D](64, 64, 1, 1).setName("conv2/3x3_reduce"))
    features1.add(SpatialBatchNormalization(64, 1e-3).setName("conv2/3x3_reduce/bn"))
    features1.add(ReLU[D](true).setName("conv2/3x3_reduce/bn/sc/relu"))
    features1.add(SpatialConvolution[D](64, 192, 3, 3, 1, 1, 1, 1).setName("conv2/3x3"))
    features1.add(SpatialBatchNormalization(192, 1e-3).setName("conv2/3x3/bn"))
    features1.add(ReLU[D](true).setName("conv2/3x3/bn/sc/relu"))
    features1.add(SpatialMaxPooling[D](3, 3, 2, 2).ceil().setName("pool2/3x3_s2"))
    features1.add(inception(192, T(T(64), T(64, 64), T(64, 96), T("avg", 32)), "inception_3a/"))
    features1.add(inception(256, T(T(64), T(64, 96), T(64, 96), T("avg", 64)), "inception_3b/"))
    features1.add(inception(320, T(T(0), T(128, 160), T(64, 96), T("max", 0)), "inception_3c/"))

    val output1 = Sequential[D]
    output1.add(SpatialAveragePooling[D](5, 5, 3, 3).ceil().setName("pool3/5x5_s3"))
    output1.add(SpatialConvolution[D](576, 128, 1, 1, 1, 1).setName("loss1/conv"))
    output1.add(SpatialBatchNormalization(128, 1e-3).setName("loss1/conv/bn"))
    output1.add(ReLU[D](true).setName("loss1/conv/bn/sc/relu"))
    output1.add(View[D](128 * 4 * 4).setNumInputDims(3))
    output1.add(Linear[D](128 * 4 * 4, 1024).setName("loss1/fc"))
    output1.add(ReLU[D](true).setName("loss1/fc/bn/sc/relu"))
    output1.add(Linear[D](1024, classNum).setName("loss1/classifier"))
    output1.add(LogSoftMax[D].setName("loss1/loss"))


    val features2 = Sequential[D]
    features2.add(inception(576, T(T(224), T(64, 96), T(96, 128), T("avg", 128)), "inception_4a/"))
    features2.add(inception(576, T(T(192), T(96, 128), T(96, 128), T("avg", 128)), "inception_4b/"))
    features2.add(inception(576, T(T(160), T(128, 160), T(128, 160), T("avg", 96)),
      "inception_4c/"))
    features2.add(inception(576, T(T(96), T(128, 192), T(160, 192), T("avg", 96)), "inception_4d/"))
    features2.add(inception(576, T(T(0), T(128, 192), T(192, 256), T("max", 0)), "inception_4e/"))

    val output2 = Sequential[D]
    output2.add(SpatialAveragePooling[D](5, 5, 3, 3).ceil().setName("pool4/5x5_s3"))
    output2.add(SpatialConvolution[D](1024, 128, 1, 1, 1, 1).setName("loss2/conv"))
    output2.add(SpatialBatchNormalization(128, 1e-3).setName("loss2/conv/bn"))
    output2.add(ReLU[D](true).setName("loss2/conv/bn/sc/relu"))
    output2.add(View[D](128 * 2 * 2).setNumInputDims(3))
    output2.add(Linear[D](128 * 2 * 2, 1024).setName("loss2/fc"))
    output2.add(ReLU[D](true).setName("loss2/fc/bn/sc/relu"))
    output2.add(Linear[D](1024, classNum).setName("loss2/classifier"))
    output2.add(LogSoftMax[D].setName("loss2/loss"))

    val output3 = Sequential[D]
    output3.add(inception(1024, T(T(352), T(192, 320), T(160, 224), T("avg", 128)),
      "inception_5a/"))
    output3.add(inception(1024, T(T(352), T(192, 320), T(192, 224), T("max", 128)),
      "inception_5b/"))
    output3.add(SpatialAveragePooling[D](7, 7, 1, 1).ceil().setName("pool5/7x7_s1"))
    output3.add(View[D](1024).setNumInputDims(3))
    output3.add(Linear[D](1024, classNum).setName("loss3/classifier"))
    output3.add(LogSoftMax[D].setName("loss3/loss"))

    val split2 = Concat[D](2)
    split2.add(output3)
    split2.add(output2)

    val mainBranch = Sequential[D]()
    mainBranch.add(features2)
    mainBranch.add(split2)

    val split1 = Concat[D](2)
    split1.add(mainBranch)
    split1.add(output1)

    val model = Sequential[D]()

    model.add(features1)
    model.add(split1)

    model.reset()
    model
  }

  def inception[D: ClassTag](inputSize: Int, config: Table, namePrefix : String)(
    implicit ev: TensorNumeric[D]): Module[D] = {
    val concat = Concat[D](2)
    if (config[Table](1)[Int](1) != 0) {
      val conv1 = Sequential[D]
      conv1.add(SpatialConvolution[D](inputSize, config[Table](1)(1), 1, 1, 1, 1)
        .setName(namePrefix + "1x1"))
      conv1.add(SpatialBatchNormalization(config[Table](1)(1), 1e-3)
        .setName(namePrefix + "1x1/bn"))
      conv1.add(ReLU[D](true).setName(namePrefix + "1x1/bn/sc/relu"))
      concat.add(conv1)
    }

    val conv3 = Sequential[D]
    conv3.add(SpatialConvolution[D](inputSize, config[Table](2)(1), 1, 1, 1, 1)
      .setName(namePrefix + "3x3_reduce"))
    conv3.add(SpatialBatchNormalization(config[Table](2)(1), 1e-3)
      .setName(namePrefix + "3x3_reduce/bn"))
    conv3.add(ReLU[D](true). setName(namePrefix + "3x3_reduce/bn/sc/relu"))
    if(config[Table](4)[String](1) == "max" && config[Table](4)[Int](2) == 0) {
      conv3.add(SpatialConvolution[D](config[Table](2)(1),
        config[Table](2)(2), 3, 3, 2, 2, 1, 1).setName(namePrefix + "3x3"))
    } else {
      conv3.add(SpatialConvolution[D](config[Table](2)(1),
        config[Table](2)(2), 3, 3, 1, 1, 1, 1).setName(namePrefix + "3x3"))
    }
    conv3.add(SpatialBatchNormalization(config[Table](2)(2), 1e-3)
      .setName(namePrefix + "3x3/bn"))
    conv3.add(ReLU[D](true).setName(namePrefix + "3x3/bn/sc/relu"))
    concat.add(conv3)

    val conv3xx = Sequential[D]
    conv3xx.add(SpatialConvolution[D](inputSize, config[Table](3)(1), 1, 1, 1, 1)
      .setName(namePrefix + "double3x3_reduce"))
    conv3xx.add(SpatialBatchNormalization(config[Table](3)(1), 1e-3)
      .setName(namePrefix + "double3x3_reduce/bn"))
    conv3xx.add(ReLU[D](true).setName(namePrefix + "double3x3_reduce/bn/sc/relu"))

    conv3xx.add(SpatialConvolution[D](config[Table](3)(1),
      config[Table](3)(2), 3, 3, 1, 1, 1, 1).setName(namePrefix + "double3x3a"))
    conv3xx.add(SpatialBatchNormalization(config[Table](3)(2), 1e-3)
      .setName(namePrefix + "double3x3a/bn"))
    conv3xx.add(ReLU[D](true).setName(namePrefix + "double3x3a/bn/sc/relu"))

    if(config[Table](4)[String](1) == "max" && config[Table](4)[Int](2) == 0) {
      conv3xx.add(SpatialConvolution[D](config[Table](3)(2),
        config[Table](3)(2), 3, 3, 2, 2, 1, 1).setName(namePrefix + "double3x3b"))
    } else {
      conv3xx.add(SpatialConvolution[D](config[Table](3)(2),
        config[Table](3)(2), 3, 3, 1, 1, 1, 1).setName(namePrefix + "double3x3b"))
    }
    conv3xx.add(SpatialBatchNormalization(config[Table](3)(2), 1e-3)
      .setName(namePrefix + "double3x3b/bn"))
    conv3xx.add(ReLU[D](true).setName(namePrefix + "double3x3b/bn/sc/relu"))
    concat.add(conv3xx)

    val pool = Sequential[D]
    config[Table](4)[String](1) match {
      case "max" =>
        if (config[Table](4)[Int](2) != 0) {
          pool.add(SpatialMaxPooling[D](3, 3, 1, 1, 1, 1).ceil().setName(namePrefix + "pool"))
        } else {
          pool.add(SpatialMaxPooling[D](3, 3, 2, 2).ceil().setName(namePrefix + "pool"))
        }
      case "avg" => pool.add(SpatialAveragePooling[D](3, 3, 1, 1, 1, 1).ceil()
        .setName(namePrefix + "pool"))
      case _ => throw new IllegalArgumentException
    }

    if (config[Table](4)[Int](2) != 0) {
      pool.add(SpatialConvolution[D](inputSize, config[Table](4)[Int](2), 1, 1, 1, 1)
        .setName(namePrefix + "pool_proj"))
      pool.add(SpatialBatchNormalization(config[Table](4)(2), 1e-3)
        .setName(namePrefix + "pool_proj/bn"))
      pool.add(ReLU[D](true).setName(namePrefix + "pool_proj/bn/sc/relu"))
    }
    concat.add(pool)
    concat.setName(namePrefix + "output")
  }
}
