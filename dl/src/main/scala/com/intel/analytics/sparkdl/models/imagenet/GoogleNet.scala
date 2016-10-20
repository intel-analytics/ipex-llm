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

package com.intel.analytics.sparkdl.models.imagenet

import com.intel.analytics.sparkdl.nn._
import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.utils.{T, Table}

import scala.reflect.ClassTag

object GoogleNet_v1 {
  private def inception[D: ClassTag](inputSize: Int, config: Table, namePrefix : String)(
    implicit ev: TensorNumeric[D]): Module[Tensor[D], Tensor[D], D] = {
    val concat = new Concat[D](2)
    val conv1 = new Sequential[Tensor[D], Tensor[D], D]
    conv1.add(new SpatialConvolution[D](inputSize,
      config[Table](1)(1), 1, 1, 1, 1).setInitMethod(Xavier).setName(namePrefix + "1x1"))
    conv1.add(new ReLU[D](true).setName(namePrefix + "relu_1x1"))
    concat.add(conv1)
    val conv3 = new Sequential[Tensor[D], Tensor[D], D]
    conv3.add(new SpatialConvolution[D](inputSize,
      config[Table](2)(1), 1, 1, 1, 1).setInitMethod(Xavier).setName(namePrefix + "3x3_reduce"))
    conv3.add(new ReLU[D](true).setName(namePrefix + "relu_3x3_reduce"))
    conv3.add(new SpatialConvolution[D](config[Table](2)(1),
      config[Table](2)(2), 3, 3, 1, 1, 1, 1).setInitMethod(Xavier).setName(namePrefix + "3x3"))
    conv3.add(new ReLU[D](true).setName(namePrefix + "relu_3x3"))
    concat.add(conv3)
    val conv5 = new Sequential[Tensor[D], Tensor[D], D]
    conv5.add(new SpatialConvolution[D](inputSize,
      config[Table](3)(1), 1, 1, 1, 1).setInitMethod(Xavier).setName(namePrefix + "5x5_reduce"))
    conv5.add(new ReLU[D](true).setName(namePrefix + "relu_5x5_reduce"))
    conv5.add(new SpatialConvolution[D](config[Table](3)(1),
      config[Table](3)(2), 5, 5, 1, 1, 2, 2).setInitMethod(Xavier).setName(namePrefix + "5x5"))
    conv5.add(new ReLU[D](true).setName(namePrefix + "relu_5x5"))
    concat.add(conv5)
    val pool = new Sequential[Tensor[D], Tensor[D], D]
    pool.add(new SpatialMaxPooling[D](3, 3, 1, 1, 1, 1).ceil().setName(namePrefix + "pool"))
    pool.add(new SpatialConvolution[D](inputSize,
      config[Table](4)(1), 1, 1, 1, 1).setInitMethod(Xavier).setName(namePrefix + "pool_proj"))
    pool.add(new ReLU[D](true).setName(namePrefix + "relu_pool_proj"))
    concat.add(pool).setName(namePrefix + "output")
    concat
  }

  def apply[D: ClassTag](classNum: Int)(implicit ev: TensorNumeric[D]): Module[Tensor[D], Tensor[D], D] = {
    val feature1 = new Sequential[Tensor[D], Tensor[D], D]
    feature1.add(new SpatialConvolution[D](3, 64, 7, 7, 2, 2, 3, 3, 1, false).setInitMethod(Xavier)
      .setName("conv1/7x7_s2"))
    feature1.add(new ReLU[D](true).setName("conv1/relu_7x7"))
    feature1.add(new SpatialMaxPooling[D](3, 3, 2, 2).ceil().setName("pool1/3x3_s2"))
    feature1.add(new SpatialCrossMapLRN[D](5, 0.0001, 0.75).setName("pool1/norm1"))
    feature1.add(new SpatialConvolution[D](64, 64, 1, 1, 1, 1).setInitMethod(Xavier)
      .setName("conv2/3x3_reduce"))
    feature1.add(new ReLU[D](true).setName("conv2/relu_3x3_reduce"))
    feature1.add(new SpatialConvolution[D](64, 192, 3, 3, 1, 1, 1, 1).setInitMethod(Xavier)
      .setName("conv2/3x3"))
    feature1.add(new ReLU[D](true).setName("conv2/relu_3x3"))
    feature1.add(new SpatialCrossMapLRN[D](5, 0.0001, 0.75). setName("conv2/norm2"))
    feature1.add(new SpatialMaxPooling[D](3, 3, 2, 2).ceil().setName("pool2/3x3_s2"))
    feature1.add(inception[D](192, T(T(64), T(96, 128), T(16, 32), T(32)), "inception_3a/"))
    feature1.add(inception[D](256, T(T(128), T(128, 192), T(32, 96), T(64)), "inception_3b/"))
    feature1.add(new SpatialMaxPooling[D](3, 3, 2, 2).ceil().setName("pool3/3x3_s2"))
    feature1.add(inception[D](480, T(T(192), T(96, 208), T(16, 48), T(64)), "inception_4a/"))

    val output1 = new Sequential[Tensor[D], Tensor[D], D]
    output1.add(new SpatialAveragePooling[D](5, 5, 3, 3).ceil().setName("loss1/ave_pool"))
    output1.add(new SpatialConvolution[D](512, 128, 1, 1, 1, 1).setName("loss1/conv"))
    output1.add(new ReLU[D](true).setName("loss1/relu_conv"))
    output1.add(new View[D](128 * 4 * 4).setNumInputDims(3))
    output1.add(new Linear[D](128 * 4 * 4, 1024).setName("loss1/fc"))
    output1.add(new ReLU[D](true).setName("loss1/relu_fc"))
    output1.add(new Dropout[D](0.7).setName("loss1/drop_fc"))
    output1.add(new Linear[D](1024, classNum).setName("loss1/classifier"))
    output1.add(new LogSoftMax[D].setName("loss1/loss"))

    val feature2 = new Sequential[Tensor[D], Tensor[D], D]
    feature2.add(inception[D](512, T(T(160), T(112, 224), T(24, 64), T(64)), "inception_4b/"))
    feature2.add(inception[D](512, T(T(128), T(128, 256), T(24, 64), T(64)), "inception_4c/"))
    feature2.add(inception[D](512, T(T(112), T(144, 288), T(32, 64), T(64)), "inception_4d/"))

    val output2 = new Sequential[Tensor[D], Tensor[D], D]
    output2.add(new SpatialAveragePooling[D](5, 5, 3, 3).setName("loss2/ave_pool"))
    output2.add(new SpatialConvolution[D](528, 128, 1, 1, 1, 1).setName("loss2/conv"))
    output2.add(new ReLU[D](true).setName("loss2/relu_conv"))
    output2.add(new View[D](128 * 4 * 4).setNumInputDims(3))
    output2.add(new Linear[D](128 * 4 * 4, 1024).setName("loss2/fc"))
    output2.add(new ReLU[D](true).setName("loss2/relu_fc"))
    output2.add(new Dropout[D](0.7).setName("loss2/drop_fc"))
    output2.add(new Linear[D](1024, classNum).setName("loss2/classifier"))
    output2.add(new LogSoftMax[D].setName("loss2/loss"))

    val output3 = new Sequential[Tensor[D], Tensor[D], D]
    output3.add(inception[D](528, T(T(256), T(160, 320), T(32, 128), T(128)), "inception_4e/"))
    output3.add(new SpatialMaxPooling[D](3, 3, 2, 2).ceil().setName("pool4/3x3_s2"))
    output3.add(inception[D](832, T(T(256), T(160, 320), T(32, 128), T(128)), "inception_5a/"))
    output3.add(inception[D](832, T(T(384), T(192, 384), T(48, 128), T(128)), "inception_5b/"))
    output3.add(new SpatialAveragePooling[D](7, 7, 1, 1).setName("pool5/7x7_s1"))
    output3.add(new Dropout[D](0.4).setName("pool5/drop_7x7_s1"))
    output3.add(new View[D](1024).setNumInputDims(3))
    output3.add(new Linear[D](1024, classNum).setInitMethod(Xavier).setName("loss3/classifier"))
    output3.add(new LogSoftMax[D].setName("loss3/loss3"))

    val split2 = new Concat[D](2).setName("split2")
    split2.add(output3)
    split2.add(output2)

    val mainBranch = new Sequential[Tensor[D], Tensor[D], D]()
    mainBranch.add(feature2)
    mainBranch.add(split2)

    val split1 = new Concat[D](2).setName("split1")
    split1.add(mainBranch)
    split1.add(output1)

    val model = new Sequential[Tensor[D], Tensor[D], D]()

    model.add(feature1)
    model.add(split1)

    model.reset()
    model
  }
}

object GoogleNet_v2 {
  def apply[D: ClassTag](classNum: Int)(implicit ev: TensorNumeric[D]): Module[Tensor[D], Tensor[D], D] = {
    val features1 = new Sequential[Tensor[D], Tensor[D], D]
    features1.add(new SpatialConvolution[D](3, 64, 7, 7, 2, 2, 3, 3, 1, false)
      .setName("conv1/7x7_s2"))
    features1.add(new SpatialBatchNormalization(64, 1e-3).setName("conv1/7x7_s2/bn"))
    features1.add(new ReLU[D](true).setName("conv1/7x7_s2/bn/sc/relu"))
    features1.add(new SpatialMaxPooling[D](3, 3, 2, 2).ceil().setName("pool1/3x3_s2"))
    features1.add(new SpatialConvolution[D](64, 64, 1, 1).setName("conv2/3x3_reduce"))
    features1.add(new SpatialBatchNormalization(64, 1e-3).setName("conv2/3x3_reduce/bn"))
    features1.add(new ReLU[D](true).setName("conv2/3x3_reduce/bn/sc/relu"))
    features1.add(new SpatialConvolution[D](64, 192, 3, 3, 1, 1, 1, 1).setName("conv2/3x3"))
    features1.add(new SpatialBatchNormalization(192, 1e-3).setName("conv2/3x3/bn"))
    features1.add(new ReLU[D](true).setName("conv2/3x3/bn/sc/relu"))
    features1.add(new SpatialMaxPooling[D](3, 3, 2, 2).ceil().setName("pool2/3x3_s2"))
    features1.add(inception(192, T(T(64), T(64, 64), T(64, 96), T("avg", 32)), "inception_3a/"))
    features1.add(inception(256, T(T(64), T(64, 96), T(64, 96), T("avg", 64)), "inception_3b/"))
    features1.add(inception(320, T(T(0), T(128, 160), T(64, 96), T("max", 0)), "inception_3c/"))

    val output1 = new Sequential[Tensor[D], Tensor[D], D]
    output1.add(new SpatialAveragePooling[D](5, 5, 3, 3).ceil().setName("pool3/5x5_s3"))
    output1.add(new SpatialConvolution[D](576, 128, 1, 1, 1, 1).setName("loss1/conv"))
    output1.add(new SpatialBatchNormalization(128, 1e-3).setName("loss1/conv/bn"))
    output1.add(new ReLU[D](true).setName("loss1/conv/bn/sc/relu"))
    output1.add(new View[D](128 * 4 * 4).setNumInputDims(3))
    output1.add(new Linear[D](128 * 4 * 4, 1024).setName("loss1/fc"))
    output1.add(new ReLU[D](true).setName("loss1/fc/bn/sc/relu"))
    output1.add(new Linear[D](1024, classNum).setName("loss1/classifier"))
    output1.add(new LogSoftMax[D].setName("loss1/loss"))


    val features2 = new Sequential[Tensor[D], Tensor[D], D]
    features2.add(inception(576, T(T(224), T(64, 96), T(96, 128), T("avg", 128)), "inception_4a/"))
    features2.add(inception(576, T(T(192), T(96, 128), T(96, 128), T("avg", 128)), "inception_4b/"))
    features2.add(inception(576, T(T(160), T(128, 160), T(128, 160), T("avg", 96)),
      "inception_4c/"))
    features2.add(inception(576, T(T(96), T(128, 192), T(160, 192), T("avg", 96)), "inception_4d/"))
    features2.add(inception(576, T(T(0), T(128, 192), T(192, 256), T("max", 0)), "inception_4e/"))

    val output2 = new Sequential[Tensor[D], Tensor[D], D]
    output2.add(new SpatialAveragePooling[D](5, 5, 3, 3).ceil().setName("pool4/5x5_s3"))
    output2.add(new SpatialConvolution[D](1024, 128, 1, 1, 1, 1).setName("loss2/conv"))
    output2.add(new SpatialBatchNormalization(128, 1e-3).setName("loss2/conv/bn"))
    output2.add(new ReLU[D](true).setName("loss2/conv/bn/sc/relu"))
    output2.add(new View[D](128 * 2 * 2).setNumInputDims(3))
    output2.add(new Linear[D](128 * 2 * 2, 1024).setName("loss2/fc"))
    output2.add(new ReLU[D](true).setName("loss2/fc/bn/sc/relu"))
    output2.add(new Linear[D](1024, classNum).setName("loss2/classifier"))
    output2.add(new LogSoftMax[D].setName("loss2/loss"))

    val output3 = new Sequential[Tensor[D], Tensor[D], D]
    output3.add(inception(1024, T(T(352), T(192, 320), T(160, 224), T("avg", 128)),
      "inception_5a/"))
    output3.add(inception(1024, T(T(352), T(192, 320), T(192, 224), T("max", 128)),
      "inception_5b/"))
    output3.add(new SpatialAveragePooling[D](7, 7, 1, 1).ceil().setName("pool5/7x7_s1"))
    output3.add(new View[D](1024).setNumInputDims(3))
    output3.add(new Linear[D](1024, classNum).setName("loss3/classifier"))
    output3.add(new LogSoftMax[D].setName("loss3/loss"))

    val split2 = new Concat[D](2)
    split2.add(output3)
    split2.add(output2)

    val mainBranch = new Sequential[Tensor[D], Tensor[D], D]()
    mainBranch.add(features2)
    mainBranch.add(split2)

    val split1 = new Concat[D](2)
    split1.add(mainBranch)
    split1.add(output1)

    val model = new Sequential[Tensor[D], Tensor[D], D]()

    model.add(features1)
    model.add(split1)

    model.reset()
    model
  }

  def inception[D: ClassTag](inputSize: Int, config: Table, namePrefix : String)(
    implicit ev: TensorNumeric[D]): Module[Tensor[D], Tensor[D], D] = {
    val concat = new Concat[D](2)
    if (config[Table](1)[Int](1) != 0) {
      val conv1 = new Sequential[Tensor[D], Tensor[D], D]
      conv1.add(new SpatialConvolution[D](inputSize, config[Table](1)(1), 1, 1, 1, 1)
        .setName(namePrefix + "1x1"))
      conv1.add(new SpatialBatchNormalization(config[Table](1)(1), 1e-3)
        .setName(namePrefix + "1x1/bn"))
      conv1.add(new ReLU[D](true).setName(namePrefix + "1x1/bn/sc/relu"))
      concat.add(conv1)
    }

    val conv3 = new Sequential[Tensor[D], Tensor[D], D]
    conv3.add(new SpatialConvolution[D](inputSize, config[Table](2)(1), 1, 1, 1, 1)
      .setName(namePrefix + "3x3_reduce"))
    conv3.add(new SpatialBatchNormalization(config[Table](2)(1), 1e-3)
      .setName(namePrefix + "3x3_reduce/bn"))
    conv3.add(new ReLU[D](true). setName(namePrefix + "3x3_reduce/bn/sc/relu"))
    if(config[Table](4)[String](1) == "max" && config[Table](4)[Int](2) == 0) {
      conv3.add(new SpatialConvolution[D](config[Table](2)(1),
        config[Table](2)(2), 3, 3, 2, 2, 1, 1).setName(namePrefix + "3x3"))
    } else {
      conv3.add(new SpatialConvolution[D](config[Table](2)(1),
        config[Table](2)(2), 3, 3, 1, 1, 1, 1).setName(namePrefix + "3x3"))
    }
    conv3.add(new SpatialBatchNormalization(config[Table](2)(2), 1e-3)
      .setName(namePrefix + "3x3/bn"))
    conv3.add(new ReLU[D](true).setName(namePrefix + "3x3/bn/sc/relu"))
    concat.add(conv3)

    val conv3xx = new Sequential[Tensor[D], Tensor[D], D]
    conv3xx.add(new SpatialConvolution[D](inputSize, config[Table](3)(1), 1, 1, 1, 1)
      .setName(namePrefix + "double3x3_reduce"))
    conv3xx.add(new SpatialBatchNormalization(config[Table](3)(1), 1e-3)
      .setName(namePrefix + "double3x3_reduce/bn"))
    conv3xx.add(new ReLU[D](true).setName(namePrefix + "double3x3_reduce/bn/sc/relu"))

    conv3xx.add(new SpatialConvolution[D](config[Table](3)(1),
      config[Table](3)(2), 3, 3, 1, 1, 1, 1).setName(namePrefix + "double3x3a"))
    conv3xx.add(new SpatialBatchNormalization(config[Table](3)(2), 1e-3)
      .setName(namePrefix + "double3x3a/bn"))
    conv3xx.add(new ReLU[D](true).setName(namePrefix + "double3x3a/bn/sc/relu"))

    if(config[Table](4)[String](1) == "max" && config[Table](4)[Int](2) == 0) {
      conv3xx.add(new SpatialConvolution[D](config[Table](3)(2),
        config[Table](3)(2), 3, 3, 2, 2, 1, 1).setName(namePrefix + "double3x3b"))
    } else {
      conv3xx.add(new SpatialConvolution[D](config[Table](3)(2),
        config[Table](3)(2), 3, 3, 1, 1, 1, 1).setName(namePrefix + "double3x3b"))
    }
    conv3xx.add(new SpatialBatchNormalization(config[Table](3)(2), 1e-3)
      .setName(namePrefix + "double3x3b/bn"))
    conv3xx.add(new ReLU[D](true).setName(namePrefix + "double3x3b/bn/sc/relu"))
    concat.add(conv3xx)

    val pool = new Sequential[Tensor[D], Tensor[D], D]
    config[Table](4)[String](1) match {
      case "max" =>
        if (config[Table](4)[Int](2) != 0) {
          pool.add(new SpatialMaxPooling[D](3, 3, 1, 1, 1, 1).ceil().setName(namePrefix + "pool"))
        } else {
          pool.add(new SpatialMaxPooling[D](3, 3, 2, 2).ceil().setName(namePrefix + "pool"))
        }
      case "avg" => pool.add(new SpatialAveragePooling[D](3, 3, 1, 1, 1, 1).ceil()
        .setName(namePrefix + "pool"))
      case _ => throw new IllegalArgumentException
    }

    if (config[Table](4)[Int](2) != 0) {
      pool.add(new SpatialConvolution[D](inputSize, config[Table](4)[Int](2), 1, 1, 1, 1)
        .setName(namePrefix + "pool_proj"))
      pool.add(new SpatialBatchNormalization(config[Table](4)(2), 1e-3)
        .setName(namePrefix + "pool_proj/bn"))
      pool.add(new ReLU[D](true).setName(namePrefix + "pool_proj/bn/sc/relu"))
    }
    concat.add(pool)
    concat.setName(namePrefix + "output")
  }
}
