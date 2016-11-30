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

/*
 * TODO & Note:
 *
 * 1. because the implementation of SpatialBatchNormalization isn't the
 *    same, so we set comment all of the SpatialBatchNormalization layer.
 * 2. Currently, the output and gradInput of Dnn model and Blas model
 *    are not the same, the error is 1e-4 ~ 1e-5 for output and
 *    1e-4 ~ 1e-5 for gradInput after 10 iterations.
 */

package com.intel.analytics.bigdl.nn.mkl

import com.intel.analytics.bigdl.nn
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}
import org.scalatest.{FlatSpec, Matchers}

import scala.reflect.ClassTag

object GoogleNet_v2Blas {
  def apply[D: ClassTag](classNum: Int)(implicit ev: TensorNumeric[D]): Module[Tensor[D], Tensor[D], D] = {
    val features1 = new Sequential[Tensor[D], Tensor[D], D]
    features1.add(
      new nn.SpatialConvolution[D](3, 64, 7, 7, 2, 2, 3, 3)
        .setName("conv1/7x7_s2")
        .setNeedComputeBack(false)
        .setInitMethod(Xavier))
    features1.add(new nn.SpatialBatchNormalization(64, 1e-3).setName("conv1/7x7_s2/bn"))
    features1.add(new nn.ReLU[D](true).setName("conv1/7x7_s2/bn/sc/relu"))
    features1.add(new nn.SpatialMaxPooling[D](3, 3, 2, 2).ceil().setName("pool1/3x3_s2"))
    features1.add(
      new nn.SpatialConvolution[D](64, 64, 1, 1).setName("conv2/3x3_reduce").setInitMethod(Xavier))
    features1.add(new nn.SpatialBatchNormalization(64, 1e-3).setName("conv2/3x3_reduce/bn"))
    features1.add(new nn.ReLU[D](true).setName("conv2/3x3_reduce/bn/sc/relu"))
    features1.add(
      new nn.SpatialConvolution[D](64, 192, 3, 3, 1, 1, 1, 1)
        .setName("conv2/3x3")
        .setInitMethod(Xavier))
    features1.add(new nn.SpatialBatchNormalization(192, 1e-3).setName("conv2/3x3/bn"))
    features1.add(new nn.ReLU[D](true).setName("conv2/3x3/bn/sc/relu"))
    features1.add(new nn.SpatialMaxPooling[D](3, 3, 2, 2).ceil().setName("pool2/3x3_s2"))
    features1.add(inception(192, T(T(64), T(64, 64), T(64, 96), T("avg", 32)), "inception_3a/"))
    features1.add(inception(256, T(T(64), T(64, 96), T(64, 96), T("avg", 64)), "inception_3b/"))
    features1.add(inception(320, T(T(0), T(128, 160), T(64, 96), T("max", 0)), "inception_3c/"))

    val output1 = new Sequential[Tensor[D], Tensor[D], D]
    output1.add(new nn.SpatialAveragePooling[D](5, 5, 3, 3).ceil().setName("pool3/5x5_s3"))
    output1.add(
      new nn.SpatialConvolution[D](576, 128, 1, 1, 1, 1)
        .setName("loss1/conv")
        .setInitMethod(Xavier))
    output1.add(new nn.SpatialBatchNormalization(128, 1e-3).setName("loss1/conv/bn"))
    output1.add(new nn.ReLU[D](true).setName("loss1/conv/bn/sc/relu"))
    output1.add(new View[D](128 * 4 * 4).setNumInputDims(3))
    output1.add(new nn.Linear[D](128 * 4 * 4, 1024).setName("loss1/fc"))
    output1.add(new nn.ReLU[D](true).setName("loss1/fc/bn/sc/relu"))
    output1.add(new nn.Linear[D](1024, classNum).setName("loss1/classifier"))
    output1.add(new LogSoftMax[D].setName("loss1/loss"))

    val features2 = new Sequential[Tensor[D], Tensor[D], D]
    features2.add(inception(576, T(T(224), T(64, 96), T(96, 128), T("avg", 128)), "inception_4a/"))
    features2.add(
      inception(576, T(T(192), T(96, 128), T(96, 128), T("avg", 128)), "inception_4b/"))
    features2.add(
      inception(576, T(T(160), T(128, 160), T(128, 160), T("avg", 96)), "inception_4c/"))
    features2.add(
      inception(576, T(T(96), T(128, 192), T(160, 192), T("avg", 96)), "inception_4d/"))
    features2.add(inception(576, T(T(0), T(128, 192), T(192, 256), T("max", 0)), "inception_4e/"))

    val output2 = new Sequential[Tensor[D], Tensor[D], D]
    output2.add(new nn.SpatialAveragePooling[D](5, 5, 3, 3).ceil().setName("pool4/5x5_s3"))
    output2.add(
      new nn.SpatialConvolution[D](1024, 128, 1, 1, 1, 1)
        .setName("loss2/conv")
        .setInitMethod(Xavier))
    output2.add(new nn.SpatialBatchNormalization(128, 1e-3).setName("loss2/conv/bn"))
    output2.add(new nn.ReLU[D](true).setName("loss2/conv/bn/sc/relu"))
    output2.add(new View[D](128 * 2 * 2).setNumInputDims(3))
    output2.add(new nn.Linear[D](128 * 2 * 2, 1024).setName("loss2/fc"))
    output2.add(new nn.ReLU[D](true).setName("loss2/fc/bn/sc/relu"))
    output2.add(new nn.Linear[D](1024, classNum).setName("loss2/classifier"))
    output2.add(new LogSoftMax[D].setName("loss2/loss"))

    val output3 = new Sequential[Tensor[D], Tensor[D], D]
    output3.add(
      inception(1024, T(T(352), T(192, 320), T(160, 224), T("avg", 128)), "inception_5a/"))
    output3.add(
      inception(1024, T(T(352), T(192, 320), T(192, 224), T("max", 128)), "inception_5b/"))
    output3.add(new nn.SpatialAveragePooling[D](7, 7, 1, 1).ceil().setName("pool5/7x7_s1"))
    output3.add(new View[D](1024).setNumInputDims(3))
    output3.add(new nn.Linear[D](1024, classNum).setName("loss3/classifier").setInitMethod(Xavier))
    output3.add(new LogSoftMax[D].setName("loss3/loss"))

    val split2 = new nn.Concat[D](2)
    split2.add(output3)
    split2.add(output2)

    val mainBranch = new Sequential[Tensor[D], Tensor[D], D]()
    mainBranch.add(features2)
    mainBranch.add(split2)

    val split1 = new nn.Concat[D](2)
    split1.add(mainBranch)
    split1.add(output1)

    val model = new Sequential[Tensor[D], Tensor[D], D]()

    model.add(features1)
    model.add(split1)

    model.reset()
    model
  }

  def inception[D: ClassTag](inputSize: Int, config: Table, namePrefix: String)(
      implicit ev: TensorNumeric[D]): Module[Tensor[D], Tensor[D], D] = {
    val concat = new nn.Concat[D](2)
    if (config[Table](1)[Int](1) != 0) {
      val conv1 = new Sequential[Tensor[D], Tensor[D], D]
      conv1.add(
        new nn.SpatialConvolution[D](inputSize, config[Table](1)(1), 1, 1, 1, 1)
          .setName(namePrefix + "1x1")
          .setInitMethod(Xavier))
      conv1.add(new nn.SpatialBatchNormalization(config[Table](1)(1), 1e-3)
                  .setName(namePrefix + "1x1/bn"))
      conv1.add(new nn.ReLU[D](true).setName(namePrefix + "1x1/bn/sc/relu"))
      concat.add(conv1)
    }

    val conv3 = new Sequential[Tensor[D], Tensor[D], D]
    conv3.add(
      new nn.SpatialConvolution[D](inputSize, config[Table](2)(1), 1, 1, 1, 1)
        .setName(namePrefix + "3x3_reduce")
        .setInitMethod(Xavier))
    conv3.add(new nn.SpatialBatchNormalization(config[Table](2)(1), 1e-3)
                .setName(namePrefix + "3x3_reduce/bn"))
    conv3.add(new nn.ReLU[D](true).setName(namePrefix + "3x3_reduce/bn/sc/relu"))
    if (config[Table](4)[String](1) == "max" && config[Table](4)[Int](2) == 0) {
      conv3.add(
        new nn.SpatialConvolution[D](config[Table](2)(1), config[Table](2)(2), 3, 3, 2, 2, 1, 1)
          .setName(namePrefix + "3x3")
          .setInitMethod(Xavier))
    } else {
      conv3.add(
        new nn.SpatialConvolution[D](config[Table](2)(1), config[Table](2)(2), 3, 3, 1, 1, 1, 1)
          .setName(namePrefix + "3x3")
          .setInitMethod(Xavier))
    }
    conv3.add(new nn.SpatialBatchNormalization(config[Table](2)(2), 1e-3)
                .setName(namePrefix + "3x3/bn"))
    conv3.add(new nn.ReLU[D](true).setName(namePrefix + "3x3/bn/sc/relu"))
    concat.add(conv3)

    val conv3xx = new Sequential[Tensor[D], Tensor[D], D]
    conv3xx.add(
      new nn.SpatialConvolution[D](inputSize, config[Table](3)(1), 1, 1, 1, 1)
        .setName(namePrefix + "double3x3_reduce")
        .setInitMethod(Xavier))
    conv3xx.add(new nn.SpatialBatchNormalization(config[Table](3)(1), 1e-3)
                  .setName(namePrefix + "double3x3_reduce/bn"))
    conv3xx.add(new nn.ReLU[D](true).setName(namePrefix + "double3x3_reduce/bn/sc/relu"))

    conv3xx.add(
      new nn.SpatialConvolution[D](config[Table](3)(1), config[Table](3)(2), 3, 3, 1, 1, 1, 1)
        .setName(namePrefix + "double3x3a")
        .setInitMethod(Xavier))
    conv3xx.add(new nn.SpatialBatchNormalization(config[Table](3)(2), 1e-3)
                  .setName(namePrefix + "double3x3a/bn"))
    conv3xx.add(new nn.ReLU[D](true).setName(namePrefix + "double3x3a/bn/sc/relu"))

    if (config[Table](4)[String](1) == "max" && config[Table](4)[Int](2) == 0) {
      conv3xx.add(
        new nn.SpatialConvolution[D](config[Table](3)(2), config[Table](3)(2), 3, 3, 2, 2, 1, 1)
          .setName(namePrefix + "double3x3b")
          .setInitMethod(Xavier))
    } else {
      conv3xx.add(
        new nn.SpatialConvolution[D](config[Table](3)(2), config[Table](3)(2), 3, 3, 1, 1, 1, 1)
          .setName(namePrefix + "double3x3b")
          .setInitMethod(Xavier))
    }
    conv3xx.add(new nn.SpatialBatchNormalization(config[Table](3)(2), 1e-3)
                  .setName(namePrefix + "double3x3b/bn"))
    conv3xx.add(new nn.ReLU[D](true).setName(namePrefix + "double3x3b/bn/sc/relu"))
    concat.add(conv3xx)

    val pool = new Sequential[Tensor[D], Tensor[D], D]
    config[Table](4)[String](1) match {
      case "max" =>
        if (config[Table](4)[Int](2) != 0) {
          pool.add(
            new nn.SpatialMaxPooling[D](3, 3, 1, 1, 1, 1).ceil().setName(namePrefix + "pool"))
        } else {
          pool.add(new nn.SpatialMaxPooling[D](3, 3, 2, 2).ceil().setName(namePrefix + "pool"))
        }
      case "avg" =>
        pool.add(
          new SpatialAveragePooling[D](3, 3, 1, 1, 1, 1).ceil().setName(namePrefix + "pool"))
      case _ => throw new IllegalArgumentException
    }

    if (config[Table](4)[Int](2) != 0) {
      pool.add(
        new nn.SpatialConvolution[D](inputSize, config[Table](4)[Int](2), 1, 1, 1, 1)
          .setName(namePrefix + "pool_proj")
          .setInitMethod(Xavier))
      pool.add(new nn.SpatialBatchNormalization(config[Table](4)(2), 1e-3)
                 .setName(namePrefix + "pool_proj/bn"))
      pool.add(new nn.ReLU[D](true).setName(namePrefix + "pool_proj/bn/sc/relu"))
    }
    concat.add(pool)
    concat.setName(namePrefix + "output")
  }
}

object GoogleNet_v2Dnn {
  def apply[D: ClassTag](classNum: Int)(implicit ev: TensorNumeric[D]): Module[Tensor[D], Tensor[D], D] = {
    val features1 = new Sequential[Tensor[D], Tensor[D], D]
    features1.add(
      new SpatialConvolution[D](3, 64, 7, 7, 2, 2, 3, 3)
        .setName("conv1/7x7_s2")
        .setNeedComputeBack(false)
        .setInitMethod(Constant))
    features1.add(new SpatialBatchNormalization(64, 1e-3).setName("conv1/7x7_s2/bn"))
    features1.add(new ReLU[D](true).setName("conv1/7x7_s2/bn/sc/relu"))
    features1.add(new SpatialMaxPooling[D](3, 3, 2, 2).ceil().setName("pool1/3x3_s2"))
    features1.add(
      new SpatialConvolution[D](64, 64, 1, 1).setName("conv2/3x3_reduce").setInitMethod(Constant))
    features1.add(new SpatialBatchNormalization(64, 1e-3).setName("conv2/3x3_reduce/bn"))
    features1.add(new ReLU[D](true).setName("conv2/3x3_reduce/bn/sc/relu"))
    features1.add(
      new SpatialConvolution[D](64, 192, 3, 3, 1, 1, 1, 1)
        .setName("conv2/3x3")
        .setInitMethod(Constant))
    features1.add(new SpatialBatchNormalization(192, 1e-3).setName("conv2/3x3/bn"))
    features1.add(new ReLU[D](true).setName("conv2/3x3/bn/sc/relu"))
    features1.add(new SpatialMaxPooling[D](3, 3, 2, 2).ceil().setName("pool2/3x3_s2"))
    features1.add(inception(192, T(T(64), T(64, 64), T(64, 96), T("avg", 32)), "inception_3a/"))
    features1.add(inception(256, T(T(64), T(64, 96), T(64, 96), T("avg", 64)), "inception_3b/"))
    features1.add(inception(320, T(T(0), T(128, 160), T(64, 96), T("max", 0)), "inception_3c/"))

    val output1 = new Sequential[Tensor[D], Tensor[D], D]
    output1.add(new SpatialAveragePooling[D](5, 5, 3, 3).ceil().setName("pool3/5x5_s3"))
    output1.add(
      new SpatialConvolution[D](576, 128, 1, 1, 1, 1)
        .setName("loss1/conv")
        .setInitMethod(Constant))
    output1.add(new SpatialBatchNormalization(128, 1e-3).setName("loss1/conv/bn"))
    output1.add(new ReLU[D](true).setName("loss1/conv/bn/sc/relu"))
    output1.add(new View[D](128 * 4 * 4).setNumInputDims(3))
    output1.add(new Linear[D](128 * 4 * 4, 1024).setName("loss1/fc").setInitMethod(Constant))
    output1.add(new ReLU[D](true).setName("loss1/fc/bn/sc/relu"))
    output1.add(new Linear[D](1024, classNum).setName("loss1/classifier").setInitMethod(Constant))
    output1.add(new LogSoftMax[D].setName("loss1/loss"))

    val features2 = new Sequential[Tensor[D], Tensor[D], D]
    features2.add(inception(576, T(T(224), T(64, 96), T(96, 128), T("avg", 128)), "inception_4a/"))
    features2.add(
      inception(576, T(T(192), T(96, 128), T(96, 128), T("avg", 128)), "inception_4b/"))
    features2.add(
      inception(576, T(T(160), T(128, 160), T(128, 160), T("avg", 96)), "inception_4c/"))
    features2.add(
      inception(576, T(T(96), T(128, 192), T(160, 192), T("avg", 96)), "inception_4d/"))
    features2.add(inception(576, T(T(0), T(128, 192), T(192, 256), T("max", 0)), "inception_4e/"))

    val output2 = new Sequential[Tensor[D], Tensor[D], D]
    output2.add(new SpatialAveragePooling[D](5, 5, 3, 3).ceil().setName("pool4/5x5_s3"))
    output2.add(
      new SpatialConvolution[D](1024, 128, 1, 1, 1, 1)
        .setName("loss2/conv")
        .setInitMethod(Constant))
    output2.add(new SpatialBatchNormalization(128, 1e-3).setName("loss2/conv/bn"))
    output2.add(new ReLU[D](true).setName("loss2/conv/bn/sc/relu"))
    output2.add(new View[D](128 * 2 * 2).setNumInputDims(3))
    output2.add(new Linear[D](128 * 2 * 2, 1024).setName("loss2/fc").setInitMethod(Constant))
    output2.add(new ReLU[D](true).setName("loss2/fc/bn/sc/relu"))
    output2.add(new Linear[D](1024, classNum).setName("loss2/classifier").setInitMethod(Constant))
    output2.add(new LogSoftMax[D].setName("loss2/loss"))

    val output3 = new Sequential[Tensor[D], Tensor[D], D]
    output3.add(
      inception(1024, T(T(352), T(192, 320), T(160, 224), T("avg", 128)), "inception_5a/"))
    output3.add(
      inception(1024, T(T(352), T(192, 320), T(192, 224), T("max", 128)), "inception_5b/"))
    output3.add(new SpatialAveragePooling[D](7, 7, 1, 1).ceil().setName("pool5/7x7_s1"))
    output3.add(new View[D](1024).setNumInputDims(3))
    output3.add(new Linear[D](1024, classNum).setName("loss3/classifier").setInitMethod(Constant))
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

  def inception[D: ClassTag](inputSize: Int, config: Table, namePrefix: String)(
      implicit ev: TensorNumeric[D]): Module[Tensor[D], Tensor[D], D] = {
    val concat = new nn.Concat[D](2)
    if (config[Table](1)[Int](1) != 0) {
      val conv1 = new Sequential[Tensor[D], Tensor[D], D]
      conv1.add(
        new SpatialConvolution[D](inputSize, config[Table](1)(1), 1, 1, 1, 1)
          .setName(namePrefix + "1x1")
          .setInitMethod(Constant))
      conv1.add(new SpatialBatchNormalization(config[Table](1)(1), 1e-3)
                  .setName(namePrefix + "1x1/bn"))
      conv1.add(new ReLU[D](true).setName(namePrefix + "1x1/bn/sc/relu"))
      concat.add(conv1)
    }

    val conv3 = new Sequential[Tensor[D], Tensor[D], D]
    conv3.add(
      new SpatialConvolution[D](inputSize, config[Table](2)(1), 1, 1, 1, 1)
        .setName(namePrefix + "3x3_reduce")
        .setInitMethod(Constant))
    conv3.add(new SpatialBatchNormalization(config[Table](2)(1), 1e-3)
                .setName(namePrefix + "3x3_reduce/bn"))
    conv3.add(new ReLU[D](true).setName(namePrefix + "3x3_reduce/bn/sc/relu"))
    if (config[Table](4)[String](1) == "max" && config[Table](4)[Int](2) == 0) {
      conv3.add(
        new SpatialConvolution[D](config[Table](2)(1), config[Table](2)(2), 3, 3, 2, 2, 1, 1)
          .setName(namePrefix + "3x3")
          .setInitMethod(Constant))
    } else {
      conv3.add(
        new SpatialConvolution[D](config[Table](2)(1), config[Table](2)(2), 3, 3, 1, 1, 1, 1)
          .setName(namePrefix + "3x3")
          .setInitMethod(Constant))
    }
    conv3.add(new SpatialBatchNormalization(config[Table](2)(2), 1e-3)
                .setName(namePrefix + "3x3/bn"))
    conv3.add(new ReLU[D](true).setName(namePrefix + "3x3/bn/sc/relu"))
    concat.add(conv3)

    val conv3xx = new Sequential[Tensor[D], Tensor[D], D]
    conv3xx.add(
      new SpatialConvolution[D](inputSize, config[Table](3)(1), 1, 1, 1, 1)
        .setName(namePrefix + "double3x3_reduce")
        .setInitMethod(Constant))
    conv3xx.add(new SpatialBatchNormalization(config[Table](3)(1), 1e-3)
                  .setName(namePrefix + "double3x3_reduce/bn"))
    conv3xx.add(new ReLU[D](true).setName(namePrefix + "double3x3_reduce/bn/sc/relu"))

    conv3xx.add(
      new SpatialConvolution[D](config[Table](3)(1), config[Table](3)(2), 3, 3, 1, 1, 1, 1)
        .setName(namePrefix + "double3x3a")
        .setInitMethod(Constant))
    conv3xx.add(new SpatialBatchNormalization(config[Table](3)(2), 1e-3)
                  .setName(namePrefix + "double3x3a/bn"))
    conv3xx.add(new ReLU[D](true).setName(namePrefix + "double3x3a/bn/sc/relu"))

    if (config[Table](4)[String](1) == "max" && config[Table](4)[Int](2) == 0) {
      conv3xx.add(
        new SpatialConvolution[D](config[Table](3)(2), config[Table](3)(2), 3, 3, 2, 2, 1, 1)
          .setName(namePrefix + "double3x3b")
          .setInitMethod(Constant))
    } else {
      conv3xx.add(
        new SpatialConvolution[D](config[Table](3)(2), config[Table](3)(2), 3, 3, 1, 1, 1, 1)
          .setName(namePrefix + "double3x3b")
          .setInitMethod(Constant))
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
      case "avg" =>
        pool.add(
          new SpatialAveragePooling[D](3, 3, 1, 1, 1, 1).ceil().setName(namePrefix + "pool"))
      case _ => throw new IllegalArgumentException
    }

    if (config[Table](4)[Int](2) != 0) {
      pool.add(
        new SpatialConvolution[D](inputSize, config[Table](4)[Int](2), 1, 1, 1, 1)
          .setName(namePrefix + "pool_proj")
          .setInitMethod(Constant))
      pool.add(new SpatialBatchNormalization(config[Table](4)(2), 1e-3)
                 .setName(namePrefix + "pool_proj/bn"))
      pool.add(new ReLU[D](true).setName(namePrefix + "pool_proj/bn/sc/relu"))
    }
    concat.add(pool)
    concat.setName(namePrefix + "output")
  }
}

class GoogLeNetV2Spec extends FlatSpec with Matchers {
  "GoogLeNet generete output and gradient" should "correctly" in {
    def test[T: ClassTag]()(implicit ev: TensorNumeric[T]) {
      val batchSize = 8
      val modelDnn = GoogleNet_v2Dnn(1000)
      val modelBlas = GoogleNet_v2Blas(1000)
      val seqDnn = modelDnn.asInstanceOf[Sequential[Tensor[T], Tensor[T], T]]
      val seqBlas = modelBlas.asInstanceOf[Sequential[Tensor[T], Tensor[T], T]]

      modelDnn.reset()
      modelBlas.reset()
      val paraDnn = modelDnn.parameters()
      val paraBlas = modelBlas.parameters()

      for (i <- 0 until paraDnn._1.length) {
        paraDnn._1(i).copy(paraBlas._1(i))
      }

      val input = Tensor[T](Array(batchSize, 3, 224, 224)).rand()

      val criterionBlas = new ClassNLLCriterion[T]()
      val labelsBlas = Tensor[T](batchSize).fill(ev.fromType(1))
      val criterionDnn = new ClassNLLCriterion[T]()
      val labelsDnn = Tensor[T](batchSize).fill(ev.fromType(1))

      for (i <- 0 until Tools.getRandTimes()) {
        val outputBlas = modelBlas.forward(input)
        criterionBlas.forward(outputBlas, labelsBlas)
        val gradOutputBlas = criterionBlas.backward(outputBlas, labelsBlas)
        val gradInputBlas = modelBlas.backward(input, gradOutputBlas)

        val outputDnn = modelDnn.forward(input)
        criterionDnn.forward(outputDnn, labelsDnn)
        val gradOutputDnn = criterionDnn.backward(outputDnn, labelsDnn)
        val gradInputDnn = modelDnn.backward(input, gradOutputDnn)

        for (i <- 0 until seqBlas.modules.length) {
          Tools.cumulativeError(seqDnn.modules(i).output.asInstanceOf[Tensor[T]],
                                seqBlas.modules(i).output.asInstanceOf[Tensor[T]],
                                "module " + i + " output")
        }

        Tools.cumulativeError(outputDnn, outputBlas, "iteration " + i + " output")
        Tools.cumulativeError(gradOutputBlas, gradOutputDnn, "iteration " + i + " gradoutput")
        Tools.cumulativeError(gradInputBlas, gradInputDnn, "iteration " + i + " gradinput")
      }

      Tools.averageAllTensors(modelBlas.output, "blas output")
      Tools.averageAllTensors(modelDnn.output, "dnn output")
//      Tools.cumulativeError(modelBlas.output, modelDnn.output, "output") should be(0.0 +- 1e-4)
      Tools.averageAllTensors(modelBlas.gradInput, "blas gradinput")
      Tools.averageAllTensors(modelDnn.gradInput, "dnn gradInput")
//      Tools.cumulativeError(modelDnn.gradInput, modelBlas.gradInput, "gradinput") should be(
//        0.0 +- 2 * 1e-4)
    }

    test[Float]()
  }
}
