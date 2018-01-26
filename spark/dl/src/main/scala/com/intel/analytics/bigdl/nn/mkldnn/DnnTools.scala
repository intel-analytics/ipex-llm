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

package com.intel.analytics.bigdl.nn.mkldnn

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn.{Module => _, _}
import com.intel.analytics.bigdl.utils.{T, Table}
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.{Graph, _}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.{T, Table}
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}

import scala.collection.mutable.ArrayBuffer

object DnnTools {
  def getTopTimes(times: Array[(AbstractModule[_ <: Activity, _ <: Activity, Float],
    Long, Long)]): Unit = {
    var forwardSum = 0L
    var backwardSum = 0L
    times.foreach(x => {
      forwardSum += x._2
      backwardSum += x._3
    })
    println(s"forwardSum = ${forwardSum}", s"backwardSum = ${backwardSum}")

    val timeBuffer = new ArrayBuffer[(AbstractModule[_ <: Activity,
      _ <: Activity, Float], Long, Long, Long, Double)]
    var i = 0
    while (i < times.length) {
      val all = times(i)._2 + times(i)._3
      val rate = times(i)._3.toDouble/ times(i)._2
      timeBuffer.append((times(i)._1, times(i)._2, times(i)._3, all, rate))
      i += 1
    }
    val sortData = timeBuffer.sortBy(a => a._4)
    sortData.foreach(println)
  }

  def dnnModel(classNum: Int): Module[Float] = {
    val model = Sequential[Float]()
      .add(ConvolutionDnn(3, 96, 11, 11, 4, 4, propagateBack = false))
      .add(ReLUDnn[Float](false))
      .add(LRNDnn[Float](5, 0.0001, 0.75, 1.0))
      .add(PoolingDnn[Float](3, 3, 2, 2, 0, 0))
    model
  }

  def dnnBigdl(classNum: Int): Module[Float] = {
    val model = Sequential[Float]()
      .add(SpatialConvolution(3, 96, 11, 11, 4, 4, propagateBack = false))
      .add(ReLU[Float](false))
      .add(SpatialCrossMapLRN[Float](5, 0.0001, 0.75, 1.0))
      .add(SpatialMaxPooling[Float](3, 3, 2, 2, 0, 0))
    model
  }

  def dnnLRN(classNum: Int): Module[Float] = {
    val model = Sequential[Float]()
//      .add(ConvolutionDnn(3, 96, 11, 11, 4, 4, propagateBack = false))
//      .add(ReLUDnn[Float](false))
//      // .add(LRNDnn[Float](5, 0.0001, 0.75, 1.0))
//      .add(PoolingDnn[Float](3, 3, 2, 2, 0, 0))
//
//    val feature1 = Sequential()
    model.add(ConvolutionDnn(3, 64, 7, 7, 2, 2, 3, 3, 1, false))
    model.add(ReLUDnn(true).setName("conv1/relu_7x7"))
    model.add(PoolingDnn(3, 3, 2, 2, 0, 0).ceil().setName("pool1/3x3_s2"))

    model
  }

  def dnnTest(classNum: Int): Module[Float] = {
    val model = Sequential[Float]()
      .add(ConvolutionDnn(3, 96, 11, 11, 4, 4, propagateBack = false))
      .add(LRNDnn[Float](5, 0.0001, 0.75, 1.0))
    model
  }

  def dnnOnly(classNum: Int): Module[Float] = {
    val model = Sequential[Float]()
      .add(LRNDnn[Float](5, 0.0001, 0.75, 1.0))
    model
  }
}

object Inception_Layer_v1 {
  def apply(inputSize: Int, config: Table, namePrefix : String = "") : Module[Float] = {
    val feature1 = Sequential()
    val concat = Concat(2)
    val conv1 = Sequential()
    conv1.add(ConvolutionDnn(inputSize,
      config[Table](1)(1), 1, 1, 1, 1)
      .setInitMethod(weightInitMethod = Xavier, Zeros).setName(namePrefix + "1x1"))
    conv1.add(ReLUDnn(true).setName(namePrefix + "relu_1x1"))
    conv1.add(MemoryReOrderForGradoutput(8, 5))
    concat.add(conv1)
    val conv3 = Sequential()
    conv3.add(ConvolutionDnn(inputSize,
      config[Table](2)(1), 1, 1, 1, 1)
      .setInitMethod(weightInitMethod = Xavier, Zeros).setName(namePrefix + "3x3_reduce"))
    conv3.add(ReLUDnn(true).setName(namePrefix + "relu_3x3_reduce"))
    conv3.add(ConvolutionDnn(config[Table](2)(1),
      config[Table](2)(2), 3, 3, 1, 1, 1, 1)
      .setInitMethod(weightInitMethod = Xavier, Zeros).setName(namePrefix + "3x3"))
    conv3.add(ReLUDnn(true).setName(namePrefix + "relu_3x3"))
    conv3.add(MemoryReOrderForGradoutput(8, 5))
    concat.add(conv3)
    val conv5 = Sequential()
    conv5.add(ConvolutionDnn(inputSize,
      config[Table](3)(1), 1, 1, 1, 1)
      .setInitMethod(weightInitMethod = Xavier, Zeros).setName(namePrefix + "5x5_reduce"))
    conv5.add(ReLUDnn(true).setName(namePrefix + "relu_5x5_reduce"))
    conv5.add(ConvolutionDnn(config[Table](3)(1),
      config[Table](3)(2), 5, 5, 1, 1, 2, 2)
      .setInitMethod(weightInitMethod = Xavier, Zeros).setName(namePrefix + "5x5"))
    conv5.add(ReLUDnn(true).setName(namePrefix + "relu_5x5"))
    conv5.add(MemoryReOrderForGradoutput(8, 5))
    concat.add(conv5)
    val pool = Sequential()
    pool.add(PoolingDnn(3, 3, 1, 1, 1, 1).ceil().setName(namePrefix + "pool"))
    pool.add(ConvolutionDnn(inputSize,
      config[Table](4)(1), 1, 1, 1, 1)
      .setInitMethod(weightInitMethod = Xavier, Zeros).setName(namePrefix + "pool_proj"))
    pool.add(ReLUDnn(true).setName(namePrefix + "relu_pool_proj"))
    pool.add(MemoryReOrderForGradoutput(8, 5))
    concat.add(pool).setName(namePrefix + "output")
    feature1.add(concat)
      .add(MemoryReOrder(5, 8))
    feature1
  }
}

object Inception_v1 {
  def apply(classNum: Int, hasDropout: Boolean = true): Module[Float] = {
    val feature1 = Sequential()
    feature1.add(ConvolutionDnn(3, 64, 7, 7, 2, 2, 3, 3, 1, false)
      .setInitMethod(weightInitMethod = Xavier, Zeros)
      .setName("conv1/7x7_s2"))
    feature1.add(ReLUDnn(true).setName("conv1/relu_7x7"))
    feature1.add(PoolingDnn(3, 3, 2, 2).ceil().setName("pool1/3x3_s2"))
    feature1.add(LRNDnn(5, 0.0001, 0.75).setName("pool1/norm1"))
    feature1.add(ConvolutionDnn(64, 64, 1, 1, 1, 1)
      .setInitMethod(weightInitMethod = Xavier, Zeros)
      .setName("conv2/3x3_reduce"))
    feature1.add(ReLUDnn(true).setName("conv2/relu_3x3_reduce"))
    feature1.add(ConvolutionDnn(64, 192, 3, 3, 1, 1, 1, 1)
      .setInitMethod(weightInitMethod = Xavier, Zeros)
      .setName("conv2/3x3"))
    feature1.add(ReLUDnn(true).setName("conv2/relu_3x3"))
    feature1.add(LRNDnn(5, 0.0001, 0.75). setName("conv2/norm2"))
    feature1.add(PoolingDnn(3, 3, 2, 2).ceil().setName("pool2/3x3_s2"))
    feature1.add(Inception_Layer_v1(192, T(T(64), T(96, 128), T(16, 32), T(32)), "inception_3a/"))
    feature1.add(Inception_Layer_v1(256, T(T(128), T(128, 192), T(32, 96), T(64)), "inception_3b/"))
    feature1.add(PoolingDnn(3, 3, 2, 2).ceil().setName("pool3/3x3_s2"))
    feature1.add(Inception_Layer_v1(480, T(T(192), T(96, 208), T(16, 48), T(64)), "inception_4a/"))

    val output1 = Sequential()
    output1.add(PoolingDnnAverage(5, 5, 3, 3).ceil().setName("loss1/ave_pool"))
    output1.add(ConvolutionDnn(512, 128, 1, 1, 1, 1).setName("loss1/conv"))
    output1.add(ReLUDnn(true).setName("loss1/relu_conv"))
    output1.add(View(128 * 4 * 4).setNumInputDims(3))
    output1.add(mkldnn.Linear(128 * 4 * 4, 1024).setName("loss1/fc"))
    output1.add(ReLUDnn(true).setName("loss1/relu_fc"))
    if (hasDropout) output1.add(Dropout(0.7).setName("loss1/drop_fc"))
    output1.add(mkldnn.Linear(1024, classNum).setName("loss1/classifier"))
    output1.add(LogSoftMax().setName("loss1/loss"))

    val feature2 = Sequential()
    feature2.add(Inception_Layer_v1(512, T(T(160), T(112, 224), T(24, 64), T(64)), "inception_4b/"))
    feature2.add(Inception_Layer_v1(512, T(T(128), T(128, 256), T(24, 64), T(64)), "inception_4c/"))
    feature2.add(Inception_Layer_v1(512, T(T(112), T(144, 288), T(32, 64), T(64)), "inception_4d/"))

    val output2 = Sequential()
    output2.add(PoolingDnnAverage(5, 5, 3, 3).setName("loss2/ave_pool"))
    output2.add(ConvolutionDnn(528, 128, 1, 1, 1, 1).setName("loss2/conv"))
    output2.add(ReLUDnn(true).setName("loss2/relu_conv"))
    output2.add(View(128 * 4 * 4).setNumInputDims(3))
    output2.add(mkldnn.Linear(128 * 4 * 4, 1024).setName("loss2/fc"))
    output2.add(ReLUDnn(true).setName("loss2/relu_fc"))
    if (hasDropout) output2.add(Dropout(0.7).setName("loss2/drop_fc"))
    output2.add(mkldnn.Linear(1024, classNum).setName("loss2/classifier"))
    output2.add(LogSoftMax().setName("loss2/loss"))

    val output3 = Sequential()
    output3.add(Inception_Layer_v1(528, T(T(256), T(160, 320), T(32, 128), T(128)),
      "inception_4e/"))
    output3.add(PoolingDnn(3, 3, 2, 2).ceil().setName("pool4/3x3_s2"))
    output3.add(Inception_Layer_v1(832, T(T(256), T(160, 320), T(32, 128), T(128)),
      "inception_5a/"))
    output3.add(Inception_Layer_v1(832, T(T(384), T(192, 384), T(48, 128), T(128)),
      "inception_5b/"))
    output3.add(PoolingDnnAverage(7, 7, 1, 1).setName("pool5/7x7_s1"))
    if (hasDropout) output3.add(Dropout(0.4).setName("pool5/drop_7x7_s1"))
    output3.add(View(1024).setNumInputDims(3))
    output3.add(mkldnn.Linear(1024, classNum)
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
}
