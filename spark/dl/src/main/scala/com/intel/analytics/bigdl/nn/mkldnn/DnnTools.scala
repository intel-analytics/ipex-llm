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

import breeze.numerics._
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn.{Graph, Linear, SpatialBatchNormalization, Module => _, _}
import com.intel.analytics.bigdl.utils.{T, Table}
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.{T, Table}
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.mkl.MklDnn
import com.intel.analytics.bigdl.models.inception.Inception_Layer_v2
import com.intel.analytics.bigdl.models.resnet.Convolution
import com.intel.analytics.bigdl.models.resnet.ResNet.{apply => _, _}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.optim.L2Regularizer
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{DenseTensorMath, Storage, Tensor}
import com.intel.analytics.bigdl.utils.RandomGenerator._
import org.apache.log4j.Logger

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

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

  def debugFwInfo(name : String, time: Double, format1: Int, format2: Int): Unit = {
    println(s"${name} forward ${time} input ${format1} output ${format2}")
  }

  def debugBwInfo(name : String, time: Double, format1: Int, format2: Int): Unit = {
    println(s"${name} backward ${time} gradOutput ${format1} gradInput ${format2}")
  }

  def dnnModel(classNum: Int): Module[Float] = {
    val model = Sequential[Float]()
      .add(ConvolutionDnn(3, 96, 11, 11, 4, 4, propagateBack = false))
      .add(ReLUDnn[Float](false))
      .add(LRNDnn[Float](5, 0.0001, 0.75, 1.0))
      .add(PoolingDnn[Float](3, 3, 2, 2, 0, 0))
    model
  }

  def reorderTwoTensor(input: Tensor[Float], inputFormat: Int,
                       output: Tensor[Float], outputFormat: Int): Unit = {
    val dataType = MklDnn.DataType.f32
    val engine = MklDnn.EngineCreate( MklDnn.EngineType.cpu, 0)
    val stream = MklDnn.StreamCreate(MklDnn.StreamType.eager)
    val stream_fwd = new ArrayBuffer[Long]
    val sizes = input.size()
    val dim = input.dim()
    output.resizeAs(input)

    // val src_pd = input.getPrimitiveDesc()
    val src_md = MklDnnOps.memoryDescInit(dim, sizes, dataType, inputFormat)
    val src_pd = MklDnnOps.memoryPrimitiveDescCreate(src_md, engine)

    val dst_memory = MklDnnOps.initDataMemory(dim, sizes, outputFormat, dataType, engine)
    val res = MklDnnOps.prepareReorder(dst_memory, src_pd, false)
    // val reorder_primitive = res._1
    val src_memory = res._2

    stream_fwd.clear()
    stream_fwd.append(res._1)

    /* build a simple net */
    val memoryPrimitives = Array(src_memory, dst_memory)
    val buffer = Array(input, output)
    MklDnnOps.streamSubmit(stream, 1, stream_fwd.toArray, 1, memoryPrimitives, buffer)
  }

  def nearlyEqual(a: Float, b: Float, epsilon: Double): Boolean = {
    val absA = math.abs(a)
    val absB = math.abs(b)
    val diff = math.abs(a - b)

    val result = if (a == b) {
      true
    } else {
      math.min(diff / (absA + absB), diff) < epsilon
    }

    result
  }

  def nearequals(t1: Tensor[Float], t2: Tensor[Float],
                 epsilon: Double = DenseTensorMath.floatEpsilon): Boolean = {
    var result = true
    t1.map(t2, (a, b) => {
      if (result) {
        result = nearlyEqual(a, b, epsilon)
        if (!result) {
          val diff = math.abs(a - b)
          println("epsilon " + a + "***" + b + "***" + diff / (abs(a) + abs(b)) + "***" + diff)
        }
      }
      a
    })
    return result
  }

  def getunequals(t1: Tensor[Float], t2: Tensor[Float],
                  epsilon: Double = DenseTensorMath.floatEpsilon): Boolean = {
    var result = true
    var num = 0
    t1.map(t2, (a, b) => {
      if (true) {
        result = nearlyEqual(a, b, epsilon)
        if (!result) {
          num += 1
          val diff = math.abs(a - b)
          println("epsilon " + a + "***" + b + "***" + diff / (abs(a) + abs(b)) + "***" + diff)
        }
      }
      a
    })
    println("diff num " + num)
    return true
  }

  def isEquals(t1: Tensor[Float], t2: Tensor[Float]): Boolean = {
    var result = true
    t1.map(t2, (a, b) => {
      if (result) {
        result = if (a == b) true else false
        if (!result) {
          val diff = math.abs(a - b)
          println("epsilon " + a + "***" + b + "***" + diff / (abs(a) + abs(b)) + "***" + diff)
        }
      }
      a
    })
    return result
  }
}

object Inception_Layer_v1 {
  def apply(inputSize: Int, config: Table, namePrefix : String = "", format: Int = 8) :
  Module[Float] = {
    val concat = ConcatDnn(2)
    val conv1 = Sequential()
    conv1.add(ConvolutionDnn(inputSize,
      config[Table](1)(1), 1, 1, 1, 1)
      .setInitMethod(weightInitMethod = Xavier, Zeros).setName(namePrefix + "1x1"))
    conv1.add(ReLUDnn(true).setName(namePrefix + "relu_1x1"))
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
    concat.add(conv5)

    val pool = Sequential()
    pool.add(PoolingDnn(3, 3, 1, 1, 1, 1).ceil().setName(namePrefix + "pool"))
    pool.add(ConvolutionDnn(inputSize,
      config[Table](4)(1), 1, 1, 1, 1)
      .setInitMethod(weightInitMethod = Xavier, Zeros).setName(namePrefix + "pool_proj"))
    pool.add(ReLUDnn(true).setName(namePrefix + "relu_pool_proj"))
    concat.add(pool).setName(namePrefix + "concat_output")

    concat
  }
}

object Inception_Layer_v2 {
  def apply(inputSize: Int, config: Table, namePrefix : String): Module[Float] = {
    val concat = ConcatDnn(2)
    if (config[Table](1)[Int](1) != 0) {
      val conv1 = Sequential()
      conv1.add(ConvolutionDnn(inputSize, config[Table](1)(1), 1, 1, 1, 1)
        .setName(namePrefix + "1x1"))
      conv1.add(mkldnn.SpatialBatchNormalization(config[Table](1)(1), 1e-3)
        .setName(namePrefix + "1x1/bn"))
      conv1.add(ReLUDnn(true).setName(namePrefix + "1x1/bn/sc/relu"))
      concat.add(conv1)
    }

    val conv3 = Sequential()
    conv3.add(ConvolutionDnn(inputSize, config[Table](2)(1), 1, 1, 1, 1)
      .setName(namePrefix + "3x3_reduce"))
    conv3.add(mkldnn.SpatialBatchNormalization(config[Table](2)(1), 1e-3)
      .setName(namePrefix + "3x3_reduce/bn"))
    conv3.add(ReLUDnn(true). setName(namePrefix + "3x3_reduce/bn/sc/relu"))
    if(config[Table](4)[String](1) == "max" && config[Table](4)[Int](2) == 0) {
      conv3.add(ConvolutionDnn(config[Table](2)(1),
        config[Table](2)(2), 3, 3, 2, 2, 1, 1).setName(namePrefix + "3x3"))
    } else {
      conv3.add(ConvolutionDnn(config[Table](2)(1),
        config[Table](2)(2), 3, 3, 1, 1, 1, 1).setName(namePrefix + "3x3"))
    }
    conv3.add(mkldnn.SpatialBatchNormalization(config[Table](2)(2), 1e-3)
      .setName(namePrefix + "3x3/bn"))
    conv3.add(ReLUDnn(true).setName(namePrefix + "3x3/bn/sc/relu"))
    concat.add(conv3)

    val conv3xx = Sequential()
    conv3xx.add(ConvolutionDnn(inputSize, config[Table](3)(1), 1, 1, 1, 1)
      .setName(namePrefix + "double3x3_reduce"))
    conv3xx.add(mkldnn.SpatialBatchNormalization(config[Table](3)(1), 1e-3)
      .setName(namePrefix + "double3x3_reduce/bn"))
    conv3xx.add(ReLUDnn(true).setName(namePrefix + "double3x3_reduce/bn/sc/relu"))

    conv3xx.add(ConvolutionDnn(config[Table](3)(1),
      config[Table](3)(2), 3, 3, 1, 1, 1, 1).setName(namePrefix + "double3x3a"))
    conv3xx.add(mkldnn.SpatialBatchNormalization(config[Table](3)(2), 1e-3)
      .setName(namePrefix + "double3x3a/bn"))
    conv3xx.add(ReLUDnn(true).setName(namePrefix + "double3x3a/bn/sc/relu"))

    if(config[Table](4)[String](1) == "max" && config[Table](4)[Int](2) == 0) {
      conv3xx.add(ConvolutionDnn(config[Table](3)(2),
        config[Table](3)(2), 3, 3, 2, 2, 1, 1).setName(namePrefix + "double3x3b"))
    } else {
      conv3xx.add(ConvolutionDnn(config[Table](3)(2),
        config[Table](3)(2), 3, 3, 1, 1, 1, 1).setName(namePrefix + "double3x3b"))
    }
    conv3xx.add(mkldnn.SpatialBatchNormalization(config[Table](3)(2), 1e-3)
      .setName(namePrefix + "double3x3b/bn"))
    conv3xx.add(ReLUDnn(true).setName(namePrefix + "double3x3b/bn/sc/relu"))
    concat.add(conv3xx)

    val pool = Sequential()
    config[Table](4)[String](1) match {
      case "max" =>
        if (config[Table](4)[Int](2) != 0) {
          pool.add(PoolingDnn(3, 3, 1, 1, 1, 1).ceil().setName(namePrefix + "pool"))
        } else {
          pool.add(PoolingDnn(3, 3, 2, 2).ceil().setName(namePrefix + "pool"))
        }
      case "avg" => pool.add(PoolingDnnAverage(3, 3, 1, 1, 1, 1).ceil()
        .setName(namePrefix + "pool"))
      case _ => throw new IllegalArgumentException
    }

    if (config[Table](4)[Int](2) != 0) {
      pool.add(ConvolutionDnn(inputSize, config[Table](4)[Int](2), 1, 1, 1, 1)
        .setName(namePrefix + "pool_proj"))
      pool.add(mkldnn.SpatialBatchNormalization(config[Table](4)(2), 1e-3)
        .setName(namePrefix + "pool_proj/bn"))
      pool.add(ReLUDnn(true).setName(namePrefix + "pool_proj/bn/sc/relu"))
    }
    concat.add(pool)
    concat.setName(namePrefix + "concat_output")

    concat
  }
}

object Inception_v1_NoAuxClassifier_dnn {
  def apply(classNum: Int, hasDropout: Boolean = true): Module[Float] = {
    val model = Sequential()
    model.add(ConvolutionDnn(3, 64, 7, 7, 2, 2, 3, 3, 1, false)
      .setInitMethod(weightInitMethod = Xavier, Zeros)
      .setName("conv1/7x7_s2"))
    model.add(ReLUDnn(true).setName("conv1/relu_7x7"))
    model.add(PoolingDnn(3, 3, 2, 2).ceil().setName("pool1/3x3_s2"))
    model.add(LRNDnn(5, 0.0001, 0.75).setName("pool1/norm1"))
    model.add(ConvolutionDnn(64, 64, 1, 1, 1, 1).setInitMethod(weightInitMethod = Xavier, Zeros)
      .setName("conv2/3x3_reduce"))
    model.add(ReLUDnn(true).setName("conv2/relu_3x3_reduce"))
    model.add(ConvolutionDnn(64, 192, 3, 3, 1, 1, 1, 1)
      .setInitMethod(weightInitMethod = Xavier, Zeros).setName("conv2/3x3"))
    model.add(ReLUDnn(true).setName("conv2/relu_3x3"))
    model.add(LRNDnn(5, 0.0001, 0.75). setName("conv2/norm2"))
    model.add(PoolingDnn(3, 3, 2, 2).ceil().setName("pool2/3x3_s2"))
    model.add(Inception_Layer_v1(192, T(T(64), T(96, 128), T(16, 32), T(32)), "inception_3a/"))
    model.add(Inception_Layer_v1(256, T(T(128), T(128, 192), T(32, 96), T(64)), "inception_3b/"))
    model.add(PoolingDnn(3, 3, 2, 2).ceil().setName("pool3/3x3_s2"))
    model.add(Inception_Layer_v1(480, T(T(192), T(96, 208), T(16, 48), T(64)), "inception_4a/"))
    model.add(Inception_Layer_v1(512, T(T(160), T(112, 224), T(24, 64), T(64)), "inception_4b/"))
    model.add(Inception_Layer_v1(512, T(T(128), T(128, 256), T(24, 64), T(64)), "inception_4c/"))
    model.add(Inception_Layer_v1(512, T(T(112), T(144, 288), T(32, 64), T(64)), "inception_4d/"))
    model.add(Inception_Layer_v1(528, T(T(256), T(160, 320), T(32, 128), T(128)), "inception_4e/"))
    model.add(PoolingDnn(3, 3, 2, 2).ceil().setName("pool4/3x3_s2"))
    model.add(Inception_Layer_v1(832, T(T(256), T(160, 320), T(32, 128), T(128)), "inception_5a/"))
    model.add(Inception_Layer_v1(832, T(T(384), T(192, 384), T(48, 128), T(128)), "inception_5b/"))
    model.add(PoolingDnnAverage(7, 7, 1, 1).setName("pool5/7x7_s1"))
    // if (hasDropout) model.add(Dropout(0.4).setName("pool5/drop_7x7_s1"))

    if (hasDropout) model.add(DropoutDnn(0.4).setName("pool5/drop_7x7_s1"))

    // model.add(View(1024).setNumInputDims(3))
    model.add(mkldnn.Linear(1024, classNum)
      .setInitMethod(weightInitMethod = Xavier, Zeros).setName("loss3/classifier"))
    model.add(LogSoftMax().setName("loss3/loss3"))
    model
  }
}

object Inception_v1_dnn {
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
    // output1.add(View(128 * 4 * 4).setNumInputDims(3))
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
    // output2.add(View(128 * 4 * 4).setNumInputDims(3))
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
    // output3.add(View(1024).setNumInputDims(3))
    output3.add(mkldnn.Linear(1024, classNum)
      .setInitMethod(weightInitMethod = Xavier, Zeros).setName("loss3/classifier"))
    output3.add(LogSoftMax().setName("loss3/loss3"))

    val split2 = ConcatDnn(2).setName("split2")
    split2.add(output3)
    split2.add(output2)

    val mainBranch = Sequential()
    mainBranch.add(feature2)
    mainBranch.add(split2)

    val split1 = ConcatDnn(2).setName("split1")
    split1.add(mainBranch)
    split1.add(output1)

    val model = Sequential()

    model.add(feature1)
    model.add(split1)

    model
  }
}

object Inception_v2_dnn {
  def apply(classNum: Int): Module[Float] = {
    val features1 = Sequential()
    features1.add(ConvolutionDnn(3, 64, 7, 7, 2, 2, 3, 3, 1, false)
      .setName("conv1/7x7_s2"))
    features1.add(mkldnn.SpatialBatchNormalization(64, 1e-3).setName("conv1/7x7_s2/bn"))
    features1.add(ReLUDnn(true).setName("conv1/7x7_s2/bn/sc/relu"))
    features1.add(PoolingDnn(3, 3, 2, 2).ceil().setName("pool1/3x3_s2"))
    features1.add(ConvolutionDnn(64, 64, 1, 1).setName("conv2/3x3_reduce"))
    features1.add(mkldnn.SpatialBatchNormalization(64, 1e-3).setName("conv2/3x3_reduce/bn"))
    features1.add(ReLUDnn(true).setName("conv2/3x3_reduce/bn/sc/relu"))
    features1.add(ConvolutionDnn(64, 192, 3, 3, 1, 1, 1, 1).setName("conv2/3x3"))
    features1.add(mkldnn.SpatialBatchNormalization(192, 1e-3).setName("conv2/3x3/bn"))
    features1.add(ReLUDnn(true).setName("conv2/3x3/bn/sc/relu"))
    features1.add(PoolingDnn(3, 3, 2, 2).ceil().setName("pool2/3x3_s2"))
    features1.add(Inception_Layer_v2(192, T(T(64), T(64, 64), T(64, 96), T("avg", 32)),
      "inception_3a/"))
    features1.add(Inception_Layer_v2(256, T(T(64), T(64, 96), T(64, 96), T("avg", 64)),
      "inception_3b/"))
    features1.add(Inception_Layer_v2(320, T(T(0), T(128, 160), T(64, 96), T("max", 0)),
      "inception_3c/"))

    val output1 = Sequential()
    output1.add(PoolingDnnAverage(5, 5, 3, 3).ceil().setName("pool3/5x5_s3"))
    output1.add(ConvolutionDnn(576, 128, 1, 1, 1, 1).setName("loss1/conv"))
    output1.add(mkldnn.SpatialBatchNormalization(128, 1e-3).setName("loss1/conv/bn"))
    output1.add(ReLUDnn(true).setName("loss1/conv/bn/sc/relu"))
    // output1.add(View(128 * 4 * 4).setNumInputDims(3))
    output1.add(mkldnn.Linear(128 * 4 * 4, 1024).setName("loss1/fc"))
    output1.add(ReLUDnn(true).setName("loss1/fc/bn/sc/relu"))
    output1.add(mkldnn.Linear(1024, classNum).setName("loss1/classifier"))
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
    output2.add(PoolingDnnAverage(5, 5, 3, 3).ceil().setName("pool4/5x5_s3"))
    output2.add(ConvolutionDnn(1024, 128, 1, 1, 1, 1).setName("loss2/conv"))
    output2.add(mkldnn.SpatialBatchNormalization(128, 1e-3).setName("loss2/conv/bn"))
    output2.add(ReLUDnn(true).setName("loss2/conv/bn/sc/relu"))
    // output2.add(View(128 * 2 * 2).setNumInputDims(3))
    output2.add(mkldnn.Linear(128 * 2 * 2, 1024).setName("loss2/fc"))
    output2.add(ReLUDnn(true).setName("loss2/fc/bn/sc/relu"))
    output2.add(mkldnn.Linear(1024, classNum).setName("loss2/classifier"))
    output2.add(LogSoftMax().setName("loss2/loss"))

    val output3 = Sequential()
    output3.add(Inception_Layer_v2(1024, T(T(352), T(192, 320), T(160, 224), T("avg", 128)),
      "inception_5a/"))
    output3.add(Inception_Layer_v2(1024, T(T(352), T(192, 320), T(192, 224), T("max", 128)),
      "inception_5b/"))
    output3.add(PoolingDnnAverage(7, 7, 1, 1).ceil().setName("pool5/7x7_s1"))
    // output3.add(View(1024).setNumInputDims(3))
    output3.add(mkldnn.Linear(1024, classNum).setName("loss3/classifier"))
    output3.add(LogSoftMax().setName("loss3/loss"))

    val split2 = ConcatDnn(2)
    split2.add(output3)
    split2.add(output2)

    val mainBranch = Sequential()
    mainBranch.add(features2)
    mainBranch.add(split2)

    val split1 = ConcatDnn(2)
    split1.add(mainBranch)
    split1.add(output1)

    val model = Sequential()

    model.add(features1)
    model.add(split1)

    model.reset()
    model
  }
}

object Vgg_16_dnn {
  def apply(classNum: Int, hasDropout: Boolean = true): Module[Float] = {
    val model = Sequential()
    model.add(ConvolutionDnn(3, 64, 3, 3, 1, 1, 1, 1))
    model.add(ReLUDnn(true))
    model.add(ConvolutionDnn(64, 64, 3, 3, 1, 1, 1, 1))
    model.add(ReLUDnn(true))
    model.add(PoolingDnn(2, 2, 2, 2))

    model.add(ConvolutionDnn(64, 128, 3, 3, 1, 1, 1, 1))
    model.add(ReLUDnn(true))
    model.add(ConvolutionDnn(128, 128, 3, 3, 1, 1, 1, 1))
    model.add(ReLUDnn(true))
    model.add(PoolingDnn(2, 2, 2, 2))

    model.add(ConvolutionDnn(128, 256, 3, 3, 1, 1, 1, 1))
    model.add(ReLUDnn(true))
    model.add(ConvolutionDnn(256, 256, 3, 3, 1, 1, 1, 1))
    model.add(ReLUDnn(true))
    model.add(ConvolutionDnn(256, 256, 3, 3, 1, 1, 1, 1))
    model.add(ReLUDnn(true))
    model.add(PoolingDnn(2, 2, 2, 2))

    model.add(ConvolutionDnn(256, 512, 3, 3, 1, 1, 1, 1))
    model.add(ReLUDnn(true))
    model.add(ConvolutionDnn(512, 512, 3, 3, 1, 1, 1, 1))
    model.add(ReLUDnn(true))
    model.add(ConvolutionDnn(512, 512, 3, 3, 1, 1, 1, 1))
    model.add(ReLUDnn(true))
    model.add(PoolingDnn(2, 2, 2, 2))

    model.add(ConvolutionDnn(512, 512, 3, 3, 1, 1, 1, 1))
    model.add(ReLUDnn(true))
    model.add(ConvolutionDnn(512, 512, 3, 3, 1, 1, 1, 1))
    model.add(ReLUDnn(true))
    model.add(ConvolutionDnn(512, 512, 3, 3, 1, 1, 1, 1))
    model.add(ReLUDnn(true))
    model.add(PoolingDnn(2, 2, 2, 2))

    model.add(mkldnn.Linear(512 * 7 * 7, 4096))
    // model.add(Threshold(0, 1e-6))
    model.add(ReLUDnn(value = 1e-6f))
    if (hasDropout) model.add(Dropout(0.5))
    model.add(mkldnn.Linear(4096, 4096))
    // model.add(Threshold(0, 1e-6))
    model.add(ReLUDnn(value = 1e-6f))
    if (hasDropout) model.add(Dropout(0.5))
    model.add(mkldnn.Linear(4096, classNum))
    model.add(LogSoftMax())

    model
  }
}

object Vgg_19_dnn {
  def apply(classNum: Int, hasDropout: Boolean = true): Module[Float] = {
    val model = Sequential()
    model.add(ConvolutionDnn(3, 64, 3, 3, 1, 1, 1, 1))
    model.add(ReLUDnn(true))
    model.add(ConvolutionDnn(64, 64, 3, 3, 1, 1, 1, 1))
    model.add(ReLUDnn(true))
    model.add(PoolingDnn(2, 2, 2, 2))

    model.add(ConvolutionDnn(64, 128, 3, 3, 1, 1, 1, 1))
    model.add(ReLUDnn(true))
    model.add(ConvolutionDnn(128, 128, 3, 3, 1, 1, 1, 1))
    model.add(ReLUDnn(true))
    model.add(PoolingDnn(2, 2, 2, 2))

    model.add(ConvolutionDnn(128, 256, 3, 3, 1, 1, 1, 1))
    model.add(ReLUDnn(true))
    model.add(ConvolutionDnn(256, 256, 3, 3, 1, 1, 1, 1))
    model.add(ReLUDnn(true))
    model.add(ConvolutionDnn(256, 256, 3, 3, 1, 1, 1, 1))
    model.add(ReLUDnn(true))
    model.add(ConvolutionDnn(256, 256, 3, 3, 1, 1, 1, 1))
    model.add(ReLUDnn(true))
    model.add(PoolingDnn(2, 2, 2, 2))

    model.add(ConvolutionDnn(256, 512, 3, 3, 1, 1, 1, 1))
    model.add(ReLUDnn(true))
    model.add(ConvolutionDnn(512, 512, 3, 3, 1, 1, 1, 1))
    model.add(ReLUDnn(true))
    model.add(ConvolutionDnn(512, 512, 3, 3, 1, 1, 1, 1))
    model.add(ReLUDnn(true))
    model.add(ConvolutionDnn(512, 512, 3, 3, 1, 1, 1, 1))
    model.add(ReLUDnn(true))
    model.add(PoolingDnn(2, 2, 2, 2))

    model.add(ConvolutionDnn(512, 512, 3, 3, 1, 1, 1, 1))
    model.add(ReLUDnn(true))
    model.add(ConvolutionDnn(512, 512, 3, 3, 1, 1, 1, 1))
    model.add(ReLUDnn(true))
    model.add(ConvolutionDnn(512, 512, 3, 3, 1, 1, 1, 1))
    model.add(ReLUDnn(true))
    model.add(ConvolutionDnn(512, 512, 3, 3, 1, 1, 1, 1))
    model.add(ReLUDnn(true))
    model.add(PoolingDnn(2, 2, 2, 2))

    // model.add(View(512 * 7 * 7))
    model.add(mkldnn.Linear(512 * 7 * 7, 4096))
    // model.add(Threshold(0, 1e-6))
    model.add(ReLUDnn(value = 1e-6f))
    if (hasDropout) model.add(Dropout(0.5))
    model.add(mkldnn.Linear(4096, 4096))
    // model.add(Threshold(0, 1e-6))
    model.add(ReLUDnn(value = 1e-6f))
    if (hasDropout) model.add(Dropout(0.5))
    model.add(mkldnn.Linear(4096, classNum))
    model.add(LogSoftMax())

    model
  }
}

object SbnDnn {
  def apply[@specialized(Float, Double) T: ClassTag](
    nOutput: Int,
    eps: Double = 1e-3,
    momentum: Double = 0.1,
    //    momentum: Double = 0.9,
    affine: Boolean = true)
  (implicit ev: TensorNumeric[T]): SpatialBatchNormalization[T] = {
    mkldnn.SpatialBatchNormalization[T](nOutput, eps, momentum, affine).setInitMethod(Ones, Zeros)
    //    SpatialBatchNormalization[T](nOutput, eps, momentum, affine).setInitMethod(Ones)
  }
}

object Convolution {
  def apply(
      nInputPlane: Int,
      nOutputPlane: Int,
      kernelW: Int,
      kernelH: Int,
      strideW: Int = 1,
      strideH: Int = 1,
      padW: Int = 0,
      padH: Int = 0,
      nGroup: Int = 1,
      propagateBack: Boolean = true,
      optnet: Boolean = true,
      weightDecay: Double = 1e-4): ConvolutionDnn = {
    val wReg = L2Regularizer[Float](weightDecay)
    val bReg = L2Regularizer[Float](weightDecay)
    val conv = mkldnn.ConvolutionDnn(nInputPlane, nOutputPlane, kernelW, kernelH,
        strideW, strideH, padW, padH, nGroup, propagateBack, wReg, bReg)
    conv.setInitMethod(MsraFiller(false), Zeros)
    //    conv.setInitMethod(MsraFiller(false))
    conv
  }
}

object ResNet_dnn {
  val logger = Logger.getLogger(getClass)

  def modelInit(model: Module[Float]): Unit = {
    logger.info("Initialize ResNet")
    def initModules(model: Module[Float]): Unit = {
      model match {
        case container: Container[Activity, Activity, Float]
        => container.modules.foreach(m => initModules(m))
        case convolutionDnn
          if (convolutionDnn.isInstanceOf[mkldnn.ConvolutionDnn]) =>
          val curModel = convolutionDnn.asInstanceOf[mkldnn.ConvolutionDnn]
          val n: Float = curModel.kernelW * curModel.kernelW * curModel.nOutputPlane
          curModel.weight.apply1(_ => RNG.normal(0, Math.sqrt(2.0f / n)).toFloat)
          curModel.bias.apply1(_ => 0.0f)
          curModel
        case spatialBatchNormalization
          if (spatialBatchNormalization.isInstanceOf[mkldnn.SpatialBatchNormalization[Float]]) =>
          val curModel =
            spatialBatchNormalization.asInstanceOf[mkldnn.SpatialBatchNormalization[Float]]
          curModel.weight.apply1(_ => 1.0f)
          curModel.bias.apply1(_ => 0.0f)
        case linear
          if (linear.isInstanceOf[mkldnn.Linear[Float]]) =>
          linear.asInstanceOf[Linear[Float]].bias.apply1(_ => 0.0f)
        case _ => Unit
      }
    }
    initModules(model)
  }

  var iChannels = 0
  def apply(classNum: Int, opt: Table): Module[Float] = {

    val depth = opt.get("depth").getOrElse(18)
    val shortCutType = opt.get("shortcutType")
    val shortcutType = shortCutType.getOrElse(ShortcutType.B).asInstanceOf[ShortcutType]
    val dataSet = opt.getOrElse[DatasetType]("dataSet", DatasetType.CIFAR10)
    val optnet = opt.get("optnet").getOrElse(true)

    def shortcut(nInputPlane: Int, nOutputPlane: Int, stride: Int, name: String): Module[Float] = {
      val useConv = shortcutType == ShortcutType.C ||
        (shortcutType == ShortcutType.B && nInputPlane != nOutputPlane)

      if (useConv) {
        Sequential()
          .add(mkldnn.Convolution(nInputPlane, nOutputPlane, 1, 1, stride, stride,
            optnet = optnet).setName(s"res${name}_branch1"))
          .add(mkldnn.SbnDnn(nOutputPlane).setName(s"bn${name}_branch1"))
      } else if (nInputPlane != nOutputPlane) {
        throw new IllegalArgumentException(s"useConv false")
      } else {
        Identity()
      }
    }

    def bottleneck(n: Int, stride: Int, name: String = ""): Module[Float] = {
      val nInputPlane = iChannels
      iChannels = n * 4

      val s = Sequential()
      s.add(mkldnn.Convolution(nInputPlane, n, 1, 1, 1, 1, 0, 0,
        optnet = optnet).setName(s"res${name}_branch2a"))
        .add(mkldnn.SbnDnn(n).setName(s"bn${name}_branch2a"))
        .add(ReLUDnn(true).setName(s"res${name}_branch2a_relu"))
        .add(mkldnn.Convolution(n, n, 3, 3, stride, stride, 1, 1,
          optnet = optnet).setName(s"res${name}_branch2b"))
        .add(mkldnn.SbnDnn(n).setName(s"bn${name}_branch2b"))
        .add(ReLUDnn(true).setName(s"res${name}_branch2b_relu"))
        .add(mkldnn.Convolution(n, n*4, 1, 1, 1, 1, 0, 0,
          optnet = optnet).setName(s"res${name}_branch2c"))
        .add(mkldnn.SbnDnn(n * 4).setInitMethod(Zeros, Zeros).setName(s"bn${name}_branch2c"))

      val model = Sequential()
        .add(ConcatTableDnn().
          add(s).
          add(shortcut(nInputPlane, n*4, stride, name)).setName(s"$name/concatTable"))
        .add(CAddTableDnn(true).setName(s"$name/caddTable"))
        .add(ReLUDnn(true).setName(s"res${name}_relu"))
      model
    }

    def getName(i: Int, name: String): String = {
      val name1 = i match {
        case 1 => name + "a"
        case 2 => name + "b"
        case 3 => name + "c"
        case 4 => name + "d"
        case 5 => name + "e"
        case 6 => name + "f"
      }
      return name1
    }

    def layer(block: (Int, Int, String) => Module[Float], features: Int,
              count: Int, stride: Int = 1, name : String): Module[Float] = {
      val s = Sequential()
      for (i <- 1 to count) {
        s.add(block(features, if (i == 1) stride else 1, getName(i, name)))
      }
      s
    }

    val model = Sequential()
    if (dataSet == DatasetType.ImageNet) {
      val cfg = Map(
        50 -> ((3, 4, 6, 3), 2048, bottleneck: (Int, Int, String) => Module[Float])
      )

      require(cfg.keySet.contains(depth), s"Invalid depth ${depth}")

      val (loopConfig, nFeatures, block) = cfg.get(depth).get
      iChannels = 64
      logger.info(" | ResNet-" + depth + " ImageNet")

      model.add(mkldnn.Convolution(3, 64, 7, 7, 2, 2, 3, 3, propagateBack = false,
        optnet = optnet).setName("conv1"))
        .add(mkldnn.SbnDnn(64).setName("bn_conv1"))
        .add(ReLUDnn(true).setName("conv1_relu"))
        .add(PoolingDnn(3, 3, 2, 2, 1, 1).setName("pool1"))
        .add(layer(block, 64, loopConfig._1, name = "2"))
        .add(layer(block, 128, loopConfig._2, 2, name = "3"))
        .add(layer(block, 256, loopConfig._3, 2, name = "4"))
        .add(layer(block, 512, loopConfig._4, 2, name = "5"))
        .add(PoolingDnnAverage(7, 7, 1, 1).setName("pool5"))
        .add(mkldnn.Linear(nFeatures, classNum, true, L2Regularizer(1e-4), L2Regularizer(1e-4))
          .setInitMethod(RandomNormal(0.0, 0.01), Zeros).setName("fc1000"))
    } else {
      throw new IllegalArgumentException(s"Invalid dataset ${dataSet}")
    }
    model
  }

  /**
    * dataset type
    * @param typeId type id
    */
  sealed abstract class DatasetType(typeId: Int)
    extends Serializable

  /**
    *  define some dataset type
    */
  object DatasetType {
    case object CIFAR10 extends DatasetType(0)
    case object ImageNet extends DatasetType(1)
  }

  /**
    * ShortcutType
    * @param typeId type id
    */
  sealed abstract class ShortcutType(typeId: Int)
    extends Serializable

  /**
    * ShortcutType-A is used for Cifar-10, ShortcutType-B is used for ImageNet.
    * ShortcutType-C is used for others.
    */
  object ShortcutType{
    case object A extends ShortcutType(0)
    case object B extends ShortcutType(1)
    case object C extends ShortcutType(2)
  }
}