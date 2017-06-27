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

import java.util

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.optim.SGD
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric._
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

@com.intel.analytics.bigdl.tags.Parallel
object Inception {
  def getModel[D: ClassTag](classNum: Int, modelName: String = "")(
    implicit ev: TensorNumeric[D]): Module[D] = {
    modelName match {
      case "inception-bn" =>
        def inception(inputSize: Int, config: Table)(
          implicit ev: TensorNumeric[D]): Module[D] = {
          val concat = Concat[D](2)
          if (config[Table](1)[Int](1) != 0) {
            val conv1 = Sequential[D]
            conv1.add(SpatialConvolution[D](inputSize, config[Table](1)(1), 1, 1, 1, 1))
            conv1.add(SpatialBatchNormalization(config[Table](1)(1), 1e-3))
            conv1.add(ReLU[D](true))
            concat.add(conv1)
          }

          val conv3 = Sequential[D]
          conv3.add(SpatialConvolution[D](inputSize, config[Table](2)(1), 1, 1, 1, 1))
          conv3.add(SpatialBatchNormalization(config[Table](2)(1), 1e-3))
          conv3.add(ReLU[D](true))
          conv3.add(SpatialConvolution[D](config[Table](2)(1),
            config[Table](2)(2), 3, 3, 1, 1, 1, 1))
          conv3.add(SpatialBatchNormalization(config[Table](2)(2), 1e-3))
          conv3.add(ReLU[D](true))
          concat.add(conv3)

          val conv3xx = Sequential[D]
          conv3xx.add(SpatialConvolution[D](inputSize, config[Table](3)(1), 1, 1, 1, 1))
          conv3xx.add(SpatialBatchNormalization(config[Table](3)(1), 1e-3))
          conv3xx.add(ReLU[D](true))

          conv3xx.add(SpatialConvolution[D](config[Table](3)(1),
            config[Table](3)(2), 3, 3, 1, 1, 1, 1))
          conv3xx.add(SpatialBatchNormalization(config[Table](3)(2), 1e-3))
          conv3xx.add(ReLU[D](true))

          conv3xx.add(SpatialConvolution[D](config[Table](3)(2),
            config[Table](3)(2), 3, 3, 1, 1, 1, 1))
          conv3xx.add(SpatialBatchNormalization(config[Table](3)(2), 1e-3))
          conv3xx.add(ReLU[D](true))
          concat.add(conv3xx)

          val pool = Sequential[D]
          pool.add(SpatialZeroPadding[D](1, 1, 1, 1))
          config[Table](4)[String](1) match {
            case "max" => pool.add(SpatialMaxPooling[D](3, 3, 1, 1).ceil())
            case "avg" => pool.add(SpatialAveragePooling[D](3, 3, 1, 1).ceil())
            case _ => throw new IllegalArgumentException
          }

          if (config[Table](4)[Int](2) != 0) {
            pool.add(SpatialConvolution[D](inputSize, config[Table](4)[Int](2), 1, 1, 1, 1))
            pool.add(SpatialBatchNormalization(config[Table](4)(2), 1e-3))
            pool.add(ReLU[D](true))
          }
          concat.add(pool)

          concat
        }
        val features = Sequential[D]
        features.add(SpatialConvolution[D](3, 64, 7, 7, 2, 2, 3, 3))
        features.add(SpatialBatchNormalization(64, 1e-3))
        features.add(ReLU[D](true))
        features.add(SpatialMaxPooling[D](3, 3, 2, 2).ceil())
        features.add(SpatialConvolution[D](64, 64, 1, 1))
        features.add(ReLU[D](true))
        features.add(SpatialConvolution[D](64, 192, 3, 3, 1, 1, 1, 1))
        features.add(SpatialBatchNormalization(192, 1e-3))
        features.add(ReLU[D](true))
        features.add(SpatialMaxPooling[D](3, 3, 2, 2).ceil())
        features.add(inception(192, T(T(64), T(64, 64), T(64, 96), T("avg", 32))))
        features.add(inception(256, T(T(64), T(64, 96), T(64, 96), T("avg", 64))))
        features.add(inception(320, T(T(0), T(128, 160), T(64, 96), T("max", 0))))
        features.add(SpatialConvolution[D](576, 576, 2, 2, 2, 2))
        features.add(inception(576, T(T(224), T(64, 96), T(96, 128), T("avg", 128))))
        features.add(inception(576, T(T(192), T(96, 128), T(96, 128), T("avg", 128))))
        features.add(inception(576, T(T(160), T(128, 160), T(128, 160), T("avg", 96))))
        features.add(inception(576, T(T(96), T(128, 192), T(160, 192), T("avg", 96))))

        val mainBranch = Sequential[D]
        mainBranch.add(inception(576, T(T(0), T(128, 192), T(192, 256), T("max", 0))))
        mainBranch.add(SpatialConvolution[D](1024, 1024, 2, 2, 2, 2))
        mainBranch.add(SpatialBatchNormalization(1024, 1e-3))
        mainBranch.add(inception(1024, T(T(352), T(192, 320), T(160, 224), T("avg", 128))))
        mainBranch.add(inception(1024, T(T(352), T(192, 320), T(192, 224), T("max", 128))))
        mainBranch.add(SpatialAveragePooling[D](7, 7, 1, 1))
        mainBranch.add(View[D](1024).setNumInputDims(3))
        mainBranch.add(Linear[D](1024, classNum))
        mainBranch.add(LogSoftMax[D])

        val auxClassifier = Sequential[D]
        auxClassifier.add(SpatialAveragePooling[D](5, 5, 3, 3).ceil())
        auxClassifier.add(SpatialConvolution[D](576, 128, 1, 1, 1, 1))
        auxClassifier.add(SpatialBatchNormalization(128, 1e-3))
        auxClassifier.add(View[D](128 * 4 * 4).setNumInputDims(3))
        auxClassifier.add(Linear[D](128 * 4 * 4, 768))
        auxClassifier.add(ReLU[D](true))
        auxClassifier.add(Linear[D](768, classNum))
        auxClassifier.add(LogSoftMax[D])

        val splitter = Concat[D](2)
        splitter.add(mainBranch)
        splitter.add(auxClassifier)

        val model = Sequential[D]
        model.add(features)
        model.add(splitter)

        model
      case default =>
        val features = Sequential[D]
        features.add(SpatialConvolution[D](3, 64, 7, 7, 2, 2, 3, 3))
        features.add(ReLU[D](true))
        features.add(SpatialMaxPooling[D](3, 3, 2, 2).ceil())
        features.add(SpatialConvolution[D](64, 64, 1, 1))
        features.add(ReLU[D](true))
        features.add(SpatialConvolution[D](64, 192, 3, 3, 1, 1, 1, 1))
        features.add(ReLU[D](true))
        features.add(SpatialMaxPooling[D](3, 3, 2, 2).ceil())
        features.add(inception(192, T(T(64), T(64, 64), T(64, 96), T("avg", 32))))
        features.add(inception(256, T(T(64), T(64, 96), T(64, 96), T("avg", 64))))
        features.add(inception(320, T(T(0), T(128, 160), T(64, 96), T("max", 0))))
        features.add(SpatialConvolution[D](576, 576, 2, 2, 2, 2))
        features.add(inception(576, T(T(224), T(64, 96), T(96, 128), T("avg", 128))))
        features.add(inception(576, T(T(192), T(96, 128), T(96, 128), T("avg", 128))))
        features.add(inception(576, T(T(160), T(128, 160), T(128, 160), T("avg", 96))))
        features.add(inception(576, T(T(96), T(128, 192), T(160, 192), T("avg", 96))))

        val mainBranch = Sequential[D]
        mainBranch.add(inception(576, T(T(0), T(128, 192), T(192, 256), T("max", 0))))
        mainBranch.add(SpatialConvolution[D](1024, 1024, 2, 2, 2, 2))
        mainBranch.add(inception(1024, T(T(352), T(192, 320), T(160, 224), T("avg", 128))))
        mainBranch.add(inception(1024, T(T(352), T(192, 320), T(192, 224), T("max", 128))))
        mainBranch.add(SpatialAveragePooling[D](7, 7, 1, 1))
        mainBranch.add(View[D](1024).setNumInputDims(3))
        mainBranch.add(Linear[D](1024, classNum))
        mainBranch.add(LogSoftMax[D])

        val auxClassifier = Sequential[D]
        auxClassifier.add(SpatialAveragePooling[D](5, 5, 3, 3).ceil())
        auxClassifier.add(SpatialConvolution[D](576, 128, 1, 1, 1, 1))
        auxClassifier.add(View[D](128 * 4 * 4).setNumInputDims(3))
        auxClassifier.add(Linear[D](128 * 4 * 4, 768))
        auxClassifier.add(ReLU[D](true))
        auxClassifier.add(Linear[D](768, classNum))
        auxClassifier.add(LogSoftMax[D])

        val splitter = Concat[D](2)
        splitter.add(mainBranch)
        splitter.add(auxClassifier)

        val model = Sequential[D]
        model.add(features)
        model.add(splitter)

        model
    }
  }

  def inception[D: ClassTag](inputSize: Int, config: Table)(
    implicit ev: TensorNumeric[D]): Module[D] = {
    val concat = Concat[D](2)
    if (config[Table](1)[Int](1) != 0) {
      val conv1 = Sequential[D]
      conv1.add(SpatialConvolution[D](inputSize, config[Table](1)(1), 1, 1, 1, 1))
      conv1.add(ReLU[D](true))
      concat.add(conv1)
    }

    val conv3 = Sequential[D]
    conv3.add(SpatialConvolution[D](inputSize, config[Table](2)(1), 1, 1, 1, 1))
    conv3.add(ReLU[D](true))
    conv3.add(SpatialConvolution[D](config[Table](2)(1),
      config[Table](2)(2), 3, 3, 1, 1, 1, 1))
    conv3.add(ReLU[D](true))
    concat.add(conv3)

    val conv3xx = Sequential[D]
    conv3xx.add(SpatialConvolution[D](inputSize, config[Table](3)(1), 1, 1, 1, 1))
    conv3xx.add(ReLU[D](true))
    conv3xx.add(SpatialConvolution[D](config[Table](3)(1),
      config[Table](3)(2), 3, 3, 1, 1, 1, 1))
    conv3xx.add(ReLU[D](true))
    conv3xx.add(SpatialConvolution[D](config[Table](3)(2),
      config[Table](3)(2), 3, 3, 1, 1, 1, 1))
    conv3xx.add(ReLU[D](true))
    concat.add(conv3xx)

    val pool = Sequential[D]
    pool.add(SpatialZeroPadding[D](1, 1, 1, 1))
    config[Table](4)[String](1) match {
      case "max" => pool.add(SpatialMaxPooling[D](3, 3, 1, 1).ceil())
      case "avg" => pool.add(SpatialAveragePooling[D](3, 3, 1, 1).ceil())
      case _ => throw new IllegalArgumentException
    }

    if (config[Table](4)[Int](2) != 0) {
      pool.add(SpatialConvolution[D](inputSize, config[Table](4)[Int](2), 1, 1, 1, 1))
      pool.add(ReLU[D](true))
    }
    concat.add(pool)

    concat
  }

  def getModelCaffe[D: ClassTag](classNum: Int)
    (implicit ev: TensorNumeric[D]): Module[D] = {
    def inception[D: ClassTag](inputSize: Int, config: Table)(
      implicit ev: TensorNumeric[D]): Module[D] = {
      val concat = Concat[D](2)
      val conv1 = Sequential[D]
      conv1.add(SpatialConvolution[D](inputSize,
        config[Table](1)(1), 1, 1, 1, 1).setInitMethod(weightInitMethod = Xavier))
      conv1.add(ReLU[D](true))
      concat.add(conv1)

      val conv3 = Sequential[D]
      conv3.add(SpatialConvolution[D](inputSize, config[Table](2)(1), 1, 1, 1, 1).
        setInitMethod(Xavier))
      conv3.add(ReLU[D](true))
      conv3.add(SpatialConvolution[D](config[Table](2)(1),
        config[Table](2)(2), 3, 3, 1, 1, 1, 1).setInitMethod(weightInitMethod = Xavier))
      conv3.add(ReLU[D](true))
      concat.add(conv3)

      val conv5 = Sequential[D]
      conv5.add(SpatialConvolution[D](inputSize, config[Table](3)(1), 1, 1, 1, 1).
        setInitMethod(Xavier))
      conv5.add(ReLU[D](true))
      conv5.add(SpatialConvolution[D](config[Table](3)(1),
        config[Table](3)(2), 5, 5, 1, 1, 2, 2).setInitMethod(weightInitMethod = Xavier))
      conv5.add(ReLU[D](true))
      concat.add(conv5)

      val pool = Sequential[D]
      pool.add(SpatialMaxPooling[D](3, 3, 1, 1, 1, 1))
      pool.add(SpatialConvolution[D](inputSize, config[Table](4)(1), 1, 1, 1, 1).
        setInitMethod(Xavier))
      concat.add(pool)

      concat
    }

    val features = Sequential[D]
    features.add(SpatialConvolution[D](3, 64, 7, 7, 2, 2, 3, 3)
      .setInitMethod(weightInitMethod = Xavier))
    features.add(ReLU[D](true))
    features.add(SpatialMaxPooling[D](3, 3, 2, 2, 1, 1))
    features.add(SpatialCrossMapLRN[D](5, 0.0001, 0.75))
    features.add(SpatialConvolution[D](64, 64, 1, 1, 1, 1, 0, 0)
      .setInitMethod(weightInitMethod = Xavier))
    features.add(ReLU[D](true))
    features.add(SpatialConvolution[D](64, 192, 3, 3, 1, 1, 1, 1)
      .setInitMethod(weightInitMethod = Xavier))
    features.add(ReLU[D](true))
    features.add(SpatialCrossMapLRN[D](5, 0.0001, 0.75))
    features.add(SpatialMaxPooling[D](3, 3, 2, 2, 1, 1))
    features.add(inception(192, T(T(64), T(96, 128), T(16, 32), T(32))))
    features.add(inception(256, T(T(128), T(128, 192), T(32, 96), T(64))))
    features.add(SpatialMaxPooling[D](3, 3, 2, 2, 1, 1))
    features.add(inception(480, T(T(192), T(96, 208), T(16, 48), T(64))))

    features.add(inception(512, T(T(160), T(112, 224), T(24, 64), T(64))))
    features.add(inception(512, T(T(128), T(128, 256), T(24, 64), T(64))))
    features.add(inception(512, T(T(112), T(144, 288), T(32, 64), T(64))))

    features.add(inception(528, T(T(256), T(160, 320), T(32, 128), T(128))))
    features.add(SpatialMaxPooling[D](3, 3, 2, 2, 1, 1))
    features.add(inception(832, T(T(256), T(160, 320), T(32, 128), T(128))))
    features.add(inception(832, T(T(384), T(192, 384), T(48, 128), T(128))))
    features.add(SpatialAveragePooling[D](7, 7, 1, 1))
    features.add(Dropout[D](0.4))
    features.add(View[D](1024).setNumInputDims(3))
    features.add(Linear[D](1024, classNum).setInitMethod(weightInitMethod = Xavier))
    features.add(LogSoftMax[D])
    features.reset()
    features
  }

  def performanceDouble(batchSize: Int, iter: Int, netType: String): Unit = {
    val input = Tensor[Double](batchSize, 3, 224, 224).fill(0.5)
    val model = getModelCaffe[Double](1000)
    val criterion = ClassNLLCriterion[Double]()
    var i = 0
    val sgd = new SGD[Double]
    val labelData = new Array[Double](batchSize)
    util.Arrays.fill(labelData, 10)
    val labels = Tensor[Double](Storage(labelData))

    println(model)
    println("warm up")
    while (i < 5) {
      val output = model.forward(input)
      val loss = criterion.forward(output, labels)
      val gradOutput = criterion.backward(output, labels)
      model.backward(input, gradOutput)
      i += 1
    }
    println("warm up done")
    model.resetTimes()
    var forwardTime = 0L
    var backwardTime = 0L
    while (i < iter) {
      var start = System.nanoTime()
      val output = model.forward(input)
      val loss = criterion.forward(output, labels)
      forwardTime += System.nanoTime() - start
      start = System.nanoTime()
      val gradOutput = criterion.backward(output, labels)
      model.backward(input, gradOutput)
      backwardTime += System.nanoTime() - start
      i += 1
    }
    println(s"forward time is ${forwardTime / iter / 1e6}ms")
    println(s"backward time is ${backwardTime / iter / 1e6}ms")
    val times = model.getTimes()
    var n = 0
    println(times.map(t => ( {
      n += 1;
      s"${t._1}-$n"
    }, (t._2 + t._3) / 1e9 / iter,
      t._2 / 1e9 / iter, t._3 / 1e9 / iter))
      .sortWith(_._2 > _._2).mkString("\n"))
  }

  def performanceFloat(batchSize: Int, iter: Int, netType: String): Unit = {
    val input = Tensor[Float](batchSize, 3, 224, 224).fill(0.5f)
    val model = getModelCaffe[Float](1000)
    val criterion = ClassNLLCriterion[Float]()
    var i = 0
    val sgd = new SGD[Float]
    val labelData = new Array[Float](batchSize)
    util.Arrays.fill(labelData, 10)
    val labels = Tensor[Float](Storage(labelData))

    println(model)
    println("warm up")
    while (i < 5) {
      val output = model.forward(input)
      val loss = criterion.forward(output, labels)
      val gradOutput = criterion.backward(output, labels)
      model.backward(input, gradOutput)
      i += 1
    }
    println("warm up done")
    model.resetTimes()
    var forwardTime = 0L
    var backwardTime = 0L
    while (i < iter) {
      var start = System.nanoTime()
      val output = model.forward(input)
      val loss = criterion.forward(output, labels)
      forwardTime += System.nanoTime() - start
      start = System.nanoTime()
      val gradOutput = criterion.backward(output, labels)
      model.backward(input, gradOutput)
      backwardTime += System.nanoTime() - start
      i += 1
    }
    val times = model.getTimes()
    var n = 0
    println(times.map(t => ( {
      n += 1;
      s"${t._1}-$n"
    }, (t._2 + t._3) / 1e9 / iter,
      t._2 / 1e9 / iter, t._3 / 1e9 / iter))
      .sortWith(_._2 > _._2).mkString("\n"))
    println(s"forward time is ${forwardTime / iter / 1e6}ms")
    println(s"backward time is ${backwardTime / iter / 1e6}ms")
    println(s"total time is ${(forwardTime + backwardTime) / iter / 1e6}ms")
  }

  def main(args: Array[String]): Unit = {
    require(args.length >= 1)
    args(0) match {
      case "perf" => args(3) match {
        case "double" => performanceDouble(args(1).toInt, args(2).toInt, "default")
        case "float" => performanceFloat(args(1).toInt, args(2).toInt, "default")
        case _ => throw new IllegalArgumentException
      }
      case _ => throw new IllegalArgumentException
    }
    System.exit(0)
  }
}
