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

package com.intel.analytics.sparkdl.example

import java.awt.color.ColorSpace
import java.util

import com.intel.analytics.sparkdl.nn.ClassNLLCriterion
import com.intel.analytics.sparkdl.optim.SGD
import com.intel.analytics.sparkdl.tensor.Tensor
import com.intel.analytics.sparkdl.utils.{File, T}

object ImageNetLocal {
  val startTime = System.nanoTime()

  val regimes = Map(
    "alexnet" -> Array(
      (1, 18, 1e-2, 5e-4),
      (19, 29, 5e-3, 5e-4),
      (30, 43, 1e-3, 0.0),
      (44, 52, 5e-4, 0.0),
      (53, 100000000, 1e-4, 0.0)
    ),
    "googlenet-cf" -> Array(
      (1, 18, 1e-2, 2e-4),
      (19, 29, 5e-3, 2e-4),
      (30, 43, 1e-3, 0.0),
      (44, 52, 5e-4, 0.0),
      (53, 100000000, 1e-4, 0.0)
    )
  )

  def log(msg: String): Unit = {
    println(s"[${(System.nanoTime() - startTime) / 1e9}s] $msg")
  }

  def runDouble(donkey: Donkey, dataSet: DataSets, netType: String, classNum: Int,
    labelsMap: Map[String, Double], testInterval: Int, donkeyVal: Donkey,
    dataSetVal: DataSets, batchSize: Int): Unit = {
    // Compute Mean on amount of samples
    val samples = 10000
    log(s"Start to calculate Mean on $samples samples")
    var (meanR, meanG, meanB) = Array.tabulate(samples)(n => {
      print(".")
      val data = donkey.pull
      dataSet.post(data._2)
      ImageNetUtils.computeMean(data._1, data._2.dataOffset)
    }).reduce((a, b) => (a._1 + b._1, a._2 + b._2, a._3 + b._3))
    meanR /= samples
    meanG /= samples
    meanB /= samples
    println()

    // Compute std on amount of samples
    log(s"Start to calculate std on $samples samples")
    var (varR, varG, varB) = Array.tabulate(samples)(n => {
      print(".")
      val data = donkey.pull
      dataSet.post(data._2)
      ImageNetUtils.computeVar(data._1, meanR, meanG, meanB, data._2.dataOffset)
    }).reduce((a, b) => (a._1 + b._1, a._2 + b._2, a._3 + b._3))
    varR /= samples
    varG /= samples
    varB /= samples

    val model = netType match {
      case "alexnet" => AlexNet.getModel[Double](classNum)
      case "googlenet" => GoogleNet.getModel[Double](classNum)
      case "googlenet-bn" => GoogleNet.getModel[Double](classNum, "googlenet-bn")
      case "googlenet-cf" => GoogleNet.getModelCaffe[Double](classNum)
      case _ => throw new IllegalArgumentException
    }
    val (weights, grad) = model.getParameters()
    println(s"modelsize ${weights.nElement()}")
    println(model)
    val criterion = new ClassNLLCriterion[Double]()
    val epochNum = 90
    val featureShape = Array(3, 224, 224)
    val targetShape = Array(1)
    val sgd = new SGD[Double]
    val state = T("momentum" -> 0.9, "dampening" -> 0.0)
    val stageImgs = new util.ArrayDeque[Image](batchSize)
    val input = Tensor[Double](batchSize, 3, 224, 224)
    val target = Tensor[Double](batchSize)
    val iter = ImageNetUtils.toTensorDouble(
      donkey.map(d => {
        stageImgs.push(d._2)
        (labelsMap(d._2.label), d._1)
      }),
      featureShape,
      targetShape,
      batchSize,
      (meanR, meanG, meanB),
      (varR, varG, varB),
      input,
      target
    )

    val stageImgsVal = new util.ArrayDeque[Image](batchSize)
    val iterVal = ImageNetUtils.toTensorDouble(
      donkeyVal.map(d => {
        stageImgsVal.push(d._2)
        (labelsMap(d._2.label), d._1)
      }),
      featureShape,
      targetShape,
      batchSize,
      (meanR, meanG, meanB),
      (varR, varG, varB),
      input,
      target
    )

    log(s"meanR is $meanR meanG is $meanG meanB is $meanB")
    log(s"varR is $varR varG is $varG varB is $varB")
    log("Start to train...")

    var wallClockTime = 0L
    for (i <- 1 to epochNum) {
      println(s"Epoch[$i] Train")

      for (regime <- regimes(netType)) {
        if (i >= regime._1 && i <= regime._2) {
          state("learningRate") = regime._3
          state("weightDecay") = regime._4
        }
      }

      var j = 0
      var c = 0
      model.training()
      while (j < dataSet.getTotal) {
        val start = System.nanoTime()
        val (input, target) = iter.next()
        val readImgTime = System.nanoTime()
        model.zeroGradParameters()
        val output = model.forward(input)
        val loss = criterion.forward(output, target)
        val gradOutput = criterion.backward(output, target)
        model.backward(input, gradOutput)
        sgd.optimize(_ => (loss, grad), weights, state, state)
        val end = System.nanoTime()
        wallClockTime += end - start
        log(s"Epoch[$i][Iteration $c $j/${dataSet.getTotal}][Wall Clock ${wallClockTime / 1e9}s]" +
          s" loss is $loss time ${(end - start) / 1e9}s read " +
          s"time ${(readImgTime - start) / 1e9}s train time ${(end - readImgTime) / 1e9}s." +
          s" Throughput is ${input.size(1).toDouble / (end - start) * 1e9} img / second")
        while (!stageImgs.isEmpty) {
          dataSet.post(stageImgs.poll())
        }
        j += input.size(1)
        c += 1
      }

      if (i % testInterval == 0) {
        model.evaluate()
        var correct = 0
        var k = 0
        while (k < dataSetVal.getTotal) {
          val (input, target) = iterVal.next()
          val output = model.forward(input)
          output.max(2)._2.squeeze().map(target, (a, b) => {
            if (a == b) {
              correct += 1
            }
            a
          })
          while (!stageImgsVal.isEmpty) {
            dataSetVal.post(stageImgsVal.poll())
          }
          k += input.size(1)
        }

        val accuracy = correct.toDouble / dataSetVal.getTotal
        println(s"[Wall Clock ${wallClockTime / 1e9}s] Accuracy is $accuracy")

        // Save model to a file each epoch
        File.save(model, s"${netType}${accuracy}.model${i}", true)
        File.save(state, s"${netType}${accuracy}.state${i}", true)
      }

      log("shuffle")
      dataSet.shuffle
      log("shuffle end")
    }
  }

  def runFloat(donkey: Donkey, dataSet: DataSets, netType: String, classNum: Int,
    labelsMap: Map[String, Double], testInterval: Int, donkeyVal: Donkey,
    dataSetVal: DataSets, batchSize: Int): Unit = {
    // Compute Mean on amount of samples
    val samples = 10000
    log(s"Start to calculate Mean on $samples samples")
    var (meanR, meanG, meanB) = Array.tabulate(samples)(n => {
      print(".")
      val data = donkey.pull
      dataSet.post(data._2)
      ImageNetUtils.computeMean(data._1, data._2.dataOffset)
    }).reduce((a, b) => (a._1 + b._1, a._2 + b._2, a._3 + b._3))
    meanR /= samples
    meanG /= samples
    meanB /= samples
    println()

    // Compute std on amount of samples
    log(s"Start to calculate std on $samples samples")
    var (varR, varG, varB) = Array.tabulate(samples)(n => {
      print(".")
      val data = donkey.pull
      dataSet.post(data._2)
      ImageNetUtils.computeVar(data._1, meanR, meanG, meanB, data._2.dataOffset)
    }).reduce((a, b) => (a._1 + b._1, a._2 + b._2, a._3 + b._3))
    varR /= samples
    varG /= samples
    varB /= samples

    val model = netType match {
      case "alexnet" => AlexNet.getModel[Float](classNum)
      case "googlenet" => GoogleNet.getModel[Float](classNum)
      case "googlenet-bn" => GoogleNet.getModel[Float](classNum, "googlenet-bn")
      case "googlenet-cf" => GoogleNet.getModelCaffe[Float](classNum)
      case _ => throw new IllegalArgumentException
    }
    val (weights, grad) = model.getParameters()
    println(s"modelsize ${weights.nElement()}")
    println(model)
    val criterion = new ClassNLLCriterion[Float]()
    val epochNum = 90
    val featureShape = Array(3, 224, 224)
    val targetShape = Array(1)
    val sgd = new SGD[Float]
    val state = T("momentum" -> 0.9, "dampening" -> 0.0)
    val stageImgs = new util.ArrayDeque[Image](batchSize)
    val input = Tensor[Float](batchSize, 3, 224, 224)
    val target = Tensor[Float](batchSize)
    val meanRFloat = meanR.toFloat
    val meanGFloat = meanG.toFloat
    val meanBFloat = meanB.toFloat
    val varRFloat = varR.toFloat
    val varGFloat = varG.toFloat
    val varBFloat = varB.toFloat
    val iter = ImageNetUtils.toTensorFloat(
      donkey.map(d => {
        stageImgs.push(d._2)
        (labelsMap(d._2.label).toFloat, d._1)
      }),
      featureShape,
      targetShape,
      batchSize,
      (meanRFloat, meanGFloat, meanBFloat),
      (varRFloat, varGFloat, varBFloat),
      input,
      target
    )


    val stageImgsVal = new util.ArrayDeque[Image](batchSize)
    val iterVal = ImageNetUtils.toTensorFloat(
      donkeyVal.map(d => {
        stageImgsVal.push(d._2)
        (labelsMap(d._2.label).toFloat, d._1)
      }),
      featureShape,
      targetShape,
      batchSize,
      (meanRFloat, meanGFloat, meanBFloat),
      (varRFloat, varGFloat, varBFloat),
      input,
      target
    )

    log(s"meanR is $meanR meanG is $meanG meanB is $meanB")
    log(s"varR is $varR varG is $varG varB is $varB")
    log("Start to train...")

    var wallClockTime = 0L
    for (i <- 1 to epochNum) {
      println(s"Epoch[$i] Train")

      for (regime <- regimes(netType)) {
        if (i >= regime._1 && i <= regime._2) {
          state("learningRate") = regime._3
          state("weightDecay") = regime._4
        }
      }

      var j = 0
      var c = 0
      model.training()
      while (j < dataSet.getTotal) {
        val start = System.nanoTime()
        val (input, target) = iter.next()
        val readImgTime = System.nanoTime()
        model.zeroGradParameters()
        val output = model.forward(input)
        val loss = criterion.forward(output, target)
        val gradOutput = criterion.backward(output, target)
        model.backward(input, gradOutput)
        sgd.optimize(_ => (loss, grad), weights, state, state)
        val end = System.nanoTime()
        wallClockTime += end - start
        log(s"Epoch[$i][Iteration $c $j/${dataSet.getTotal}][Wall Clock ${wallClockTime / 1e9}s]" +
          s" loss is $loss time ${(end - start) / 1e9}s read " +
          s"time ${(readImgTime - start) / 1e9}s train time ${(end - readImgTime) / 1e9}s." +
          s" Throughput is ${input.size(1).toDouble / (end - start) * 1e9} img / second")
        while (!stageImgs.isEmpty) {
          dataSet.post(stageImgs.poll())
        }
        j += input.size(1)
        c += 1
      }

      if (i % testInterval == 0) {
        model.evaluate()
        var correct = 0
        var k = 0
        while (k < dataSetVal.getTotal) {
          val (input, target) = iterVal.next()
          val output = model.forward(input)
          output.max(2)._2.squeeze().map(target, (a, b) => {
            if (a == b) {
              correct += 1
            }
            a
          })
          while (!stageImgsVal.isEmpty) {
            dataSetVal.post(stageImgsVal.poll())
          }
          k += input.size(1)
        }

        val accuracy = correct.toDouble / dataSetVal.getTotal
        println(s"[Wall Clock ${wallClockTime / 1e9}s] Accuracy is $accuracy")
      }

      log("shuffle")
      dataSet.shuffle
      log("shuffle end")
    }
  }

  def main(args: Array[String]): Unit = {
    // See http://stackoverflow.com/questions/26535842/multithreaded-jpeg-image-processing-in-java
    Class.forName("javax.imageio.ImageIO")
    Class.forName("java.awt.color.ICC_ColorSpace")
    Class.forName("sun.java2d.cmm.lcms.LCMS")
    ColorSpace.getInstance(ColorSpace.CS_sRGB).toRGB(Array[Float](0, 0, 0))

    require(args.length == 9, "invalid args, should be <path> <parallelism> <labelPath>" +
      " <testInterval> <netType> <classNum> <dataType> <batchSize>")

    val path = args(0)
    val parallelism = args(1).toInt
    val labelsMap = ImageNetUtils.getLabels(args(2))
    val pathVal = args(3)
    val testInterval = args(4).toInt
    val netType = args(5)
    val classNum = args(6).toInt
    val dataType = args(7)
    val batchSize = args(8).toInt

    val dataSet = new DataSets(path, classNum, labelsMap)
    val donkey = new Donkey(parallelism, dataSet)
    val dataSetVal = new DataSets(pathVal, classNum, labelsMap)
    val donkeyVal = new Donkey(parallelism, dataSetVal)

    log("shuffle")
    dataSet.shuffle
    log("shuffle end")

    dataType match {
      case "double" => runDouble(donkey, dataSet, netType, classNum, labelsMap, testInterval,
        donkeyVal, dataSetVal, batchSize)
      case "float" => runFloat(donkey, dataSet, netType, classNum, labelsMap, testInterval,
        donkeyVal, dataSetVal, batchSize)
      case _ => throw new IllegalArgumentException
    }
  }
}
