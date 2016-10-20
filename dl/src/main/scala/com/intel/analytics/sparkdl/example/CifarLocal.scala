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

import com.intel.analytics.sparkdl.nn._
import com.intel.analytics.sparkdl.optim.SGD
import com.intel.analytics.sparkdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.sparkdl.tensor.{Storage, Tensor}
import com.intel.analytics.sparkdl.utils.{File, T}

import scala.io.Source
import scala.reflect.ClassTag

object CifarLocal {

  def main(args: Array[String]) {
    if ("float" == args(10)) {
      new CifarLocal[Float]().train(args)
    } else {
      new CifarLocal[Double]().train(args)
    }
  }
}

class CifarLocal[@specialized(Float, Double) T: ClassTag](implicit ev: TensorNumeric[T]) {

  var curIter = 1
  var correct = 0
  var count = 0
  var l = 0.0
  var wallClockTime = 0L

  def train(args: Array[String]) {
    require(args.length >= 10,
      "invalid args, should be <trainFile> <testFile> <batchSize> <trainSize>" +
        " <testSize> <learningRate> <weightDecay> <momentum> <learningRateDecay> <netType>")

    val trainFile = args(0)
    val testFile = args(1)
    val batchSize = args(2).toInt
    val trainSize = args(3).toInt
    val testSize = args(4).toInt
    val learningRate = args(5).toDouble
    val weightDecay = args(6).toDouble
    val momentum = args(7).toDouble
    val learningRateDecay = args(8).toDouble
    val netType = args(9)


    val fullTrainData = Tensor[T](50000, 3, 32, 32)
    val fullTrainLabel = Tensor[T](50000)
    var i = 1
    val trainDataFile = Source.fromFile(trainFile).getLines().foreach { line =>
      val record = line.split("\\|")
      val label = ev.fromType[Double](record(0).toDouble)
      val image = Tensor[T](Storage[T](record.slice(1, record.length).map(_.toDouble).
        map(ev.fromType[Double](_))))
      fullTrainLabel(i) = label
      fullTrainData(i).copy(image)
      i += 1
    }
    i = 1
    val fullTestData = Tensor[T](10000, 3, 32, 32)
    val fullTestLabel = Tensor[T](10000)
    val testDataFile = Source.fromFile(testFile).getLines().foreach { line =>
      val record = line.split("\\|")
      val label = ev.fromType[Double](record(0).toDouble)
      val image = Tensor[T](Storage[T](record.slice(1, record.length).map(_.toDouble).
        map(ev.fromType[Double](_))))
      fullTestLabel(i) = label
      fullTestData(i).copy(image)
      i += 1
    }
    val trainData = fullTrainData.narrow(1, 1, trainSize)
    val trainLabel = fullTrainLabel.narrow(1, 1, trainSize)
    val testData = fullTestData.narrow(1, 1, testSize)
    val testLabel = fullTestLabel.narrow(1, 1, testSize)


    val module = Cifar.getModel[T](10, netType)
    val criterion = new ClassNLLCriterion[T]()
    val optm = new SGD[T]()
    //    val optm = new Adagrad()
    val config = T("learningRate" -> learningRate, "weightDecay" -> weightDecay,
      "momentum" -> momentum, "learningRateDecay" -> learningRateDecay)
    val state = T()

    val (masterWeights, masterGrad) = module.getParameters()

    println(s"model length is ${masterWeights.nElement()}")
    println(module)

    var epoch = 1
    while (epoch < 300) {
      if (epoch % 25 == 0) {
        config("learningRate") = config.get[Double]("learningRate").getOrElse(0.0) / 2
      }
      println(config)
      i = 1
      var epochTrainTime = 0L
      while (i < trainSize) {
        val size = if (i + batchSize - 1 <= trainSize) batchSize else (trainSize - i + 1)
        val startTime = System.nanoTime()
        optm.optimize(feval(masterGrad, module, criterion, trainData.narrow(1, i, size),
          trainLabel.narrow(1, i, size)), masterWeights, config, state)
        val endTime = System.nanoTime()
        epochTrainTime += (endTime - startTime)
        wallClockTime += (endTime - startTime)
        i += size
      }
      println(s"[$epoch]At training $correct of $count, ${
        correct.toDouble /
          count.toDouble
      }. Time cost ${(epochTrainTime) / 1e6 / trainSize.toDouble}" +
        s"ms for a single sample. Throughput is ${count.toDouble * 1e9 / (epochTrainTime)}" +
        s" records/second")
      println(s"Average Loss: ${l * batchSize / count}")

      evaluate(masterGrad, module, criterion, testData, testLabel, 128)

      l = 0.0
      correct = 0
      count = 0
      epoch += 1
    }
  }


  def feval(grad: Tensor[T],
    module: Module[Tensor[T], Tensor[T], T],
    criterion: Criterion[T], input: Tensor[T],
    target: Tensor[T])(weights: Tensor[T])
  : (T, Tensor[T]) = {
    module.training()
    grad.zero()
    val output = module.forward(input)
    for (d <- 1 to output.size(1)) {
      val pre = maxIndex(output.select(1, d))
      //          print(s"|pre:${pre}tar:${target(Array(d))}|")
      count += 1
      if (pre == ev.toType[Int](target.valueAt(d))) correct += 1
    }

    val loss = criterion.forward(output, target)
    val gradOut = criterion.backward(output, target)
    module.backward(input, gradOut)
    //    print(module.gradInput)
    l += ev.toType[Double](loss)
    //    print(" " + loss + " ")
    (loss, grad)
  }


  def evaluate(masterGrad: Tensor[T],
    module: Module[Tensor[T], Tensor[T], T], criterion: Criterion[T],
    testData: Tensor[T], testLabel: Tensor[T], batchSize: Int = 1000): Unit = {
    module.evaluate()
    var i = 1
    var evaCorrect = 0
    val testSize = testData.size(1)

    val startTime = System.nanoTime()
    while (i < testSize) {
      val size = if (i + batchSize - 1 <= testSize) batchSize else (testSize - i + 1)
      evaCorrect += evaluate(masterGrad, module, criterion, testData.narrow(1, i, size),
        testLabel.narrow(1, i, size))
      i += size
    }
    val endTime = System.nanoTime()
    println(s"At ${wallClockTime / 1e9}s test $evaCorrect of $testSize, ${
      evaCorrect / testSize.toDouble
    }. " +
      s"Time cost ${(endTime - startTime) / 1e6 / testSize.toDouble}ms for a single sample.")

  }


  def evaluate(grad: Tensor[T], module: Module[Tensor[T], Tensor[T], T], criterion: Criterion[T],
    input: Tensor[T], target: Tensor[T]): Int = {
    val output = module.forward(input)
    var corrects = 0
    for (d <- 1 to output.size(1)) {
      val pre = maxIndex(output.select(1, d))
      //                print(s"pre:${pre}tar:${target(Array(d))}   ")
      if (pre == ev.toType[Int](target.valueAt(d))) corrects += 1
    }
    corrects
  }


  def maxIndex(data: Tensor[T]): Int = {
    require(data.dim() == 1)
    var index = 1
    val indexer = Array(1)
    var maxVal = data(indexer)
    var j = 2
    while (j <= data.size(1)) {
      indexer(0) = j
      if (ev.toType[Double](data(indexer)) > ev.toType[Double](maxVal)) {
        maxVal = data(indexer)
        index = j
      }
      j += 1
    }
    index
  }

  def getModel(file: String): Module[Tensor[Double], Tensor[Double], Double] = {
    val model = File.load[Module[Tensor[Double], Tensor[Double], Double]](file)

    model
  }
}
