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

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.optim.SGD
import org.scalatest.FlatSpec
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import com.intel.analytics.bigdl.utils.T
import java.io._
import java.math.MathContext

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.models.RnnCell

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

class RecurrentSpec extends FlatSpec {
/*

  "A Recurrent Module " should "generate correct output and grad" in {


    // Initialize layer parameters to sqrt(1/n)
    def layerInit(model: Module[Double]): Unit = {
      Utils.rnnLayerInit(model)
    }


    // load in the data, sentence is indicated by an array of dictionary index numbers
    // example: input = [61, 4, 2], label = [315, 5, 1]
    def readNumbers(filename: String, dictionarySize: Int): ArrayBuffer[Array[Int]] = {
      val inputBuffer = ArrayBuffer[Array[Int]]()
      val br = new BufferedReader(new FileReader(filename))
      var line = br.readLine()
      var sampleIndex = 1
      while (line != null) {
        sampleIndex += 1
        line = line.replaceAll("[^\\d,]", "")
        val stringArray = line.split(",")
        val curInput = new Array[Int](stringArray.length)
        for ((index, i) <- stringArray.zipWithIndex) {
          curInput(i) = index.toInt + 1
        }
        inputBuffer.append(curInput)
        line = br.readLine()
      }
      br.close()
      inputBuffer
    }

    // paths pointing to data
    val trainFile =
    "/home/ywan/Documents/data/shakespeare/" +
      "preprocessing/wordByWord/train.bn"
    val trainLabelFile =
      "/home/ywan/Documents/data/shakespeare/" +
        "preprocessing/wordByWord/trainLabel.bn"
    val dictionaryFile =
      "/home/ywan/Documents/data/shakespeare/" +
        "preprocessing/wordByWord/dictionary.txt"

    val testFile =
      "/home/ywan/Documents/data/shakespeare/" +
        "preprocessing/wordByWord/test.bn"
    val testLabelFile =
      "/home/ywan/Documents/data/shakespeare/" +
        "preprocessing/wordByWord/testLabel.bn"


    // loading the dictionary and data
    val dictionaryBuffer = ArrayBuffer[String]()
    var br = new BufferedReader(new FileReader(dictionaryFile))
    var sb = new StringBuilder()
    var line = br.readLine()

    while (line != null) {
      var dictMap = line.split("\t")
      dictionaryBuffer.append(dictMap(1))
      line = br.readLine
    }
    br.close()

    val dictionary = dictionaryBuffer.toArray
    println(s"dictionary size = ${dictionary.length}")

    val inputTrainArray = readNumbers(trainFile, dictionary.length)
    val trainLabelArray = readNumbers(trainLabelFile, dictionary.length)

    val inputTestArray = readNumbers(testFile, dictionary.length)
    val testLabelArray = readNumbers(testLabelFile, dictionary.length)

    println("train: input size = " + inputTrainArray.length)
    println("train: label size = " + trainLabelArray.length)
    println("test: input size = " + inputTestArray.length)
    println("test: label size = " + testLabelArray.length)


    // initializing the model

    val inputSize = dictionary.length
    val hiddenSize = 40
    val outputSize = dictionary.length
    val bpttTruncate = 4

    val model = Sequential[Double]()
    model.add(Recurrent[Double](hiddenSize, bpttTruncate)
      .add(RnnCell[Double](inputSize, hiddenSize))
      .add(Tanh[Double]()))
      .add(Linear[Double](hiddenSize, outputSize))

    layerInit(model)

    val (weights, grad) = model.getParameters()
    val criterion = CrossEntropyCriterion[Double]()
    val logSoftMax = LogSoftMax[Double]()

    val modelPath = "/home/ywan/Documents/data/shakespeare/model/"


    // loading one sentence, and use model to predict the corresponding sentence

    def predict(input: Tensor[Double]): Array[Int] = {
      model.forward(input)
      logSoftMax.forward(model.output.asInstanceOf[Tensor[Double]])
      val outputIndex = logSoftMax.output.max(2)._2
      val output = outputIndex.storage().array
      val outputInt = new Array[Int](output.length)
      for ((x, i) <- output.zipWithIndex) {
        outputInt(i) = x.toInt
      }
      outputInt
    }

    def perplex(model: Module[Double], input: Array[Int],
                label: Array[Int], nGram: Int = 2): (Double, Double) = {
      var pp: Double = 0.0
      val length = input.length
      val dictionarySize = dictionary.length
      var acc: Double = 0.0
      for (i <- 0 to (length - nGram)) {
        val curInput = Tensor[Double](Array(nGram, dictionarySize))
        for (j <- i until (i + nGram)) {
          curInput.setValue((j - i + 1), input(j), 1.0)
        }
        model.forward(curInput)
        val logOutput = logSoftMax.forward(model.output.asInstanceOf[Tensor[Double]])
        for (j <- 1 to nGram) {
          pp += logOutput(Array(j, label(i + j - 1)))
        }
      }

      val curInput = Tensor[Double](Array(length, dictionarySize))
      for (i <- 0 until length) {
        curInput.setValue(i + 1, input(i), 1.0)
      }
      model.forward(curInput)
      logSoftMax.forward(model.output.asInstanceOf[Tensor[Double]])
      val predLabel = logSoftMax.output.max(2)._2
      for (i <- 1 to predLabel.size(1)) {
        if (predLabel.valueAt(i, 1).toInt == label(i - 1)) acc += 1
      }
      (-pp / (nGram * (length - nGram + 1)), acc / length.toDouble)
    }

    def testMetric(unitTestLength: Int = 10, nGram: Int = 2):
    (Double, Double, Double) = {
      var pp, acc, accLoss = 0.0
      var testLength = Math.min(unitTestLength, inputTestArray.length)
      for (i <- 0 until testLength) {
        val (curPP, curAcc) = perplex(model, inputTestArray(i), testLabelArray(i), nGram)
        val sampleLength = inputTestArray(i).length
        val input = Tensor[Double](sampleLength, dictionary.length)
        for ((wordIndex, sampleIndex) <- inputTestArray(i).zipWithIndex) {
          input.setValue(sampleIndex + 1, wordIndex, 1.0)
        }
        val labels = Tensor[Double](sampleLength)
        for ((wordIndex, sampleIndex) <- testLabelArray(i).zipWithIndex) {
          labels.setValue(sampleIndex + 1, wordIndex)
        }
        model.forward(input)
        criterion.forward(model.output.asInstanceOf[Tensor[Double]], labels)
        accLoss += criterion.output
        pp += curPP
        acc += curAcc
      }
      (pp/testLength, accLoss/testLength, acc/testLength)
    }

    def trainWithSGD(
      model: Module[Double],
      inputArray: ArrayBuffer[Array[Int]], labelArray: ArrayBuffer[Array[Int]],
      numOfSents: Int = 100, unitTestLength: Int = 6557,
      learningRate: Double = 0.01, nepochs: Int = 30, evaluateLossAfter: Int = 5)
    : Double = {

      val state = T("learningRate" -> learningRate, "momentum" -> 0.0, "weightDecay" -> 0.0,
        "dampening" -> 0.0)
      val sgd = new SGD[Double]

      var curLearningRate = learningRate
      var totalLoss = 0.0
      var preLoss = Double.PositiveInfinity

      val seq = (0 until inputArray.length).toList

      for (epoch <- 1 to nepochs) {
        state.update("learningRate", curLearningRate)
        println(s"learning rate = ${state.get("learningRate").get}")
        val nGram = 2
        var numExampleSeen = 0
        var averageLoss = 0.0
        var averageAcc = 0.0

        val shuffledSeq = Random.shuffle(seq)

        for (i <- 0 until numOfSents) {
          numExampleSeen += 1
          val inputBuffer = inputArray(shuffledSeq(i))
          val labelBuffer = labelArray(shuffledSeq(i))
          val sampleLength = inputBuffer.length
          val input = Tensor[Double](sampleLength, dictionary.length)
          for ((wordIndex, sampleIndex) <- inputBuffer.zipWithIndex) {
            input.setValue(sampleIndex + 1, wordIndex, 1.0)
          }
          val labels = Tensor[Double](sampleLength)
          for ((wordIndex, sampleIndex) <- labelBuffer.zipWithIndex) {
            labels.setValue(sampleIndex + 1, wordIndex)
          }

          def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
            model.forward(input)
            val _loss = criterion.forward(model.output.asInstanceOf[Tensor[Double]], labels)
            model.zeroGradParameters()
            val gradOutputTest =
              criterion.backward(model.output.asInstanceOf[Tensor[Double]], labels)
            model.backward(input, gradOutputTest)
            (_loss, grad)
          }
          val (_, loss) = sgd.optimize(feval, weights, state)
          averageLoss += loss(0)
          if (i % 1000 == 0) println(s"numOfSents = ${i}, Training: Loss = ${loss(0)}")
        }

        if (epoch % 2 == 0) {
          for (i <- 1 to 20) {
            val inputBuffer = inputArray(i)
            val labelBuffer = labelArray(i)
            val sampleLength = inputBuffer.length
            val input = Tensor[Double](sampleLength, dictionary.length)
            for ((wordIndex, sampleIndex) <- inputBuffer.zipWithIndex) {
              input.setValue(sampleIndex + 1, wordIndex, 1.0)
            }
            val predictArray = predict(input)
            println(predictArray.mkString(","))
            println(labelBuffer.mkString(","))
            println()
          }
        }
        averageLoss /= numOfSents
        if (averageLoss > preLoss) curLearningRate /= 2

        println(s"epoch = ${epoch}, Training: Loss = ${averageLoss}")

        val (testPP, testLoss, testAcc) = testMetric(unitTestLength, nGram)
        println(s"epoch = ${epoch}, Testing: Loss =" +
          s"${testLoss}, Perplexity = ${testPP}, Acc = ${testAcc}")
        preLoss = averageLoss
      }
      0
    }

    trainWithSGD(
      model,
      inputTrainArray,
      trainLabelArray,
      learningRate = 0.01,
      numOfSents = inputTrainArray.length,
      unitTestLength = inputTestArray.length,
      nepochs = 30, evaluateLossAfter = 1)

    // 26220, 6557
  }

*/



/*

  "A Recurrent Module " should "converge" in {

    val batchSize = 8
    val rho = 5

    val hiddenSize = 4
    val inputSize = 5
    val outputSize = 5
    val bpttTruncate = 3

    val model = new Sequential[Tensor[Double], Tensor[Double], Double]()
    model.add(new Recurrent[Double](hiddenSize, bpttTruncate)
      .add(new RnnCell[Double](inputSize, hiddenSize)))
      .add(new Linear[Double](hiddenSize, outputSize))

    val criterion = new CrossEntropyCriterion[Double]()
    val logSoftMax = new LogSoftMax[Double]()

    val (weights, grad) = model.getParameters()

   // println(weights)
    val input = Tensor[Double](5, inputSize)
    val labels = Tensor[Double](5)
            for (i <- 1 to 5) {
              val rdmLabel = Math.ceil(Math.random()*inputSize).toInt
              val rdmInput = Math.ceil(Math.random()*inputSize).toInt
              input.setValue(i, rdmInput, 1.0)
              labels.setValue(i, rdmLabel)
            }

//    input.setValue(1, 3, 1.0)
//    input.setValue(2, 5, 1.0)
//    input.setValue(3, 4, 1.0)
//    input.setValue(4, 4, 1.0)
//    input.setValue(5, 3, 1.0)
//    labels.setValue(1, 3)
//    labels.setValue(2, 5)
//    labels.setValue(3, 3)
//    labels.setValue(4, 2)
//    labels.setValue(5, 3)

    val state = T("learningRate" -> 0.05, "momentum" -> 0.9, "weightDecay" -> 5e-4,
      "dampening" -> 0.0)
    val sgd = new SGD[Double]
    //println(weights.size.mkString(","))
    def feval(x: Tensor[Double]): (Double, Tensor[Double]) = {
      model.forward(input)
      criterion.forward(model.output, labels)
      model.zeroGradParameters()
      criterion.backward(model.output, labels)
      model.backward(input, criterion.gradInput)
      (criterion.output, grad)
    }
    for (i <- 1 to 20) {
      val (_, loss) = sgd.optimize(feval, weights, state)
      println(loss.mkString)
    }
    //println()
    //println(grad)
    val output = model.forward(input)
    val logOutput = logSoftMax.forward(output)
    val prediction = logOutput.max(2)._2
    val inputIndex = input.max(2)._2
    println(inputIndex)
    println(labels)
    println(prediction)
  }
*/


  "A Recurrent Module " should "perform correct gradient check" in {
    def layerInit(model: Module[Double]): Unit = {
      Utils.rnnLayerInit(model)
    }

    val hiddenSize = 4
    val inputSize = 5
    val outputSize = 5
    val bpttTruncate = 10
    val seed = 100
    RNG.setSeed(seed)
    val model = Sequential[Double]()
    model.add(Recurrent[Double](hiddenSize, bpttTruncate)
      .add(RnnCell[Double](inputSize, hiddenSize))
      .add(Tanh[Double]()))
      .add(Linear[Double](hiddenSize, outputSize))

    val packageName: String = model.getName().stripSuffix("Sequential")
    layerInit(model)
    // val input = Tensor[Double](4, inputSize).randn
    val input = Tensor[Double](5, inputSize)
    val labels = Tensor[Double](5)
    for (i <- 1 to 5) {
      val rdmLabel = Math.ceil(Math.random()*inputSize).toInt
      val rdmInput = Math.ceil(Math.random()*inputSize).toInt
      input.setValue(i, rdmInput, 1.0)
      labels.setValue(i, rdmLabel)
    }


//        input.setValue(1, 3, 1.0)
//        input.setValue(2, 5, 1.0)
//        input.setValue(3, 4, 1.0)
//        input.setValue(4, 4, 1.0)
//        input.setValue(5, 3, 1.0)
//
//        labels.setValue(1, 3)
//        labels.setValue(2, 5)
//        labels.setValue(3, 3)
//        labels.setValue(4, 2)
//        labels.setValue(5, 3)




    println("gradient check for input")
    val gradCheckerInput = new GradientChecker(1e-2, 1)
    val checkFlagInput = gradCheckerInput.checkLayer[Double](model, input)
    println("gradient check for weights")
    val gradCheck = new GradientCheckerRNN(1e-2, 1)
    val checkFlag = gradCheck.checkLayer(model, input, labels)
  }
}
