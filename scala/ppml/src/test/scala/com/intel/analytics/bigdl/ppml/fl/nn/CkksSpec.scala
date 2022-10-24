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

package com.intel.analytics.bigdl.ppml.fl.nn

import com.intel.analytics.bigdl.ckks.CKKS
import com.intel.analytics.bigdl.dllib.NNContext
import com.intel.analytics.bigdl.dllib.feature.dataset.{DataSet, Sample, SampleToMiniBatch, TensorSample}
import com.intel.analytics.bigdl.dllib.nn._
import com.intel.analytics.bigdl.dllib.optim.{Adagrad, Ftrl, SGD}
import com.intel.analytics.bigdl.dllib.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.dllib.utils.{Engine, RandomGenerator, T}
import com.intel.analytics.bigdl.ppml.BigDLSpecHelper
import com.intel.analytics.bigdl.ppml.fl.nn.ckks.{Encryptor, FusedBCECriterion, CAddTable => CkksAddTable}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.FlatSpec

import java.util
import scala.math.abs
import scala.util.Random

class CkksSpec extends BigDLSpecHelper {
  "ckks add" should "return right result" in {

    val ckks = new CKKS()
    val secrets = ckks.createSecrets()
    val encryptorPtr = ckks.createCkksEncryptor(secrets)
    val ckksRunnerPtr = ckks.createCkksCommon(secrets)

    val input1 = Array(0.1f, 0.2f, 1.1f, -1f)
    val input2 = Array(-0.1f, 1.2f, 2.1f, 1f)
    val enInput1 = ckks.ckksEncrypt(encryptorPtr, input1)
    val enInput2 = ckks.ckksEncrypt(encryptorPtr, input2)

    val cadd = CkksAddTable(ckksRunnerPtr)

    val enOutput = cadd.updateOutput(enInput1, enInput2)
    val output = ckks.ckksDecrypt(encryptorPtr, enOutput)
    (0 until 4).foreach{i =>
      output(i) should be (input1(i) + input2(i) +- 1e-5f)
    }
  }

  "ckks layer" should "generate correct output and grad" in {
    val eps = 1e-12f
    val module = new Sigmoid[Float]
    val criterion = new BCECriterion[Float]()

    val input = Tensor[Float](2, 2)
    input(Array(1, 1)) = 0.063364277360961f
    input(Array(1, 2)) = 0.90631252736785f
    input(Array(2, 1)) = 0.22275671223179f
    input(Array(2, 2)) = 0.37516756891273f
    val target = Tensor[Float](2, 2)
    target(Array(1, 1)) = 1
    target(Array(1, 2)) = 1
    target(Array(2, 1)) = 0
    target(Array(2, 2)) = 1

    val exceptedOutput = module.forward(input)

    val exceptedLoss = criterion.forward(exceptedOutput, target)
    val exceptedGradOutput = criterion.backward(exceptedOutput, target)
    val exceptedGradInput = module.backward(input, exceptedGradOutput)


    val ckks = new CKKS()
    val secrets = ckks.createSecrets()
    val encryptorPtr = ckks.createCkksEncryptor(secrets)
    val ckksRunnerPtr = ckks.createCkksCommon(secrets)
    val enTarget =
      Tensor[Byte](Storage[Byte](ckks.ckksEncrypt(encryptorPtr, target.storage().array())))
        .resize(target.size())

    val module2 = Encryptor[Float](encryptorPtr)
    val criterion2 = FusedBCECriterion(ckksRunnerPtr)

    val output2 = module2.forward(input).toTensor[Byte]
    val loss2 = criterion2.forward(output2, enTarget)
    val gradOutput2 = criterion2.backward(output2, enTarget)
    val gradInput2 = module2.backward(input, gradOutput2).toTensor[Float]


    val enLoss = ckks.ckksDecrypt(encryptorPtr, loss2.storage().array)
    gradInput2.div(4)
    val loss = enLoss.slice(0, 4).sum / 4
    loss should be (exceptedLoss +- 0.02f)
    (1 to 2).foreach{i =>
      (1 to 2).foreach{j =>
        gradInput2.valueAt(i, j) should be (exceptedGradInput.valueAt(i, j) +- 0.02f)
      }
    }
  }

  "ckks jni api" should "generate correct output and grad" in {
    val eps = 1e-12f
    val module = new Sigmoid[Float]
    val criterion = new BCECriterion[Float]()
    val input = Tensor[Float](2, 2)
    input(Array(1, 1)) = 0.063364277360961f
    input(Array(1, 2)) = 0.90631252736785f
    input(Array(2, 1)) = 0.22275671223179f
    input(Array(2, 2)) = 0.37516756891273f
    val target = Tensor[Float](2, 2)
    target(Array(1, 1)) = 1
    target(Array(1, 2)) = 1
    target(Array(2, 1)) = 0
    target(Array(2, 2)) = 1

    val exceptedOutput = module.forward(input)

    val exceptedLoss = criterion.forward(exceptedOutput, target)
    val exceptedGradOutput = criterion.backward(exceptedOutput, target)
    val exceptedGradInput = module.backward(input, exceptedGradOutput)

    val ckks = new CKKS()
    val secrets = ckks.createSecrets()
    val encryptorPtr = ckks.createCkksEncryptor(secrets)
    val ckksRunnerPtr = ckks.createCkksCommon(secrets)
    val enInput = ckks.ckksEncrypt(encryptorPtr, input.storage().array())
    val enTarget = ckks.ckksEncrypt(encryptorPtr, target.storage().array())
    val o = ckks.train(ckksRunnerPtr, enInput, enTarget)
    val enLoss = ckks.ckksDecrypt(encryptorPtr, o(0))
    val enGradInput2 = ckks.ckksDecrypt(encryptorPtr, o(1))
    val gradInput2 = Tensor[Float](enGradInput2.slice(0, 4), Array(2, 2))
    gradInput2.div(4)
    val loss = enLoss.slice(0, 4).sum / 4
    loss should be (exceptedLoss +- 0.02f)
    (1 to 2).foreach{i =>
      (1 to 2).foreach{j =>
        gradInput2.valueAt(i, j) should be (exceptedGradInput.valueAt(i, j) +- 0.02f)
      }

    }
  }

  "ckks forward" should "generate correct output" in {
    val module = new Sigmoid[Float]
    val input = Tensor[Float](2, 4)
    input(Array(1, 1)) = 0.063364277360961f
    input(Array(1, 2)) = 0.90631252736785f
    input(Array(1, 3)) = 0.22275671223179f
    input(Array(1, 4)) = 0.37516756891273f
    input(Array(2, 1)) = 0.99284988618456f
    input(Array(2, 2)) = 0.97488326719031f
    input(Array(2, 3)) = 0.94414822547697f
    input(Array(2, 4)) = 0.68123375508003f
    val exceptedOutput = module.forward(input)

    val ckks = new CKKS()
    val secrets = ckks.createSecrets()
    val encryptorPtr = ckks.createCkksEncryptor(secrets)
    val ckksRunnerPtr = ckks.createCkksCommon(secrets)
    val enInput = ckks.ckksEncrypt(encryptorPtr, input.storage().array())
    val enOutput = ckks.sigmoidForward(ckksRunnerPtr, enInput)
    val outputArray = ckks.ckksDecrypt(encryptorPtr, enOutput(0))
    val output = Tensor[Float](outputArray.slice(0, 8), Array(2, 4))
    println(output)
    println(exceptedOutput)
    (1 to 2).foreach{i =>
      (1 to 4).foreach{j =>
        output.valueAt(i, j) should be (exceptedOutput.valueAt(i, j) +- 0.03f)
      }
    }
  }

  "ckks train" should "converge" in {
    val random = new Random()
    random.setSeed(10)
    val featureLen = 10
    val bs = 20
    val totalSize = 1000
    val dummyData = Array.tabulate(totalSize)(i =>
      {
        val features = Array.tabulate(featureLen)(_ => random.nextFloat())
        val label = math.round(features.sum / featureLen).toFloat
        Sample[Float](Tensor[Float](features, Array(featureLen)), label)
      }
    )
    val dataset = DataSet.array(dummyData) ->
      SampleToMiniBatch[Float](bs, parallelizing = false)

    val module = Sequential[Float]()
    module.add(Linear[Float](10, 1))
    module.add(Sigmoid[Float]())
    val criterion = new BCECriterion[Float]()
    val sgd = new SGD[Float](0.1)
    val sgd2 = new SGD[Float](0.1)
    val (weight, gradient) = module.getParameters()

    val module2 = Linear[Float](10, 1)
    val (weight2, gradient2) = module2.getParameters()
    weight2.copy(weight)
    val ckks = new CKKS()
    val secrets = ckks.createSecrets()
    val encryptorPtr = ckks.createCkksEncryptor(secrets)
    val ckksRunnerPtr = ckks.createCkksCommon(secrets)

    val epochNum = 2
    val lossArray = new Array[Float](epochNum)
    val loss2Array = new Array[Float](epochNum)
    (0 until epochNum).foreach{epoch =>
      var countLoss = 0f
      var countLoss2 = 0f
      dataset.shuffle()
      val trainData = dataset.toLocal().data(false)
      while(trainData.hasNext) {
        val miniBatch = trainData.next()
        val input = miniBatch.getInput()
        val target = miniBatch.getTarget()
        val output = module.forward(input)
        val loss = criterion.forward(output, target)
        countLoss += loss
        val gradOutput = criterion.backward(output, target)
        module.backward(input, gradOutput)
        sgd.optimize(_ => (loss, gradient), weight)

        val output2 = module2.forward(input).toTensor[Float]
        val enInput = ckks.ckksEncrypt(encryptorPtr, output2.storage().array())
        val enTarget = ckks.ckksEncrypt(encryptorPtr, target.toTensor[Float].storage().array())
        val o = ckks.train(ckksRunnerPtr, enInput, enTarget)

        val enLoss = ckks.ckksDecrypt(encryptorPtr, o(0))
        val enGradInput2 = ckks.ckksDecrypt(encryptorPtr, o(1))
        val gradInput2 = Tensor[Float](enGradInput2.slice(0, bs), Array(bs, 1))
        gradInput2.div(bs)
        module2.backward(input, gradInput2)
        val loss2 = enLoss.slice(0, bs).sum / bs
        sgd2.optimize(_ => (loss2, gradient2), weight2)
        countLoss2 += loss2
        module.zeroGradParameters()
        module2.zeroGradParameters()
      }
      lossArray(epoch) = countLoss / (totalSize / bs)
      loss2Array(epoch) = countLoss2 / (totalSize / bs)
      println(countLoss / (totalSize / bs))
      println("           " + countLoss2 / (totalSize / bs))
    }
    println("loss1: ")
    println(lossArray.mkString("\n"))
    println("loss2: ")
    println(loss2Array.mkString("\n"))
    lossArray.last - lossArray(0) should be (loss2Array.last -
      loss2Array(0) +- 1e-2f)
  }

}
