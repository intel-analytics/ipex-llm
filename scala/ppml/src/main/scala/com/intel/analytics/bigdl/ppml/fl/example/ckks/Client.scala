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
package com.intel.analytics.bigdl.ppml.fl.example.ckks

import com.intel.analytics.bigdl.ckks.CKKS
import com.intel.analytics.bigdl.dllib.NNContext
import com.intel.analytics.bigdl.dllib.nn.SparseLinear
import com.intel.analytics.bigdl.dllib.utils.{Log4Error, RandomGenerator}
import com.intel.analytics.bigdl.ppml.fl.{FLContext, NNModel}
import com.intel.analytics.bigdl.ppml.fl.algorithms.{VFLLogisticRegression, VFLLogisticRegressionCkks}
import com.intel.analytics.bigdl.ppml.fl.utils.FlContextForTest
import org.apache.spark.sql.SparkSession

class Client(trainDataPath: String,
             testDataPath: String,
             clientId: Int,
             appName: String) extends Thread {
  override def run(): Unit = {
    FLContext.initFLContext(clientId)
    val sqlContext = SparkSession.builder().getOrCreate()
    val pre = new DataPreprocessing(sqlContext, trainDataPath, testDataPath, clientId)
    val (trainDataset, validationDataset) = pre.loadCensusData()

    val numFeature = if (clientId == 1) {
      3049 - 6 - 1000 - 11 - 9 - 7
    } else {
      6 + 1000 + 11 + 9 + 7
    }

    val linear = if (clientId == 1) {
      SparseLinear[Float](numFeature, 1, withBias = false)
    } else {
      SparseLinear[Float](numFeature, 1, withBias = true)
    }
    linear.getParameters()._1.randn(0, 0.001)

    val lr: NNModel = appName match {
      case "dllib" => new VFLLogisticRegression(numFeature, 0.005f, linear)
      case "ckks" =>
        FLContext.initCkks("ppml/src/main/scala/com/intel/analytics/bigdl/ppml/fl/example/ckks/ckksSecret")
        new VFLLogisticRegression(numFeature, 0.005f, linear, "vfl_logistic_regression_ckks")
      case _ => throw new Error()
    }

    val epochNum = 40
    var accTime: Long = 0
    RandomGenerator.RNG.setSeed(2L)
    (0 until epochNum).foreach { epoch =>
      println(epoch)
      trainDataset.shuffle()
      val trainData = trainDataset.toLocal().data(false)
      var count = 0
      while (trainData.hasNext) {
        println(count)
        val miniBatch = trainData.next()
        val input = miniBatch.getInput()
        val currentBs = input.toTensor[Float].size(1)
        count += currentBs
        val target = miniBatch.getTarget()
        val dllibStart = System.nanoTime()
        lr.trainStep(input, target)
        accTime += System.nanoTime() - dllibStart
      }
      println(s"$appName Time: " + accTime / 1e9)
    }

    linear.evaluate()
    val evalData = validationDataset.toLocal().data(false)
    var accDllib = 0
    while (evalData.hasNext) {
      val miniBatch = evalData.next()
      val input = miniBatch.getInput()
      val currentBs = input.toTensor[Float].size(1)
      val targets = miniBatch.getTarget()
      val predict = lr.predictStep(input)
      println(s"Predicting $appName")
      if (targets != null) {
        val target = targets.toTensor[Float]
        (0 until currentBs).foreach { i =>
          val dllibPre = predict.toTensor[Float].valueAt(i + 1, 1)
          val t = target.valueAt(i + 1, 1)
          if (t == 0) {
            if (dllibPre <= 0.5) {
              accDllib += 1
            }
          } else {
            if (dllibPre > 0.5) {
              accDllib += 1
            }
          }
        }
        //        println(t + " " + dllibPre + " " + ckksPre)
      }
    }
    println(s"$appName predict correct: $accDllib")

  }
}
