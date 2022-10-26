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
import com.intel.analytics.bigdl.dllib.feature.dataset.{DataSet, Sample, SampleToMiniBatch, TensorSample}
import com.intel.analytics.bigdl.dllib.keras.metrics.BinaryAccuracy
import com.intel.analytics.bigdl.dllib.nn.{BCECriterion, Sigmoid, SparseLinear}
import com.intel.analytics.bigdl.dllib.optim.{Adagrad, Ftrl, SGD, Top1Accuracy}
import com.intel.analytics.bigdl.dllib.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.dllib.utils.{Engine, RandomGenerator, T}
import com.intel.analytics.bigdl.ppml.fl.algorithms.VFLLogisticRegression
import com.intel.analytics.bigdl.ppml.fl.{FLContext, FLServer, NNModel}
import com.intel.analytics.bigdl.ppml.fl.example.ckks.DataPreprocessing
import io.grpc.netty.shaded.io.netty.handler.codec.http.websocketx.WebSocketClientProtocolHandler.ClientHandshakeStateEvent
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser

import java.util


object VflLogisticRegressionCkks {
  case class CmdArgs(dataPath: String = null,
                     clientId: Int = 1,
                     mode: String = "ckks",
                     secretePath: String = ""
                    )
  val parser = new OptionParser[CmdArgs]("PPML CKKS example") {
    opt[String]('d', "dataPath")
      .text("data path")
      .action((x, c) => c.copy(dataPath = x))
      .required()
    opt[Int]('i', "id")
      .text("client id")
      .action((x, c) => c.copy(clientId = x))
      .required()
    opt[String]('m', "mode")
      .text("ckks or dllib")
      .action((x, c) => c.copy(mode = x))
    opt[String]('s', "secret")
      .text("ckks secret path, not none when mode is ckks")
      .action((x, c) => c.copy(secretePath = x))
  }


  def main(args: Array[String]): Unit = {
    parser.parse(args, CmdArgs()).foreach { param =>

      val inputDir = param.dataPath
      val clientId = param.clientId

      val trainDataPath = s"$inputDir/adult.data"
      val testDataPath = s"$inputDir/adult.test"
      val mode = param.mode
      val ckksSecretPath = param.secretePath

      FLContext.initFLContext(clientId)
      val sqlContext = SparkSession.builder().getOrCreate()
      val pre = new DataPreprocessing(sqlContext, trainDataPath, testDataPath, clientId)
      val (trainDataset, validationDataset) = pre.loadCensusData()

      val numFeature = if (clientId == 1) {
        3049 - 6 - 9 - 7
      } else {
        6 + 9 + 7
      }

      val linear = if (clientId == 1) {
        SparseLinear[Float](numFeature, 1, withBias = false)
      } else {
        SparseLinear[Float](numFeature, 1, withBias = true)
      }
      linear.getParameters()._1.randn(0, 0.001)

      val learningRate = 0.01f
      val lr: NNModel = mode match {
        case "dllib" => new VFLLogisticRegression(numFeature, learningRate, linear)
        case "ckks" =>
          FLContext.initCkks(ckksSecretPath)
          new VFLLogisticRegression(numFeature, learningRate,
            linear, "vfl_logistic_regression_ckks")
        case _ => throw new Error()
      }
      lr.estimator.train(20, trainDataset.toLocal(), null)

      val validationMethod = new BinaryAccuracy[Float]()
      val preditions = lr.estimator.predict(validationDataset.toLocal())

      if (clientId == 2) {
        // client2 has target
        val evalData = validationDataset.toLocal().data(false)
        val result = preditions.toIterator.zip(evalData).map{datas =>
          validationMethod.apply(datas._1, datas._2.getTarget())
        }.reduce(_ + _)

        println(result.toString())
      }

    }
  }

}
