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
package com.intel.analytics.bigdl.ppml.fl.example

import com.intel.analytics.bigdl.ckks.CKKS
import com.intel.analytics.bigdl.dllib.NNContext
import com.intel.analytics.bigdl.dllib.feature.dataset.{DataSet, Sample, SampleToMiniBatch, TensorSample}
import com.intel.analytics.bigdl.dllib.nn.{BCECriterion, Sigmoid, SparseLinear}
import com.intel.analytics.bigdl.dllib.optim.{Adagrad, Ftrl, SGD}
import com.intel.analytics.bigdl.dllib.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.dllib.utils.{Engine, RandomGenerator, T}
import com.intel.analytics.bigdl.ppml.fl.FLServer
import com.intel.analytics.bigdl.ppml.fl.example.ckks.Client
import io.grpc.netty.shaded.io.netty.handler.codec.http.websocketx.WebSocketClientProtocolHandler.ClientHandshakeStateEvent
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser

import java.util


object VflLogisticRegressionCkks {
  case class CmdArgs(dataPath: String = null,
                     clientId: Int = 1)
  val parser = new OptionParser[CmdArgs]("PPML CKKS example") {
    opt[String]('d', "dataPath")
      .text("data path")
      .action((x, c) => c.copy(dataPath = x))
      .required()
    opt[Int]('i', "id")
      .text("client id")
      .action((x, c) => c.copy(clientId = x))
      .required()
  }


  def main(args: Array[String]): Unit = {
    parser.parse(args, CmdArgs()).map { param =>
      val inputDir = param.dataPath
      val clientId = param.clientId
      //    val flServer = new FLServer()
      //    flServer.setClientNum(2)
      //    flServer.build()
      //    flServer.start()
      val dllibClient1 = new Client(
        s"$inputDir/adult-${clientId}.data", s"$inputDir/adult-${clientId}.test",
//        clientId, "dllib")
        clientId, "ckks")
      //    val dllibClient2 = new Client(
      //      s"$inputDir/adult-2.data", s"$inputDir/adult-2.test", 2, "dllib")
      dllibClient1.start()
    }
//    dllibClient2.start()

  }

}
