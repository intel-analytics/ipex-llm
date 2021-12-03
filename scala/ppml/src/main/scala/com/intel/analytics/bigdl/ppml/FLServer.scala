/*
 * Copyright 2021 The BigDL Authors.
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

package com.intel.analytics.bigdl.ppml

import com.intel.analytics.bigdl.grpc.GrpcServerBase
import com.intel.analytics.bigdl.ppml.common.Aggregator
import com.intel.analytics.bigdl.ppml.psi.PSIServiceImpl
import com.intel.analytics.bigdl.ppml.vfl.nn.NNServiceImpl
import com.intel.analytics.bigdl.dllib.nn.BCECriterion
import com.intel.analytics.bigdl.dllib.nn.Sigmoid
import com.intel.analytics.bigdl.dllib.optim.Top1Accuracy
import org.slf4j.Logger
import org.slf4j.LoggerFactory
import java.io.{File, IOException}

import com.intel.analytics.bigdl.ppml.vfl.nn.NNServiceImpl



/**
 * FLServer is BigDL PPML gRPC server used for FL based on GrpcServerBase
 * FLServer starts all the services, e.g. PSIServiceImpl, NNServiceImpl once server starts
 */
object FLServer {
  private val logger = LoggerFactory.getLogger(classOf[FLServer])
  @throws[Exception]
  def main(args: Array[String]): Unit = {
    val flServer = new FLServer(args)
    // Set aggregator here
    flServer.build()
    flServer.start()
    flServer.blockUntilShutdown()
  }
}

class FLServer private[ppml](val _args: Array[String] = null) extends GrpcServerBase(_args) {
  configPath = "ppml-conf.yaml"

  @throws[IOException]
  override def parseConfig(): Unit = {
    val f = new File(configPath)
    if (f.exists()) {
      val flHelper = getConfigFromYaml(classOf[FLHelper], configPath)
      port = flHelper.serverPort
    }
    // start all services without providing service list
    // start all services without providing service list
    serverServices.add(new PSIServiceImpl)
    serverServices.add(new NNServiceImpl)



  }
}
