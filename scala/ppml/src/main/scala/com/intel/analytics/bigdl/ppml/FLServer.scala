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
import java.io.{File, IOException}

import com.intel.analytics.bigdl.ppml.fgboost.FGBoostServiceImpl
import com.intel.analytics.bigdl.ppml.nn.NNServiceImpl
import org.apache.logging.log4j.{Level, LogManager}
import org.apache.logging.log4j.core.config.Configurator


/**
 * FLServer is BigDL PPML gRPC server used for FL based on GrpcServerBase
 * FLServer starts all the services, e.g. PSIServiceImpl, NNServiceImpl once server starts

 * Supports: PSI, HFL/VFL Logistic Regression / Linear Regression
 */
object FLServer {

  Configurator.setLevel("com.intel.analytics.bigdl.ppml", Level.DEBUG)

  @throws[Exception]
  def main(args: Array[String]): Unit = {
    val flServer = new FLServer(args)
    // Set aggregator here
    flServer.buildWithTls()
    flServer.start()
    flServer.blockUntilShutdown()
  }
}

class FLServer private[ppml](val _args: Array[String] = null) extends GrpcServerBase(_args) {
  private val logger = LogManager.getLogger(classOf[FLServer])
  configPath = "ppml-conf.yaml"
  var clientNum: Int = 1

  def setClientNum(clientNum: Int) = {
    this.clientNum = clientNum
  }
  @throws[IOException]
  override def parseConfig(): Unit = {
    val flHelper = getConfigFromYaml(classOf[FLHelper], configPath)
    if (flHelper != null) {
      port = flHelper.serverPort
      clientNum = flHelper.clientNum
      certChainFilePath = flHelper.certChainFilePath
      privateKeyFilePath = flHelper.privateKeyFilePath
    }

    // start all services without providing service list
    // start all services without providing service list
    serverServices.add(new PSIServiceImpl(clientNum))
    serverServices.add(new NNServiceImpl(clientNum))
    serverServices.add(new FGBoostServiceImpl(clientNum))
  }
}
