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

package com.intel.analytics.bigdl.ppml.fl

import com.intel.analytics.bigdl.grpc.GrpcServerBase
import com.intel.analytics.bigdl.ppml.fl.fgboost.FGBoostServiceImpl
import com.intel.analytics.bigdl.ppml.fl.nn.NNServiceImpl
import com.intel.analytics.bigdl.ppml.fl.psi.PSIServiceImpl
import org.apache.logging.log4j.core.config.Configurator
import org.apache.logging.log4j.{Level, LogManager}

import java.io.IOException


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
  val fgBoostConfig = new FLConfig()
  parseConfig()

  def setClientNum(clientNum: Int): Unit = {
    this.clientNum = clientNum
  }
  @throws[IOException]
  override def parseConfig(): Unit = {
    val flHelper = getConfigFromYaml(classOf[FLHelper], configPath)
    // overwrite the current config if there exists in config file
    if (flHelper != null) {
      port = flHelper.serverPort
      clientNum = flHelper.clientNum
      certChainFilePath = flHelper.certChainFilePath
      privateKeyFilePath = flHelper.privateKeyFilePath
      fgBoostConfig.setModelPath(flHelper.fgBoostServerModelPath)
    }
  }

  override def build(): Unit = {
    addService()
    super.build()
  }

  override def buildWithTls(): Unit = {
    addService()
    super.buildWithTls()
  }
  def addService(): Unit = {
    serverServices.add(new PSIServiceImpl(clientNum))
    serverServices.add(new NNServiceImpl(clientNum))
    serverServices.add(new FGBoostServiceImpl(clientNum, fgBoostConfig))
  }
}
