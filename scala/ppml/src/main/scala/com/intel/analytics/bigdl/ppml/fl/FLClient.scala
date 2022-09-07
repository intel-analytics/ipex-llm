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

import com.intel.analytics.bigdl.grpc.GrpcClientBase
import com.intel.analytics.bigdl.ppml.fl.psi.PSIStub
import com.intel.analytics.bigdl.ppml.fl.vfl.{FGBoostStub, NNStub}
import org.apache.logging.log4j.LogManager

import java.io.IOException
import java.util.concurrent.TimeUnit


/**
 * FLClient wraps the gRPC stubs corresponded to gRPC services in FLServer
 * @param _args
 */
class FLClient(val _args: Array[String]) extends GrpcClientBase(_args) {
  val logger = LogManager.getLogger(getClass)
  configPath = "ppml-conf.yaml"
  protected var taskID: String = "taskID"
  var psiStub: PSIStub = null
  var nnStub: NNStub = null
  var fgbostStub: FGBoostStub = null
  var psiSalt: String = null
  privateKeyFilePath = null
  var clientID: Int = 0
  parseConfig()

  def this() {
    this(null)
  }

  def setClientId(clientId: Int): Unit = {
    clientID = clientId
  }

  @throws[IOException]
  override protected def parseConfig(): Unit = {
    val flHelper = getConfigFromYaml(classOf[FLHelper], configPath)
    if (flHelper != null) {
      target = flHelper.clientTarget
      logger.debug(s"Loading target: $target")
      taskID = flHelper.taskID
      logger.debug(s"Loading taskID: $taskID")
      psiSalt = flHelper.psiSalt
      privateKeyFilePath = flHelper.privateKeyFilePath
    }
  }

  override def loadServices(): Unit = {
    psiStub = new PSIStub(channel, clientID)
    nnStub = new NNStub(channel, clientID)
    fgbostStub = new FGBoostStub(channel, clientID)
  }

  override def shutdown(): Unit = {
    try channel.shutdown.awaitTermination(5, TimeUnit.SECONDS)
    catch {
      case e: InterruptedException =>
        logger.error("Shutdown Client Error" + e.getMessage)
    }
  }
}
