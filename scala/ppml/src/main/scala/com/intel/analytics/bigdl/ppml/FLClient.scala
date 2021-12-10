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

import com.intel.analytics.bigdl.grpc.GrpcClientBase
import com.intel.analytics.bigdl.ppml.psi.PSIStub
import com.intel.analytics.bigdl.ppml.vfl.NNStub
import java.io.{File, IOException}
import java.util
import java.util.concurrent.TimeUnit

import org.apache.log4j.Logger


/**
 * FLClient wraps the gRPC stubs corresponded to gRPC services in FLServer
 * @param _args
 */
class FLClient(val _args: Array[String]) extends GrpcClientBase(_args) {
  val logger = Logger.getLogger(getClass)
  configPath = "ppml-conf.yaml"
  protected var taskID: String = "taskID"
  var psiStub: PSIStub = null
  var nnStub: NNStub = null

  def this() {
    this(null)
  }

  @throws[IOException]
  override protected def parseConfig(): Unit = {
    val f = new File(configPath)
    if (f.exists()) {
      val flHelper = getConfigFromYaml(classOf[FLHelper], configPath)
      target = flHelper.clientTarget
      taskID = flHelper.taskID
    }
    super.parseConfig()
  }

  override def loadServices(): Unit = {
    psiStub = new PSIStub(channel, taskID)
    nnStub = new NNStub(channel, clientUUID)
  }

  override def shutdown(): Unit = {
    try channel.shutdown.awaitTermination(5, TimeUnit.SECONDS)
    catch {
      case e: InterruptedException =>
        logger.error("Shutdown Client Error" + e.getMessage)
    }
  }
}
