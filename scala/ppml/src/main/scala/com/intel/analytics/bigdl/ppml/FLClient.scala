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
import com.intel.analytics.bigdl.ppml.generated.FLProto
import com.intel.analytics.bigdl.ppml.psi.PSIStub
import com.intel.analytics.bigdl.ppml.vfl.NNStub
import org.slf4j.Logger
import org.slf4j.LoggerFactory
import java.io.IOException
import java.util
import java.util.List
import java.util.concurrent.TimeUnit


object FLClient {
  private val logger = LoggerFactory.getLogger(classOf[FLClient])
}

class FLClient(val _args: Array[String]) extends GrpcClientBase(_args) {
  protected var taskID: String = null
  /**
   * All supported FL implementations are listed below
   * VFL includes Private Set Intersection, Neural Network, Gradient Boosting
   */
  var psiStub: PSIStub = null
  var nnStub: NNStub = null

  def this() {
    this(null)
    build()
  }

  @throws[IOException]
  override protected def parseConfig(): Unit = {
    val flHelper = getConfigFromYaml(classOf[FLHelper], configPath)
    target = flHelper.clientTarget
    taskID = flHelper.taskID
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
        FLClient.logger.error("Shutdown Client Error" + e.getMessage)
    }
  }

  /**
   * Wrap all the api of stubs to expose the API out of the stubs
   */
  // PSI stub
  def getSalt: String = psiStub.getSalt

  def getSalt(name: String, clientNum: Int, secureCode: String): String =
    psiStub.getSalt(name, clientNum, secureCode)

  def uploadSet(hashedIdArray: util.List[String]): Unit = {
    psiStub.uploadSet(hashedIdArray)
  }

  def downloadIntersection(): util.List[String] = psiStub.downloadIntersection

  // NN stub
  def downloadTrain(modelName: String, flVersion: Int): FLProto.DownloadResponse =
    nnStub.downloadTrain(modelName, flVersion)

  def uploadTrain(data: FLProto.Table): FLProto.UploadResponse = nnStub.uploadTrain(data)

  def evaluate(data: FLProto.Table, lastBatch: Boolean): FLProto.EvaluateResponse =
    nnStub.evaluate(data, lastBatch)
}
