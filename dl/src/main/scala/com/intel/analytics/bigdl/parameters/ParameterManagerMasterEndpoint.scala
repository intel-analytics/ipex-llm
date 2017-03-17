/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.sparkExtension

import java.util.HashMap

import org.apache.spark.{SparkConf, SparkContext, SparkEnv}
import org.apache.spark.rpc._
import org.apache.spark.storage.{BlockId, BlockManagerId}
import org.apache.spark.util.{RpcUtils, Utils}
import org.apache.spark.SecurityManager

import scala.collection.mutable

case class GetExecutorBlockList(executorId: Int)
case class UpdateExecutorBlockList(executorId: Int, blockId: BlockId)
case class ClearExecutorBlockList(executorId: Int)

class ParameterManagerMasterEndpoint(
  override val rpcEnv: RpcEnv,
  conf: SparkConf)
  extends ThreadSafeRpcEndpoint {

  private val blocks = new HashMap[Int, mutable.HashSet[BlockId]]
  
  override def receiveAndReply(context: RpcCallContext): PartialFunction[Any, Unit] = {
    case GetExecutorBlockList(executorId) =>
      context.reply(getExecutorBlockList(executorId))

    case UpdateExecutorBlockList(executorId, blockId) =>
      context.reply(updateExecutorBlockList(executorId, blockId))

    case ClearExecutorBlockList(executorId) =>
      context.reply(clearExecutorBlockList(executorId))
  }

  private def getExecutorBlockList(executorId: Int): Seq[BlockId] = {
    if (blocks.containsKey(executorId)) {
      blocks.get(executorId).toSeq
    } else Seq.empty
  }

  private def updateExecutorBlockList(executorId: Int, blockId: BlockId): Unit = {
    if (blocks.containsKey(executorId)) blocks.get(executorId).add(blockId)
    else {
      val hashset = new mutable.HashSet[BlockId]()
      hashset.add(blockId)
      blocks.put(executorId, hashset)
    }
  }

  private def clearExecutorBlockList(executorId: Int) = {
    if (blocks.containsKey(executorId)) blocks.get(executorId).clear()
  }
}

class ParameterManagerMaster(
  var driverEndpoint: RpcEndpointRef,
  isDriver: Boolean)
{
  def getBlockId(executorId: Int): Seq[BlockId] = {
    driverEndpoint.askWithRetry[Seq[BlockId]](GetExecutorBlockList(executorId))
  }
 
  def updateBlockId(executorId: Int, blockId: BlockId): Unit = {
    driverEndpoint.askWithRetry[Unit](UpdateExecutorBlockList(executorId, blockId))
  }

  def clearBlockId(executorId: Int): Unit = {
    driverEndpoint.askWithRetry[Unit](ClearExecutorBlockList(executorId))
  }
}

object ParameterManagerMaster {
  def createEnv(conf: SparkConf, isDriver: Boolean): ParameterManagerMaster = {
    val bindAddress = Utils.localHostName()
    val port = conf.getInt("BigDL.port", 7777)
    val systemName = if (isDriver) "BigDLDriver" else "BigDLExecutor"
    val rpcEnv = RpcEnvWrapper.create(systemName, bindAddress, port, conf,
      new SecurityManager(conf), clientMode = !isDriver)

    if (isDriver) {
      conf.set("BigDL.driver.port", rpcEnv.address.port.toString)
    }
    
    def registerOrLookupEndpoint(name: String, isDriver: Boolean, endpointCreator: => RpcEndpoint):
    RpcEndpointRef = {
      if (isDriver) {
        RpcEnvWrapper.setupEndpoint(rpcEnv, name, endpointCreator)
      } else {
        val driverHost = SparkEnv.get.blockManager.master.driverEndpoint.address.host
        val driverPort = conf.getInt("BigDL.driver.port", 7777)
        RpcEnvWrapper.setupEndpointRef(rpcEnv, systemName, RpcAddress(driverHost, driverPort), name)
      }
    }

    new ParameterManagerMaster(registerOrLookupEndpoint(
      "ParameterManagerMaster", isDriver,
      new ParameterManagerMasterEndpoint(rpcEnv, conf)), isDriver)
  }
}
