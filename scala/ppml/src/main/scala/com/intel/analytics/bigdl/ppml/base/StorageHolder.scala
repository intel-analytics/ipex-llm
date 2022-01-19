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

package com.intel.analytics.bigdl.ppml.base

import com.intel.analytics.bigdl.ppml.common.{FLDataType, FLPhase, Storage}
import com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto._
import com.intel.analytics.bigdl.ppml.generated.FlBaseProto.TensorMap
import org.apache.logging.log4j.LogManager

/**
 * The storage holder holding all types of Storage
 * Aggregator could use this generic holder type to aggregate
 * @param flDataType the instance of storage of this holder, could be one of [[FLDataType]]
 */
class StorageHolder(flDataType: FLDataType) {
  private val logger = LogManager.getLogger(getClass)

  private var clientDataSize: Int = 0
  private var tensorMapStorage: Storage[TensorMap] = null
  private var treeSplitStorage: Storage[DataSplit] = null
  private var treeLeavesStorage: Storage[TreeLeaves] = null
  private var treeEvalStorage: Storage[java.util.List[BoostEval]] = null

  flDataType match {
    case FLDataType.TENSOR_MAP => tensorMapStorage = new Storage[TensorMap](flDataType.toString)
    case FLDataType.TREE_SPLIT => treeSplitStorage = new Storage[DataSplit](flDataType.toString)
    case FLDataType.TREE_LEAVES => treeLeavesStorage = new Storage[TreeLeaves](flDataType.toString)
    case FLDataType.TREE_EVAL => treeEvalStorage =
      new Storage[java.util.List[BoostEval]](flDataType.toString)
    case _ => throw new NotImplementedError()
  }
  def getVersion(): Int = {
    if (tensorMapStorage != null) tensorMapStorage.version
    else throw new NotImplementedError()
  }

  def getClientDataSize() = this.clientDataSize

  def putClientData(clientID: String, dataHolder: DataHolder) = {
    if (dataHolder.tensorMap != null) {
      tensorMapStorage.clientData.put(clientID, dataHolder.tensorMap)
      clientDataSize = tensorMapStorage.clientData.size()
      logger.debug(s"Put Table into client data map, current size: $clientDataSize")
    } else if (dataHolder.split != null) {
      treeSplitStorage.clientData.put(clientID, dataHolder.split)
      clientDataSize = treeSplitStorage.clientData.size()
      logger.debug(s"Put Split into client data map, current size: $clientDataSize")
    } else if (dataHolder.treeLeaves != null) {
      treeLeavesStorage.clientData.put(clientID, dataHolder.treeLeaves)
      clientDataSize = treeLeavesStorage.clientData.size()
      logger.debug(s"Put TreeLeaves into client data map, current size: $clientDataSize")
    } else if (dataHolder.boostEval != null) {
      treeEvalStorage.clientData.put(clientID, dataHolder.boostEval)
      clientDataSize = treeEvalStorage.clientData.size()
      logger.debug(s"Put TreeEval into client data map, current size: $clientDataSize")
    } else {
      throw new IllegalArgumentException("Data is empty, could not uploaded to server.")
    }
  }
  def getTensorMapStorage() = this.tensorMapStorage
  def getSplitStorage() = this.treeSplitStorage
  def getLeafStorage() = this.treeLeavesStorage
  def getBranchStorage() = this.treeEvalStorage
}
