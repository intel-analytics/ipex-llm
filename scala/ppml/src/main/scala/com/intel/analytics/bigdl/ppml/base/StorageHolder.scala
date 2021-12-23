package com.intel.analytics.bigdl.ppml.base

import com.intel.analytics.bigdl.ppml.common.{FLDataType, Storage}
import com.intel.analytics.bigdl.ppml.generated.FGBoostServiceProto.{BoostEval, DataSplit, TreeLeaves}
import com.intel.analytics.bigdl.ppml.generated.FlBaseProto.Table

/**
 * The storage holder holding all types of Storage
 * Aggregator could use this generic holder type to aggregate
 * @param flDataType the instance of storage of this holder
 */
class StorageHolder(flDataType: FLDataType) {
  private var clientDataSize: Int = 0
  private var tableStorage: Storage[Table] = null
  private var splitStorage: Storage[DataSplit] = null
  private var leafStorage: Storage[TreeLeaves] = null
  private var branchStorage: Storage[List[BoostEval]] = null

  flDataType match {
    case FLDataType.TABLE => tableStorage = new Storage[Table](flDataType.toString)
    case FLDataType.SPLIT => splitStorage = new Storage[DataSplit](flDataType.toString)
    case FLDataType.LEAF => leafStorage = new Storage[TreeLeaves](flDataType.toString)
    case FLDataType.BOOST_EVAL => branchStorage = new Storage[List[BoostEval]](flDataType.toString)
    case _ => throw new NotImplementedError()
  }
  def getVersion(): Int = {
    if (tableStorage != null) tableStorage.version
    else throw new NotImplementedError()
  }

  def getClientDataSize() = this.clientDataSize

  def putClientData(clientID: String, dataHolder: DataHolder) = {
    if (dataHolder.table != null) {
      tableStorage.clientData.put(clientID, dataHolder.table)
      clientDataSize = tableStorage.clientData.size()
    }
  }
  def getTableStorage() = this.tableStorage
  def getSplitStorage() = this.splitStorage
  def getLeafStorage() = this.leafStorage
  def getBranchStorage() = this.branchStorage
}
