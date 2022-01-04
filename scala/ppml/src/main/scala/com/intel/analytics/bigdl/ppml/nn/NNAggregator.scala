package com.intel.analytics.bigdl.ppml.nn

import java.util
import java.util.HashMap

import com.intel.analytics.bigdl.ppml.base.StorageHolder
import com.intel.analytics.bigdl.ppml.common.FLPhase.{EVAL, PREDICT, TRAIN}
import com.intel.analytics.bigdl.ppml.common.{Aggregator, FLDataType, FLPhase, Storage}
import com.intel.analytics.bigdl.ppml.generated.FlBaseProto.Table

abstract class NNAggregator extends Aggregator {
  protected var hasReturn = false

  def setHasReturn(hasReturn: Boolean): Unit = {
    this.hasReturn = hasReturn
  }
  def getStorage(flPhase: FLPhase) = aggregateTypeMap.get(flPhase).getTableStorage()
  override def initStorage(): Unit = {
    aggregateTypeMap.put(TRAIN, new StorageHolder(FLDataType.TENSOR_MAP))
    aggregateTypeMap.put(EVAL, new StorageHolder(FLDataType.TENSOR_MAP))
    aggregateTypeMap.put(PREDICT, new StorageHolder(FLDataType.TENSOR_MAP))
  }

}
