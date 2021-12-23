package com.intel.analytics.bigdl.ppml.nn

import java.util
import java.util.HashMap

import com.intel.analytics.bigdl.ppml.base.StorageHolder
import com.intel.analytics.bigdl.ppml.common.FLPhase.{EVAL, PREDICT, TRAIN}
import com.intel.analytics.bigdl.ppml.common.{Aggregator, FLDataType, FLPhase, Storage}
import com.intel.analytics.bigdl.ppml.generated.FlBaseProto.Table

abstract class NNAggregator extends Aggregator {

  override def initStorage(): Unit = {
    aggregateTypeMap.put(TRAIN, new StorageHolder(FLDataType.TABLE))
    aggregateTypeMap.put(EVAL, new StorageHolder(FLDataType.TABLE))
    aggregateTypeMap.put(PREDICT, new StorageHolder(FLDataType.TABLE))
  }

}
