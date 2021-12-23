package com.intel.analytics.bigdl.ppml.nn

import java.util
import java.util.HashMap

import com.intel.analytics.bigdl.ppml.common.FLPhase.{EVAL, PREDICT, TRAIN}
import com.intel.analytics.bigdl.ppml.common.{Aggregator, FLPhase, Storage}
import com.intel.analytics.bigdl.ppml.generated.FlBaseProto.Table

abstract class NNAggregator extends Aggregator[Table] {
  override def initStorage(): Unit = {
    aggregateTypeMap = new util.HashMap[FLPhase, Storage[Table]]
    aggregateTypeMap.put(TRAIN, new Storage[Table]("train"))
    aggregateTypeMap.put(EVAL, new Storage[Table]("eval"))
    aggregateTypeMap.put(PREDICT, new Storage[Table]("predict"))
  }

}
