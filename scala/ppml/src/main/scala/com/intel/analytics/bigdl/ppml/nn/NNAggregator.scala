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

package com.intel.analytics.bigdl.ppml.nn

import java.util
import java.util.HashMap

import com.intel.analytics.bigdl.ppml.base.StorageHolder
import com.intel.analytics.bigdl.ppml.common.FLPhase.{EVAL, PREDICT, TRAIN}
import com.intel.analytics.bigdl.ppml.common.{Aggregator, FLDataType, FLPhase, Storage}
import com.intel.analytics.bigdl.ppml.generated.FlBaseProto.TensorMap

abstract class NNAggregator extends Aggregator {
  def getStorage(flPhase: FLPhase) = aggregateTypeMap.get(flPhase).getTensorMapStorage()
  protected var shouldReturn = false

  def setShouldReturn(shouldReturn: Boolean): Unit = {
    this.shouldReturn = shouldReturn
  }
  override def initStorage(): Unit = {
    aggregateTypeMap.put(TRAIN, new StorageHolder(FLDataType.TENSOR_MAP))
    aggregateTypeMap.put(EVAL, new StorageHolder(FLDataType.TENSOR_MAP))
    aggregateTypeMap.put(PREDICT, new StorageHolder(FLDataType.TENSOR_MAP))
  }

}
