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

package com.intel.analytics.bigdl.ppml.vfl.nn

import com.intel.analytics.bigdl.dllib.nn.{CAddTable, Sequential}
import com.intel.analytics.bigdl.dllib.optim.{OptimMethod, ValidationMethod}
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.T
import com.intel.analytics.bigdl.ppml.common.FLPhase
import com.intel.analytics.bigdl.ppml.common.FLPhase._
import com.intel.analytics.bigdl.ppml.generated.FlBaseProto._
import com.intel.analytics.bigdl.ppml.vfl.DLlibAggregator
import com.intel.analytics.bigdl.ppml.utils.ProtoUtils.toFloatTensor
import com.intel.analytics.bigdl.{Criterion, Module}
import org.apache.logging.log4j.LogManager


/**
 *
 * @param model
 * @param optimMethod
 * @param criterion loss function, HFL takes loss at estimator, VFL takes loss at aggregator
 * @param validationMethods
 */
class VflNNAggregator(model: Module[Float],
                      optimMethod: OptimMethod[Float],
                      criterion: Criterion[Float],
                      validationMethods: Array[ValidationMethod[Float]]) extends DLlibAggregator{
  module = Sequential[Float]().add(CAddTable[Float]())
  if (model != null) {
    module.add(model)
  }

  override def aggregate(aggType: FLPhase): Unit = {
    val inputTable = getInputTableFromStorage(TRAIN)

    val output = module.forward(inputTable)

    val metaBuilder = TableMetaData.newBuilder()
    var aggregatedTable: Table = null
    if (aggType == FLPhase.TRAIN) {
      val loss = criterion.forward(output, target)
      val gradOutputLayer = criterion.backward(output, target)
      val grad = module.backward(inputTable, gradOutputLayer)
      val meta = metaBuilder.setName("gradInput").setVersion(trainStorage.version).build()

      aggregatedTable = Table.newBuilder()
        .setMetaData(meta)
        .putTable("gradInput", toFloatTensor(Tensor[Float](grad.toTable)))
        .putTable("loss", toFloatTensor(Tensor[Float](T(loss))))
        .build()
    } else if (aggType == EVAL) {
      val meta = metaBuilder.setName("evaluateResult").setVersion(evalStorage.version).build()
      aggregatedTable = Table.newBuilder()
        .setMetaData(meta)
        .putTable("evaluateOutput", toFloatTensor(Tensor[Float](output.toTable)))
        .build()
    } else if (aggType == PREDICT) {
      val meta = metaBuilder.setName("predictResult").setVersion(predictStorage.version).build()
      aggregatedTable = Table.newBuilder()
        .setMetaData(meta)
        .putTable("predictOutput", toFloatTensor(Tensor[Float](output.toTable)))
        .build()
    }
    aggregateTypeMap.get(aggType).updateStorage(aggregatedTable)
  }

}

object VflNNAggregator {
  val logger = LogManager.getLogger(this.getClass)

  def apply(clientNum: Int,
            classifier: Module[Float],
            optimMethod: OptimMethod[Float],
            criterion: Criterion[Float]): VflNNAggregator = {
    val vflNNAggregator = new VflNNAggregator(classifier, optimMethod, criterion, null)
    vflNNAggregator.setClientNum(clientNum)
    vflNNAggregator
  }

  def apply(clientNum: Int,
            classifier: Module[Float],
            optimMethod: OptimMethod[Float],
            criterion: Criterion[Float],
            validationMethods: Array[ValidationMethod[Float]]): VflNNAggregator = {
    val vflNNAggregator = new VflNNAggregator(
      classifier, optimMethod, criterion, validationMethods)
    vflNNAggregator.setClientNum(clientNum)
    vflNNAggregator
  }
}
