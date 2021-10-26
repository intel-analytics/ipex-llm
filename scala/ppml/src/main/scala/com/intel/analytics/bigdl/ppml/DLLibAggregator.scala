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

import com.intel.analytics.bigdl.dllib.nn.Sequential
import com.intel.analytics.bigdl.dllib.nn.abstractnn.Activity
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.{T, Table}
import com.intel.analytics.bigdl.ppml.vfl.utils.ProtoUtils.toFloatTensor
import com.intel.analytics.bigdl.ppml.common.{Aggregator, FLPhase, Storage}
import com.intel.analytics.bigdl.ppml.generated.FLProto.TableMetaData
import com.intel.analytics.bigdl.ppml.common.FLPhase._
import com.intel.analytics.bigdl.ppml.generated.FLProto
import org.apache.log4j.Logger

import collection.JavaConverters._


trait DLLibAggregator extends Aggregator {
  val logger = Logger.getLogger(this.getClass)
  var module: Sequential[Float] = null
  var target: Tensor[Float] = null
  def protoTableMapToTensorIterableMap(inputMap: java.util.Map[String, FLProto.Table]):
    Map[String, Iterable[Tensor[Float]]] = {
    inputMap.asScala.mapValues(_.getTableMap).values
      .flatMap(_.asScala).groupBy(_._1)
      .map{data =>
        (data._1, data._2.map {v =>
          val data = v._2.getTensorList.asScala.toArray.map(_.toFloat)
          val shape = v._2.getShapeList.asScala.toArray.map(_.toInt)
          Tensor[Float](data, shape)
        })
      }
  }
  def getInputTableFromStorage(storageType: FLPhase): Table = {
    val storage = getServerData(storageType)
    val aggData = protoTableMapToTensorIterableMap(storage.localData)
    target = Tensor[Float]()
    if (aggData.contains("target")) {
      val t = aggData("target").head
      target.resizeAs(t).copy(t)
    }
    // TODO: multiple input
    val outputs = aggData.filter(_._1 != "target")
    require(outputs.size == 1)

    T.seq(outputs.values.head.toSeq)
  }
  def postProcess(aggType: FLPhase, grad: Activity = null, loss: Activity = null): Unit = {
    def updateStorage(storage: Storage, table: FLProto.Table): Unit = {
      storage.localData.clear()
      storage.serverData = table
      storage.version += 1
      logger.info(s"${trainStorage.version} run aggregate successfully: loss is ${loss}")
    }
    val gradProto = if (grad != null) {
      toFloatTensor(grad.toTable.apply[Tensor[Float]](1))
    } else null
    val lossProto = if (loss != null) {
      toFloatTensor(Tensor[Float](loss.toTable))
    } else null
    val metaBuilder = TableMetaData.newBuilder()
    var aggregatedTable: FLProto.Table = null
    if (aggType == TRAIN) {
      val meta = metaBuilder.setName("gradInput").setVersion(trainStorage.version).build()
      aggregatedTable = FLProto.Table.newBuilder()
        .setMetaData(meta)
        .putTable("gradInput", gradProto)
        .putTable("loss", lossProto)
        .build()
    } else if (aggType == EVAL) {
      val meta = metaBuilder.setName("evaluateResult").setVersion(evalStorage.version).build()
      aggregatedTable = FLProto.Table.newBuilder()
        .setMetaData(meta)
        .build()
    } else if (aggType == PREDICT) {
      val meta = metaBuilder.setName("predictResult").setVersion(predictStorage.version).build()
    }
    updateStorage(aggregateTypeMap.get(aggType), aggregatedTable);
  }
}
