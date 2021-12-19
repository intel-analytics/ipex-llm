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


package com.intel.analytics.bigdl.ppml.common

import com.intel.analytics.bigdl.ppml.generated.FlBaseProto._

import scala.collection.JavaConversions._
import scala.collection.JavaConverters._

class AverageAggregator extends Aggregator[Table] {
  protected var modelName = "averaged"

  /**
   * aggregate current temporary model weights and put updated model into storage
   */
  override def aggregate(flPhase: FLPhase): Unit = { // assumed that all the weights is correct !
    // sum and average, then save new weights to storage
    val sumedDataMap = new java.util.HashMap[String, FloatTensor]()
    // sum
    // to do: concurrent hashmap
    val storage = flPhase match {
      case FLPhase.TRAIN => trainStorage
      case FLPhase.EVAL => evalStorage
      case FLPhase.PREDICT => predictStorage
    }
    val dataMap = storage.clientData
    for (model <- dataMap.asScala.values) {
      val modelMap = model.getTableMap

      for (tensorName <- modelMap.keySet) {
        val shapeList = modelMap.get(tensorName).getShapeList
        val dataList = modelMap.get(tensorName).getTensorList
        if (sumedDataMap.get(tensorName) == null) sumedDataMap.put(tensorName, FloatTensor.newBuilder.addAllTensor(dataList).addAllShape(shapeList).build)
        else {
          val shapeListAgg = sumedDataMap.get(tensorName).getShapeList
          val dataListAgg = sumedDataMap.get(tensorName).getTensorList
          val dataListSum = new java.util.ArrayList[java.lang.Float]()
          for (i <- 0 until dataListAgg.size) {
            val temp = dataList.get(i) + dataListAgg.get(i)
            dataListSum.add(temp)
          }
          val FloatTensorAgg = FloatTensor.newBuilder
            .addAllTensor(dataListSum).addAllShape(shapeListAgg).build
          sumedDataMap.put(tensorName, FloatTensorAgg)
        }
      }
    }
    // average
    val averagedDataMap = new java.util.HashMap[String, FloatTensor]
    import scala.collection.JavaConversions._
    for (tensorName <- sumedDataMap.keySet) {
      val shapeList = sumedDataMap.get(tensorName).getShapeList
      val dataList = sumedDataMap.get(tensorName).getTensorList
      val averagedDataList = new java.util.ArrayList[java.lang.Float]
      for (i <- 0 until dataList.size) {
        averagedDataList.add(dataList.get(i) / clientNum)
      }
      val averagedFloatTensor = FloatTensor.newBuilder
        .addAllTensor(averagedDataList).addAllShape(shapeList).build
      averagedDataMap.put(tensorName, averagedFloatTensor)
    }

    val metaData = TableMetaData.newBuilder
      .setName(modelName).setVersion(storage.version + 1).build
    val aggregatedTable = Table.newBuilder.setMetaData(metaData).putAllTable(averagedDataMap).build
    storage.updateStorage(aggregatedTable)
  }
}
