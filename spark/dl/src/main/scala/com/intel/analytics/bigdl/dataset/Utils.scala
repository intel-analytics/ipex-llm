/*
 * Copyright 2016 The BigDL Authors.
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

package com.intel.analytics.bigdl.dataset

import com.intel.analytics.bigdl.utils.Engine
import org.apache.log4j.Logger

object Utils {
  private val logger = Logger.getLogger(getClass)

  def getBatchSize(batchSize : Int, totalPartition: Option[Int] = None): Int = {
    val nodeNumber = Engine.nodeNumber()
    val partitionNum = totalPartition.getOrElse(nodeNumber)
    logger.debug(s"partition number: $partitionNum, node number: $nodeNumber")

    require(partitionNum > 0,
      s"Utils.getBatchSize: partitionNum should be larger than 0, but get $partitionNum")
    require(batchSize % partitionNum == 0, s"Utils.getBatchSize: total batch size $batchSize " +
      s"should be divided by partitionNum ${partitionNum}")

    val batchPerUnit = batchSize / partitionNum
    logger.debug(s"Batch per unit: $batchPerUnit")
    batchPerUnit
  }
}
