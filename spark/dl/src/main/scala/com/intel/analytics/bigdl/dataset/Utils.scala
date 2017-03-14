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

  def getBatchSize(batchSize : Int): Int = {
    val nodeNumber = Engine.nodeNumber()
    val coreNumber = Engine.coreNumber()
    logger.debug(s"node number: $nodeNumber, core number: $coreNumber")
    require(batchSize % (nodeNumber * coreNumber) == 0
      , s"batch size($batchSize) can't be divided by node number($nodeNumber) * " +
        s"core number($coreNumber), please change your batch size")

    if (batchSize < nodeNumber * coreNumber * 2) {
      logger.warn(s"Warning: for better training speed, " +
        s"batch size($batchSize) is recommended to be at least two times of node number" +
        s"($nodeNumber) * core number($coreNumber), please tune your batch size accordingly")
    }

    val batchPerUnit = batchSize / nodeNumber
    logger.debug(s"Batch per unit: $batchPerUnit")
    batchPerUnit
  }

}
