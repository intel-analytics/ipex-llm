/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.dataset

import com.intel.analytics.bigdl.utils.Engine

object Utils {

  def getBatchSize(totalBatch : Int): Int = {
    if(Engine.nodeNumber().isDefined) {
      val nodeNumber = Engine.nodeNumber().get
      require(totalBatch % nodeNumber == 0 && totalBatch > nodeNumber
        , "batch size can't be divided by node number or is smaller than node number")
      totalBatch / nodeNumber
    } else {
      totalBatch
    }
  }

}
